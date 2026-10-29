import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from huggingface_hub import hf_hub_download
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "config" / "project.json").read_text(encoding="utf-8"))
CLASS_IDS = [item["id"] for item in CONFIG["classes"]]

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 1e-3

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class LeafDataset(Dataset):
    def __init__(self, dataset, class_to_index, transform):
        self.dataset = dataset
        self.class_to_index = class_to_index
        self.transform = transform
        self.original_names = dataset.features["label"].names

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        row = self.dataset[index]
        class_name = self.original_names[row["label"]]
        image = row["image"].convert("RGB")
        label = self.class_to_index[class_name]
        return self.transform(image), label


def select_classes(dataset, allowed):
    names = dataset.features["label"].names
    indices = [
        index
        for index, label in enumerate(dataset["label"])
        if names[label] in allowed
    ]
    return dataset.select(indices)


def split_by_leaf(dataset):
    """Make 70% train, 15% validation, 15% test splits by leaf ID."""
    groups = dataset["leaf_id"]

    if any(not leaf_id for leaf_id in groups):
        raise ValueError("Dataset contains an empty leaf_id")

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15,
        random_state=SEED,
    )
    train_val_indices, test_indices = next(
        first_split.split(np.zeros(len(dataset)), groups=groups)
    )

    train_val = dataset.select(train_val_indices.tolist())
    test = dataset.select(test_indices.tolist())

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.85,
        random_state=SEED + 1,
    )
    train_indices, val_indices = next(
        second_split.split(
            np.zeros(len(train_val)),
            groups=train_val["leaf_id"],
        )
    )

    train = train_val.select(train_indices.tolist())
    val = train_val.select(val_indices.tolist())

    train_groups = set(train["leaf_id"])
    val_groups = set(val["leaf_id"])
    test_groups = set(test["leaf_id"])

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    return train, val, test


def evaluate(model, loader, device):
    model.eval()
    actual = []
    predicted = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu().tolist()

            actual.extend(labels.tolist())
            predicted.extend(predictions)

    macro_f1 = f1_score(
        actual,
        predicted,
        labels=list(range(len(CLASS_IDS))),
        average="macro",
        zero_division=0,
    )
    return actual, predicted, macro_f1


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("Loading PlantVillage...")

    loader_script = hf_hub_download(
        repo_id=CONFIG["dataset"]["id"],
        filename="plant_village.py",
        repo_type="dataset",
    )
    dataset = load_dataset(
        loader_script,
        trust_remote_code=True,
    )

    # Combine the source splits before creating our own leaf-grouped splits.
    all_images = concatenate_datasets(
        [dataset["train"], dataset["test"]]
    )
    selected = select_classes(all_images, set(CLASS_IDS))

    available = {
        selected.features["label"].names[label]
        for label in selected["label"]
    }
    if available != set(CLASS_IDS):
        raise ValueError(
            f"Dataset classes do not match config. Found: {available}"
        )

    train, val, test = split_by_leaf(selected)

    for split_name, split in (
        ("train", train),
        ("validation", val),
        ("test", test),
    ):
        present = {
            split.features["label"].names[label]
            for label in split["label"]
        }
        missing = set(CLASS_IDS) - present
        if missing:
            raise ValueError(f"{split_name} is missing classes: {missing}")
        print(f"{split_name}: {len(split)} images")

    class_to_index = {
        class_id: index for index, class_id in enumerate(CLASS_IDS)
    }

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
        ]
    )
    eval_transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

    train_loader = DataLoader(
        LeafDataset(train, class_to_index, train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        LeafDataset(val, class_to_index, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        LeafDataset(test, class_to_index, eval_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Training on: {device}")

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.fc = nn.Linear(model.fc.in_features, len(CLASS_IDS))
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE,
    )
    loss_fn = nn.CrossEntropyLoss()

    best_f1 = -1.0
    model_path = MODEL_DIR / "tomato_resnet18.pt"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)

        _, _, val_f1 = evaluate(model, val_loader, device)
        train_loss = total_loss / len(train)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"train loss: {train_loss:.4f} | "
            f"validation macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), model_path)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=device,
            weights_only=True,
        )
    )

    actual, predicted, test_f1 = evaluate(
        model,
        test_loader,
        device,
    )

    report = {
        "dataset": CONFIG["dataset"],
        "architecture": "resnet18",
        "split_method": "GroupShuffleSplit by leaf_id after combining source splits",
        "class_ids": CLASS_IDS,
        "train_images": len(train),
        "validation_images": len(val),
        "test_images": len(test),
        "best_validation_macro_f1": best_f1,
        "test_macro_f1": test_f1,
        "classification_report": classification_report(
            actual,
            predicted,
            labels=list(range(len(CLASS_IDS))),
            target_names=CLASS_IDS,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=list(range(len(CLASS_IDS))),
        ).tolist(),
    }

    report_path = REPORT_DIR / "phase2_metrics.json"
    report_path.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {report_path}")


if __name__ == "__main__":
    main()