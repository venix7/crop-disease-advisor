import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Path to a tomato leaf image")
    args = parser.parse_args()

    config = json.loads(
        (ROOT / "config" / "project.json").read_text(encoding="utf-8")
    )
    classes = config["classes"]

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(
        torch.load(
            ROOT / "models" / "tomato_resnet18.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    model.eval()

    image = Image.open(args.image).convert("RGB")
    tensor = ResNet18_Weights.IMAGENET1K_V1.transforms()(image).unsqueeze(0)

    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0]
        score, index = probabilities.max(dim=0)

    print(f"Prediction: {classes[index.item()]['name']}")
    print(f"Model score: {score.item():.3f}")


if __name__ == "__main__":
    main()