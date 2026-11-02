import json
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "project.json"
MODEL_PATH = ROOT / "models" / "tomato_resnet18.pt"


class LeafClassifier:
    def __init__(self):
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Trained model not found: {MODEL_PATH}. Complete phase 2 first."
            )

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        self.classes = config["classes"]
        self.transform = ResNet18_Weights.IMAGENET1K_V1.transforms()

        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            len(self.classes),
        )
        self.model.load_state_dict(
            torch.load(
                MODEL_PATH,
                map_location="cpu",
                weights_only=True,
            )
        )
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        tensor = self.transform(image.convert("RGB")).unsqueeze(0)

        with torch.inference_mode():
            probabilities = self.model(tensor).softmax(dim=1)[0]
            score, index = probabilities.max(dim=0)

        selected_class = self.classes[index.item()]
        return {
            "class_id": selected_class["id"],
            "display_name": selected_class["name"],
            "model_score": round(score.item(), 4),
        }