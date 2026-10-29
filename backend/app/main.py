import json
from pathlib import Path

from fastapi import FastAPI

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "project.json"

with CONFIG_PATH.open(encoding="utf-8") as file:
    project_config = json.load(file)

app = FastAPI(title="Crop Disease Advisor API")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/classes")
def get_classes():
    return {
        "crop": project_config["crop"],
        "classes": project_config["classes"],
    }