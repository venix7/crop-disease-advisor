import json
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import select
from sqlalchemy.orm import Session

from .classifier import LeafClassifier
from .database import Base, Diagnosis, engine, get_db
from .schemas import DiagnosisResponse


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "project.json"
UPLOAD_DIR = ROOT / "data" / "uploads"

MAX_FILE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000

project_config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(bind=engine)
    app.state.classifier = LeafClassifier()
    yield


app = FastAPI(
    title="Crop Disease Advisor API",
    lifespan=lifespan,
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/classes")
def get_classes():
    return {
        "crop": project_config["crop"],
        "classes": project_config["classes"],
    }


@app.post(
    "/api/diagnoses",
    response_model=DiagnosisResponse,
    status_code=201,
)
def create_diagnosis(
    request: Request,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    if file.content_type not in (
        "image/jpeg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, or WebP image.",
        )

    raw = file.file.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Image must be 5 MB or smaller.",
        )

    try:
        with Image.open(BytesIO(raw)) as opened:
            if opened.format not in ("JPEG", "PNG", "WEBP"):
                raise ValueError("Unsupported image format")

            if opened.width * opened.height > MAX_IMAGE_PIXELS:
                raise HTTPException(
                    status_code=413,
                    detail="Image dimensions are too large.",
                )

            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")

    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    prediction = request.app.state.classifier.predict(image)

    diagnosis_id = str(uuid4())
    image_path = UPLOAD_DIR / f"{diagnosis_id}.jpg"
    image.save(image_path, format="JPEG", quality=85)

    diagnosis = Diagnosis(
        id=diagnosis_id,
        crop=project_config["crop"],
        class_id=prediction["class_id"],
        display_name=prediction["display_name"],
        model_score=prediction["model_score"],
        status="predicted",
        image_path=str(image_path),
    )

    try:
        db.add(diagnosis)
        db.commit()
        db.refresh(diagnosis)
    except Exception:
        db.rollback()
        image_path.unlink(missing_ok=True)
        raise

    return diagnosis


@app.get("/api/diagnoses", response_model=list[DiagnosisResponse])
def list_diagnoses(db: Session = Depends(get_db)):
    statement = (
        select(Diagnosis)
        .order_by(Diagnosis.created_at.desc())
        .limit(20)
    )
    return db.scalars(statement).all()


@app.get(
    "/api/diagnoses/{diagnosis_id}",
    response_model=DiagnosisResponse,
)
def get_diagnosis(
    diagnosis_id: str,
    db: Session = Depends(get_db),
):
    diagnosis = db.get(Diagnosis, diagnosis_id)
    if diagnosis is None:
        raise HTTPException(
            status_code=404,
            detail="Diagnosis not found.",
        )
    return diagnosis