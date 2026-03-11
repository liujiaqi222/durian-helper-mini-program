"""FastAPI entrypoint for the durian YOLO microservice."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile

from app.services.detector import DurianDetector
from app.schemas import DetectionResponse


detector = DurianDetector()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the YOLO model once during startup so requests stay lightweight."""
    detector.load()
    yield


app = FastAPI(
    title="Durian CV Service",
    version="0.1.0",
    description="Detect durians in an image with a YOLO model.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict[str, str]:
    """Return a minimal health response for liveness checks."""
    return {"status": "ok"}


@app.post("/detect", response_model=DetectionResponse)
async def detect(file: UploadFile = File(...)) -> DetectionResponse:
    """Detect durians from an uploaded image.

    Args:
        file: Image file uploaded by the caller.

    Returns:
        A normalized list of durian bounding boxes and confidences.
    """
    return await detector.detect_upload(file)
