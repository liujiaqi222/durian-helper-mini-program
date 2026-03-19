"""FastAPI entrypoint for the durian YOLO microservice."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile

from app.services.detector import DurianDetector
from app.schemas import DetectionResponse, ModelInfoResponse


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


@app.post("/detect", response_model=DetectionResponse, response_model_exclude_none=True)
async def detect(
    file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
) -> DetectionResponse:
    """Detect durians from an uploaded image.

    Args:
        file: Image file uploaded by the caller.
        image_url: Optional remote image URL. Exactly one input is required.

    Returns:
        A normalized list of durian bounding boxes and confidences.
    """
    return await detector.detect(upload=file, image_url=image_url, include_assets=False)


@app.post(
    "/detect-and-annotate",
    response_model=DetectionResponse,
    response_model_exclude_none=True,
)
async def detect_and_annotate(
    file: UploadFile | None = File(default=None),
    image_url: str | None = Form(default=None),
) -> DetectionResponse:
    """Detect durians and return annotation artifacts in one call."""
    return await detector.detect(upload=file, image_url=image_url, include_assets=True)


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info() -> ModelInfoResponse:
    """Return model metadata relevant to callers and operators."""
    return detector.get_model_info()
