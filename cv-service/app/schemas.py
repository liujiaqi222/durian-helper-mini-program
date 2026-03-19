"""Shared request and response schemas for the detection API."""

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    """A rectangular region produced by the detector."""

    x1: int = Field(..., description="Left boundary in pixels.")
    y1: int = Field(..., description="Top boundary in pixels.")
    x2: int = Field(..., description="Right boundary in pixels.")
    y2: int = Field(..., description="Bottom boundary in pixels.")


class DetectionItem(BaseModel):
    """A single durian candidate returned by YOLO."""

    label: str = Field(..., description="Stable human-readable label, such as A or B.")
    class_name: str = Field(..., description="Predicted class name.")
    confidence: float = Field(..., description="Prediction confidence in [0, 1].")
    bbox: BoundingBox
    crop_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded crop image for the detected durian.",
    )


class DetectionResponse(BaseModel):
    """Top-level response returned by the detection endpoint."""

    count: int = Field(..., description="Number of detected durians.")
    items: list[DetectionItem] = Field(..., description="All accepted detection boxes.")
    message: str | None = Field(
        default=None,
        description="Optional business message, such as when no durians are detected.",
    )
    annotated_image_base64: str | None = Field(
        default=None,
        description="Base64-encoded annotated full image.",
    )


class ModelInfoResponse(BaseModel):
    """Metadata about the currently loaded YOLO model."""

    model_path: str = Field(..., description="Resolved filesystem path to the loaded model.")
    target_class_name: str = Field(..., description="Only this class is returned to callers.")
    confidence_threshold: float = Field(..., description="Current prediction threshold.")
    supported_formats: list[str] = Field(..., description="Accepted upload image formats.")
    max_upload_size_bytes: int = Field(..., description="Maximum accepted image payload size.")
    min_image_width: int = Field(..., description="Minimum accepted image width in pixels.")
