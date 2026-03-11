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

    class_name: str = Field(..., description="Predicted class name.")
    confidence: float = Field(..., description="Prediction confidence in [0, 1].")
    bbox: BoundingBox


class DetectionResponse(BaseModel):
    """Top-level response returned by the detection endpoint."""

    count: int = Field(..., description="Number of detected durians.")
    items: list[DetectionItem] = Field(..., description="All accepted detection boxes.")
