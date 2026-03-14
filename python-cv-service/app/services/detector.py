"""YOLO detector service wrapper.

This module isolates model loading and prediction so the API layer stays small.
When the team later swaps weights, thresholds, or even the detection library,
most changes should stay inside this file.
"""

from __future__ import annotations

from io import BytesIO

from fastapi import HTTPException, UploadFile
from PIL import Image
from ultralytics import YOLO

from app.config import CONFIDENCE_THRESHOLD, MODEL_PATH, TARGET_CLASS_NAME
from app.schemas import BoundingBox, DetectionItem, DetectionResponse


class DurianDetector:
    """Wrap YOLO inference and normalize raw outputs into API-friendly data."""

    def __init__(self) -> None:
        """Load model metadata once during service startup."""
        self._model_path = MODEL_PATH
        self._target_class_name = TARGET_CLASS_NAME
        self._model: YOLO | None = None

    def load(self) -> None:
        """Load the YOLO model into memory.

        Raises:
            RuntimeError: If the model file does not exist or loading fails.
        """
        if not self._model_path.exists():
            raise RuntimeError(
                f"YOLO model file was not found: {self._model_path}. "
                "Train a durian model first, then place best.pt in the models directory."
            )

        self._model = YOLO(str(self._model_path))

    async def detect_upload(self, upload: UploadFile) -> DetectionResponse:
        """Run detection for an uploaded image file."""
        if self._model is None:
            raise RuntimeError("Detector has not been loaded.")

        image_bytes = await upload.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded image is empty.")

        try:
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:  # pragma: no cover - depends on invalid user files
            raise HTTPException(status_code=400, detail="Invalid image file.") from exc

        results = self._model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return self._build_response(results[0])

    def _build_response(self, result) -> DetectionResponse:
        """Convert one Ultralytics result object into the public response schema."""
        items: list[DetectionItem] = []

        # Ultralytics returns parallel arrays for coordinates, scores, and classes.
        # We normalize them here so the rest of the backend never depends on the
        # library's internal data structures.
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            if class_name != self._target_class_name:
                continue

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
            items.append(
                DetectionItem(
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                )
            )

        return DetectionResponse(count=len(items), items=items)
