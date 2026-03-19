"""YOLO detector service wrapper.

This module isolates model loading and prediction so the API layer stays small.
When the team later swaps weights, thresholds, or even the detection library,
most changes should stay inside this file.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass
from io import BytesIO
from statistics import median
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

from app.config import (
    ALLOWED_IMAGE_FORMATS,
    CONFIDENCE_THRESHOLD,
    MAX_UPLOAD_SIZE_BYTES,
    MIN_IMAGE_WIDTH,
    MODEL_PATH,
    TARGET_CLASS_NAME,
)
from app.schemas import BoundingBox, DetectionItem, DetectionResponse, ModelInfoResponse
from app.services.validators import load_and_validate_image


@dataclass(slots=True)
class RawDetection:
    """Internal normalized detection object before label assignment."""

    class_name: str
    confidence: float
    bbox: BoundingBox

    @property
    def center_x(self) -> float:
        return (self.bbox.x1 + self.bbox.x2) / 2

    @property
    def center_y(self) -> float:
        return (self.bbox.y1 + self.bbox.y2) / 2

    @property
    def height(self) -> int:
        return self.bbox.y2 - self.bbox.y1


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

    async def detect_upload(
        self,
        upload: UploadFile,
        *,
        include_assets: bool = False,
        apply_annotation_filter: bool = False,
    ) -> DetectionResponse:
        """Run detection for an uploaded image file."""
        if self._model is None:
            raise RuntimeError("Detector has not been loaded.")

        image_bytes = await upload.read()
        image = load_and_validate_image(image_bytes)
        return self._predict_image(
            image,
            include_assets=include_assets,
            apply_annotation_filter=apply_annotation_filter,
        )

    async def detect_url(
        self,
        image_url: str,
        *,
        include_assets: bool = False,
        apply_annotation_filter: bool = False,
    ) -> DetectionResponse:
        """Run detection for a remotely hosted image."""
        try:
            parsed = urlparse(image_url)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="Invalid image_url.") from exc

        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise HTTPException(status_code=400, detail="image_url must be a valid http or https URL.")

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
                response = await client.get(image_url)
                response.raise_for_status()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=400, detail="Failed to download image_url.") from exc

        image = load_and_validate_image(response.content)
        return self._predict_image(
            image,
            include_assets=include_assets,
            apply_annotation_filter=apply_annotation_filter,
        )

    async def detect(
        self,
        *,
        upload: UploadFile | None = None,
        image_url: str | None = None,
        include_assets: bool = False,
        apply_annotation_filter: bool = False,
    ) -> DetectionResponse:
        """Accept either file upload or remote URL input."""
        if bool(upload) == bool(image_url):
            raise HTTPException(status_code=400, detail="Provide exactly one of file or image_url.")

        if upload is not None:
            return await self.detect_upload(
                upload,
                include_assets=include_assets,
                apply_annotation_filter=apply_annotation_filter,
            )

        return await self.detect_url(
            image_url or "",
            include_assets=include_assets,
            apply_annotation_filter=apply_annotation_filter,
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Return public model metadata."""
        return ModelInfoResponse(
            model_path=str(self._model_path),
            target_class_name=self._target_class_name,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            supported_formats=sorted(ALLOWED_IMAGE_FORMATS),
            max_upload_size_bytes=MAX_UPLOAD_SIZE_BYTES,
            min_image_width=MIN_IMAGE_WIDTH,
        )

    def _predict_image(
        self,
        image: Image.Image,
        *,
        include_assets: bool,
        apply_annotation_filter: bool = False,
    ) -> DetectionResponse:
        """Run YOLO on a validated image and build the public response."""
        assert self._model is not None  # Guarded by public entrypoints.
        results = self._model.predict(image, conf=CONFIDENCE_THRESHOLD, verbose=False)
        return self._build_response(
            results[0],
            image=image,
            include_assets=include_assets,
            apply_annotation_filter=apply_annotation_filter,
        )

    def _build_response(
        self,
        result,
        *,
        image: Image.Image | None = None,
        include_assets: bool = False,
        apply_annotation_filter: bool = False,
    ) -> DetectionResponse:
        """Convert one Ultralytics result object into the public response schema."""
        raw_items: list[RawDetection] = []

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
            raw_items.append(
                RawDetection(
                    class_name=class_name,
                    confidence=round(confidence, 4),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                ),
            )

        message: str | None = None
        if apply_annotation_filter:
            raw_items, message = self._filter_for_annotation(raw_items)

        items = self._assign_labels(raw_items)
        annotated_image_base64: str | None = None

        if include_assets and items:
            if image is None:
                raise ValueError("image is required when include_assets is True.")
            items = self._attach_crop_images(image, items)
            annotated_image_base64 = self._encode_image_base64(self._annotate_image(image, items))

        return DetectionResponse(
            count=len(items),
            items=items,
            message=message,
            annotated_image_base64=annotated_image_base64,
        )

    def _filter_for_annotation(
        self,
        items: list[RawDetection],
    ) -> tuple[list[RawDetection], str | None]:
        """Apply the business filter used by /detect-and-annotate."""
        high_confidence_items = [item for item in items if item.confidence > 0.6]
        if high_confidence_items:
            return self._top_by_confidence(high_confidence_items, limit=9), None

        fallback_items = [item for item in items if item.confidence > 0.4]
        if fallback_items:
            return self._top_by_confidence(fallback_items, limit=3), None

        return [], "没有识别到榴莲"

    def _top_by_confidence(
        self,
        items: list[RawDetection],
        *,
        limit: int,
    ) -> list[RawDetection]:
        """Pick the strongest detections while keeping tie-breakers deterministic."""
        return sorted(
            items,
            key=lambda item: (-item.confidence, item.center_y, item.center_x),
        )[:limit]

    def _assign_labels(self, items: list[RawDetection]) -> list[DetectionItem]:
        """Sort detections by rows and assign stable alphabetical labels."""
        if not items:
            return []

        sorted_by_position = sorted(items, key=lambda item: (item.center_y, item.center_x))
        row_threshold = self._row_threshold(sorted_by_position)

        rows: list[list[RawDetection]] = []
        row_centers: list[float] = []
        for item in sorted_by_position:
            if not rows:
                rows.append([item])
                row_centers.append(item.center_y)
                continue

            if math.fabs(item.center_y - row_centers[-1]) <= row_threshold:
                rows[-1].append(item)
                row_centers[-1] = sum(entry.center_y for entry in rows[-1]) / len(rows[-1])
            else:
                rows.append([item])
                row_centers.append(item.center_y)

        ordered_items: list[DetectionItem] = []
        label_index = 0
        for row in rows:
            for item in sorted(row, key=lambda entry: entry.center_x):
                ordered_items.append(
                    DetectionItem(
                        label=self._index_to_label(label_index),
                        class_name=item.class_name,
                        confidence=item.confidence,
                        bbox=item.bbox,
                    ),
                )
                label_index += 1

        return ordered_items

    def _row_threshold(self, items: list[RawDetection]) -> float:
        """Derive a row grouping threshold from box heights."""
        heights = [max(item.height, 1) for item in items]
        return max(24.0, median(heights) * 0.6)

    def _attach_crop_images(
        self,
        image: Image.Image,
        items: list[DetectionItem],
    ) -> list[DetectionItem]:
        """Attach base64 crop images to response items."""
        enriched_items: list[DetectionItem] = []
        for item in items:
            bbox = item.bbox
            crop = image.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
            enriched_items.append(
                item.model_copy(
                    update={"crop_image_base64": self._encode_image_base64(crop)},
                ),
            )
        return enriched_items

    def _annotate_image(self, image: Image.Image, items: list[DetectionItem]) -> Image.Image:
        """Draw bounding boxes and labels on a copy of the original image."""
        annotated = image.copy()
        draw = ImageDraw.Draw(annotated)
        font = ImageFont.load_default()

        for item in items:
            bbox = item.bbox
            label_text = f"{item.label} {item.confidence:.2f}"
            draw.rectangle((bbox.x1, bbox.y1, bbox.x2, bbox.y2), outline="#FF6B00", width=4)
            text_box = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_box[2] - text_box[0]
            text_height = text_box[3] - text_box[1]
            text_x = bbox.x1
            text_y = max(0, bbox.y1 - text_height - 8)
            draw.rectangle(
                (text_x, text_y, text_x + text_width + 10, text_y + text_height + 8),
                fill="#FF6B00",
            )
            draw.text((text_x + 5, text_y + 4), label_text, fill="white", font=font)

        return annotated

    def _encode_image_base64(self, image: Image.Image) -> str:
        """Encode a PIL image to base64 PNG."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _index_to_label(self, index: int) -> str:
        """Convert 0-based indexes to spreadsheet-like labels."""
        label = ""
        current = index
        while True:
            current, remainder = divmod(current, 26)
            label = chr(ord("A") + remainder) + label
            if current == 0:
                return label
            current -= 1
