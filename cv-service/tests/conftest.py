"""Shared test fixtures."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, detector


class DummyArray:
    """Small tensor-like wrapper matching the few Ultralytics APIs we use."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        value = self._values[index]
        if isinstance(value, list):
            return DummyArray(value)
        return value

    def tolist(self):
        return self._values


class DummyBox:
    """Fake YOLO box structure for tests."""

    def __init__(self, cls_id: int, conf: float, coords: list[int]):
        self.cls = DummyArray([cls_id])
        self.conf = DummyArray([conf])
        self.xyxy = DummyArray([coords])


class DummyModel:
    """Predictable YOLO replacement for API tests."""

    def predict(self, image, conf: float, verbose: bool):
        boxes = []
        if image.width >= 760:
            boxes = [
                DummyBox(0, 0.95, [40, 40, 220, 220]),
                DummyBox(0, 0.91, [260, 55, 430, 235]),
            ]
        return [SimpleNamespace(boxes=boxes, names={0: "durian"})]


@pytest.fixture(autouse=True)
def stub_detector(monkeypatch):
    """Avoid loading the real YOLO model during tests."""
    monkeypatch.setattr(detector, "load", lambda: None)
    monkeypatch.setattr(detector, "_model", DummyModel())
    yield


@pytest.fixture
def client():
    """FastAPI test client with lifespan enabled."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def image_bytes() -> bytes:
    """Generate a valid JPEG image accepted by validators."""
    image = Image.new("RGB", (800, 600), "white")
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
