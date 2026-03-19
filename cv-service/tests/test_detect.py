"""API behavior tests for detection endpoints."""

from __future__ import annotations

from io import BytesIO
from types import SimpleNamespace

from PIL import Image

from app.config import MIN_IMAGE_WIDTH
from app.main import detector
from tests.conftest import DummyBox


def test_detect_returns_stable_labels(client, image_bytes):
    response = client.post(
        "/detect",
        files={"file": ("durian.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert [item["label"] for item in payload["items"]] == ["A", "B"]
    assert "annotated_image_base64" not in payload


def test_detect_and_annotate_returns_assets(client, image_bytes):
    response = client.post(
        "/detect-and-annotate",
        files={"file": ("durian.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["annotated_image_base64"]
    assert all(item["crop_image_base64"] for item in payload["items"])


def test_detect_and_annotate_keeps_top_9_over_sixty_percent(client, image_bytes, monkeypatch):
    class HighConfidenceModel:
        def predict(self, image, conf: float, verbose: bool):
            boxes = [
                DummyBox(0, 0.99 - index * 0.01, [20 + index * 10, 40, 90 + index * 10, 160])
                for index in range(12)
            ]
            return [SimpleNamespace(boxes=boxes, names={0: "durian"})]

    monkeypatch.setattr(detector, "_model", HighConfidenceModel())

    response = client.post(
        "/detect-and-annotate",
        files={"file": ("durian.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 9
    assert payload.get("message") is None
    assert [item["label"] for item in payload["items"]] == ["A", "B", "C", "D", "E", "F", "G", "H", "I"]


def test_detect_and_annotate_falls_back_to_top_3_over_forty_percent(client, image_bytes, monkeypatch):
    class FallbackModel:
        def predict(self, image, conf: float, verbose: bool):
            boxes = [
                DummyBox(0, 0.58, [260, 60, 360, 200]),
                DummyBox(0, 0.55, [40, 40, 160, 180]),
                DummyBox(0, 0.49, [420, 70, 520, 210]),
                DummyBox(0, 0.45, [580, 80, 680, 220]),
            ]
            return [SimpleNamespace(boxes=boxes, names={0: "durian"})]

    monkeypatch.setattr(detector, "_model", FallbackModel())

    response = client.post(
        "/detect-and-annotate",
        files={"file": ("durian.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 3
    assert payload.get("message") is None
    assert [item["confidence"] for item in payload["items"]] == [0.55, 0.58, 0.49]


def test_detect_and_annotate_returns_message_when_nothing_matches(client, image_bytes, monkeypatch):
    class NoMatchModel:
        def predict(self, image, conf: float, verbose: bool):
            boxes = [
                DummyBox(0, 0.40, [40, 40, 160, 180]),
                DummyBox(0, 0.32, [240, 40, 360, 180]),
            ]
            return [SimpleNamespace(boxes=boxes, names={0: "durian"})]

    monkeypatch.setattr(detector, "_model", NoMatchModel())

    response = client.post(
        "/detect-and-annotate",
        files={"file": ("durian.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "count": 0,
        "items": [],
        "message": "没有识别到榴莲",
    }


def test_detect_rejects_empty_upload(client):
    response = client.post(
        "/detect",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded image is empty."


def test_detect_rejects_non_image_file(client):
    response = client.post(
        "/detect",
        files={"file": ("notes.txt", b"not-an-image", "text/plain")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file."


def test_detect_rejects_small_image(client):
    image = Image.new("RGB", (MIN_IMAGE_WIDTH - 1, 640), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/detect",
        files={"file": ("small.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == f"Image width must be at least {MIN_IMAGE_WIDTH}px."


def test_detect_requires_exactly_one_input(client, image_bytes):
    response = client.post(
        "/detect",
        data={"image_url": "https://example.com/durian.jpg"},
        files={"file": ("durian.jpg", image_bytes, "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Provide exactly one of file or image_url."


def test_model_info(client):
    response = client.get("/model-info")

    assert response.status_code == 200
    payload = response.json()
    assert payload["target_class_name"] == "durian"
    assert payload["min_image_width"] == MIN_IMAGE_WIDTH
