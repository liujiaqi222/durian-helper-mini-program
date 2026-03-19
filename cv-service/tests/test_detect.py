"""API behavior tests for detection endpoints."""

from __future__ import annotations

from io import BytesIO

from PIL import Image


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
    image = Image.new("RGB", (640, 640), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    response = client.post(
        "/detect",
        files={"file": ("small.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Image width must be at least 720px."


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
    assert payload["min_image_width"] == 720
