"""Unit tests for sorting and label assignment."""

from __future__ import annotations

from types import SimpleNamespace

from app.schemas import BoundingBox
from app.services.detector import DurianDetector, RawDetection


class DummyArray:
    """Small tensor-like wrapper matching detector access patterns."""

    def __init__(self, values):
        self._values = values

    def __getitem__(self, index):
        value = self._values[index]
        if isinstance(value, list):
            return DummyArray(value)
        return value

    def tolist(self):
        return self._values


def test_assign_labels_sorts_by_row_then_column():
    detector = DurianDetector()
    raw_items = [
        RawDetection("durian", 0.91, BoundingBox(x1=320, y1=210, x2=420, y2=340)),
        RawDetection("durian", 0.92, BoundingBox(x1=40, y1=40, x2=180, y2=180)),
        RawDetection("durian", 0.93, BoundingBox(x1=210, y1=55, x2=300, y2=190)),
        RawDetection("durian", 0.90, BoundingBox(x1=60, y1=220, x2=170, y2=360)),
    ]

    items = detector._assign_labels(raw_items)

    assert [item.label for item in items] == ["A", "B", "C", "D"]
    assert [item.bbox.x1 for item in items] == [40, 210, 60, 320]


def test_build_response_skips_non_target_classes():
    detector = DurianDetector()
    result = SimpleNamespace(
        boxes=[
            SimpleNamespace(
                cls=DummyArray([0]),
                conf=DummyArray([0.9]),
                xyxy=DummyArray([[10, 20, 100, 140]]),
            ),
            SimpleNamespace(
                cls=DummyArray([1]),
                conf=DummyArray([0.8]),
                xyxy=DummyArray([[15, 25, 110, 150]]),
            ),
        ],
        names={0: "durian", 1: "other"},
    )

    response = detector._build_response(result)

    assert response.count == 1
    assert response.items[0].label == "A"
