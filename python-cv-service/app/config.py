"""Application configuration for the YOLO detection microservice."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "durian-best.pt"

# A lower threshold catches more candidates, but also increases false positives.
# This default is a conservative starting point for early debugging.
CONFIDENCE_THRESHOLD = 0.35

# Only boxes of this class are returned. Keeping it explicit avoids leaking
# unrelated classes if the loaded model was trained with multiple labels.
TARGET_CLASS_NAME = "durian"
