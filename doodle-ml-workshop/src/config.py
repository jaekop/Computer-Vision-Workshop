"""One consistent configuration for every workshop notebook."""
from pathlib import Path

CLASSES = ("airplane", "bicycle", "cat", "fish", "house", "tree")
IMAGE_SIZE = 28
SEED = 42
TRAIN_PER_CLASS = 600
VAL_PER_CLASS = 150
TEST_PER_CLASS = 150
SAMPLES_PER_CLASS = TRAIN_PER_CLASS + VAL_PER_CLASS + TEST_PER_CLASS
DEFAULT_CACHE = Path.home() / ".cache" / "doodle_ai_workshop"

