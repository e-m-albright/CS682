import os
from pathlib import Path


SRC_DIR = Path(os.path.dirname(os.path.realpath(__file__)))
PROJECT_DIR = SRC_DIR.parent

DATASET_DIR = Path(os.path.join(PROJECT_DIR, "data", "ds000157"))

OUTPUT_DIR = Path(os.path.join(PROJECT_DIR, "out"))

PLOTS_DIR = Path(os.path.join(OUTPUT_DIR, "plots"))
MODELS_DIR = Path(os.path.join(OUTPUT_DIR, "models"))
