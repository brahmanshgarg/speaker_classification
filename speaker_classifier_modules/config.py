from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
best_model_name = "best_model_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".pth"
BEST_MODEL = MODEL_DIR / best_model_name

# print all the paths
print(f"Project root: {PROJECT_ROOT}")
print(f"Data directory: {DATA_DIR}")
print(f"Model directory: {MODEL_DIR}")
print(f"Best model path: {BEST_MODEL}")