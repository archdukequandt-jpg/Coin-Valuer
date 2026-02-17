
from pathlib import Path

# Single unified project root (GitHub compatible)
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR
MODEL_DIR = BASE_DIR

def get_data_path(filename: str):
    return DATA_DIR / filename

def get_model_path(filename: str):
    return MODEL_DIR / filename
