import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _environment_path(name, default):
    value = os.path.expandvars(os.path.expanduser(os.environ.get(name, default)))
    if not os.path.isabs(value):
        value = os.path.join(BASE_DIR, value)
    return os.path.abspath(value)


RAW_ROOT = _environment_path("BME_RAW_ROOT", BASE_DIR)
DATA_DIR = _environment_path("BME_DATA_DIR", os.path.join(BASE_DIR, "data"))
SAVE_DIR = _environment_path("BME_SAVE_DIR", os.path.join(BASE_DIR, "train_runs"))
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
HISTORY_PATH = os.path.join(SAVE_DIR, "history.json")
CACHE_DIR = _environment_path("BME_CACHE_DIR", os.path.join(SAVE_DIR, "monai_cache"))
VISUALIZATION_PATH = os.path.join(SAVE_DIR, "ct_nodule_visualization.png")
TRAINING_CURVE_PATH = os.path.join(SAVE_DIR, "training_curves.png")
