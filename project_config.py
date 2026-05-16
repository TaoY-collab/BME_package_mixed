import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
SAVE_DIR = os.path.join(BASE_DIR, "train_runs")
PRETRAINED_PATH = os.path.join(BASE_DIR, "ssl_pretrained_weights.pth")
CHECKPOINT_PATH = os.path.join(SAVE_DIR, "checkpoint.pth")
BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
HISTORY_PATH = os.path.join(SAVE_DIR, "history.json")
CACHE_DIR = os.path.join(SAVE_DIR, "monai_cache")
VISUALIZATION_PATH = os.path.join(SAVE_DIR, "ct_nodule_visualization.png")
TRAINING_CURVE_PATH = os.path.join(SAVE_DIR, "training_curves.png")
