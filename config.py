"""
config.py — Configuración centralizada para todo el proyecto
Define rutas, parámetros, hiperparámetros compartidos
"""

import os
from pathlib import Path

# ============================================================
# 📁 RUTAS BASE
# ============================================================
PROJECT_ROOT = Path(__file__).parent.absolute()
DATASET_BASE = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# ============================================================
# 📊 DATASET
# ============================================================
DATASET_COFFEE = DATASET_BASE / "coffee_Fruit_Maturity_yolo"
TRAIN_DATA = DATASET_COFFEE / "train"
VAL_DATA = DATASET_COFFEE / "valid"
TEST_DATA = DATASET_COFFEE / "test"

# JSON files
TRAIN_JSON = TRAIN_DATA / "cdmenet_coffee_train.json"
VAL_JSON = VAL_DATA / "cdmenet_coffee_valid.json"
TEST_JSON = TEST_DATA / "cdmenet_coffee_test.json"

# YAML para YOLO
YOLO_DATA_YAML = DATASET_COFFEE / "data.yaml"

# ============================================================
# 🧠 MODELOS
# ============================================================
CDMENET_CHECKPOINTS = MODELS_DIR / "cdmenet" / "checkpoints"
YOLO_CHECKPOINTS = MODELS_DIR / "yolov8m_cdmenet_style"
YOLO_WEIGHTS_PTH = MODELS_DIR / "yolov8m" / "yolov8m-seg.pt"

# Crear directorios si no existen
CDMENET_CHECKPOINTS.mkdir(parents=True, exist_ok=True)
YOLO_CHECKPOINTS.mkdir(parents=True, exist_ok=True)

# ============================================================
# 📋 RESULTADOS
# ============================================================
RESULTS_CHECKPOINTS = RESULTS_DIR / "checkpoints"
RESULTS_EVALUATIONS = RESULTS_DIR / "evaluations"
RESULTS_VISUALIZATIONS = RESULTS_DIR / "visualizations"

for d in [RESULTS_CHECKPOINTS, RESULTS_EVALUATIONS, RESULTS_VISUALIZATIONS]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# ⚙️ HIPERPARÁMETROS GLOBALES
# ============================================================
DEVICE = "cuda"  # 'cuda' o 'cpu'
RANDOM_SEED = 123456

# --- CDMENet ---
CDMENET_IMAGE_SIZE = 512
CDMENET_LEARNING_RATE = 1e-4
CDMENET_WEIGHT_DECAY = 1e-4
CDMENET_BATCH_SIZE = 4
CDMENET_EPOCHS = 300
CDMENET_WARMUP_EPOCHS = 4
CDMENET_BLOCK_SIZE = 20

# --- YOLO ---
YOLO_IMAGE_SIZE = 1024
YOLO_LEARNING_RATE = 1e-3
YOLO_WEIGHT_DECAY = 1e-4
YOLO_BATCH_SIZE = 8
YOLO_EPOCHS = 100
YOLO_WARMUP_EPOCHS = 5
YOLO_BLOCK_SIZE = 20

# --- Density Generation ---
DENSITY_SIGMA = 7
DENSITY_PREVIEW = False
DENSITY_PREVIEW_N = 5

# --- Model Architecture ---
CDMENet_BACKBONE_PRETRAINED = True
CDMENet_DENSITY_CHANNELS = 1
CDMENet_AUX_BRANCHES = 3
CDMENet_AUX_CHANNELS = 2

# --- Thresholds y Weights ---
DENSITY_THRESHOLDS_CDMENET = {
    "low": 0.0015,
    "mid": 0.0100
}

LOSS_WEIGHTS_CDMENET = {
    "density_loss": 1.0,
    "class_loss": 0.01,
    "mutual_exc": 1.0,
    "semi_supervised": 0.01
}

# --- Normalization (ImageNet standard) ---
NORMALIZE_MEAN = [0.4931, 0.5346, 0.3792]
NORMALIZE_STD = [0.2217, 0.2025, 0.2085]

# ============================================================
# 🔧 FUNCIONES ÚTILES
# ============================================================
def get_checkpoint_path(model_name, epoch=None):
    """Retorna ruta a checkpoint de un modelo"""
    if model_name == "cdmenet":
        if epoch:
            return CDMENET_CHECKPOINTS / f"cdmenet_epoch_{epoch}.pth"
        return CDMENET_CHECKPOINTS / "cdmenet_checkpoint.pth"
    elif model_name == "yolo":
        return YOLO_CHECKPOINTS / "yolo_checkpoint.pth"
    raise ValueError(f"Unknown model: {model_name}")

def get_best_model_path(model_name):
    """Retorna ruta al mejor modelo guardado"""
    if model_name == "cdmenet":
        return CDMENET_CHECKPOINTS / "cdmenet_epoch_best.pth"
    elif model_name == "yolo":
        return YOLO_CHECKPOINTS / "yolo_best.pth"
    raise ValueError(f"Unknown model: {model_name}")

def ensure_dirs_exist():
    """Asegura que existan todos los directorios necesarios"""
    dirs = [
        CDMENET_CHECKPOINTS,
        YOLO_CHECKPOINTS,
        RESULTS_CHECKPOINTS,
        RESULTS_EVALUATIONS,
        RESULTS_VISUALIZATIONS,
        TRAIN_DATA,
        VAL_DATA,
        TEST_DATA
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Configuración centralizada cargada correctamente.")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATASET_COFFEE: {DATASET_COFFEE}")
    print(f"CDMENET_CHECKPOINTS: {CDMENET_CHECKPOINTS}")
