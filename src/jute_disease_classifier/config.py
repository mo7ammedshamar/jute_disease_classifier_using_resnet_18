import os
from pathlib import Path
import torch

# Base project directory (points to workspace root: my_project)
BASE_DIR = Path(__file__).resolve().parents[2]

# Data Paths
DATA_DIR = BASE_DIR / "data"
LOCAL_ARROW_FILE = DATA_DIR / "jute_disease_classification-train.arrow"

# Output Paths
OUTPUTS_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUTS_DIR / "checkpoints"
PLOTS_DIR = OUTPUTS_DIR / "plots"

# Ensure output directories exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Hugging Face Dataset & Categories
HF_DATASET_NAME = "Project-AgML/jute_disease_classification"
CLASS_NAMES = ["Dieback", "Fresh", "Holed", "Mosaic", "Stem Soft Rot"]
NUM_CLASSES = len(CLASS_NAMES)  # 5 categories

# Image and Augmentation Settings
IMAGE_SIZE = 224
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
VAL_SPLIT_RATIO = 0.2  # 80% train, 20% validation

# Training Hyperparameters
BATCH_SIZE = 32
NUM_EPOCHS = 4
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2
FREEZE_BACKBONE = False
SEED = 42

# DataLoader settings (0 workers is safest on Windows)
NUM_WORKERS = 0

# Device selection (CUDA GPU if available, else CPU)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
