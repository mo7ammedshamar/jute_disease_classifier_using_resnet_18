# Jute Leaf Disease Classification with Transfer Learning (ResNet-18)

An image classification project using **PyTorch**, **torchvision**, and **Matplotlib** to fine-tune **ResNet-18** on the [`Project-AgML/jute_disease_classification`](https://huggingface.co/datasets/Project-AgML/jute_disease_classification) dataset across 5 leaf health categories.

---

## Categories (5 Classes)

1. **Dieback**
2. **Fresh**
3. **Holed**
4. **Mosaic**
5. **Stem Soft Rot**

---

## Stack & Methodology

- **Model**: Pre-trained `ResNet-18` (`torchvision.models.resnet18`) with a custom `Dropout(0.2)` + `Linear(512, 5)` classification head.
- **Dataset**: 1,390 images from `Project-AgML/jute_disease_classification` (supports local Arrow file `data/jute_disease_classification-train.arrow` with automatic Hugging Face Hub fallback), split **80% train / 20% validation** using stratified sampling.
- **Data Augmentation** (`src/jute_disease_classifier/transform.py`):
  - Train: `RandomResizedCrop(224, scale=(0.8, 1.0))`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation(15)`, `ColorJitter(0.15, 0.15, 0.15)`, ImageNet normalization.
  - Validation: `Resize(256)`, `CenterCrop(224)`, ImageNet normalization.
- **Training**: `CrossEntropyLoss`, `AdamW` (`lr=1e-4`, `weight_decay=1e-4`), `ReduceLROnPlateau` scheduler, early stopping (`patience=3`), and target validation accuracy stop trigger (`98.5%`).
- **Visualization**: Matplotlib train/validation loss & accuracy curves and normalized confusion matrix.

---

## Project Structure

```text
my_project/
├── data/
│   ├── dataset_info.json
│   └── jute_disease_classification-train.arrow
├── outputs/
│   ├── checkpoints/
│   │   ├── best_resnet18.pth
│   │   └── last_resnet18.pth
│   ├── plots/
│   │   ├── loss_accuracy_curves.png
│   │   └── confusion_matrix.png
│   └── training_history.json
├── src/
│   └── jute_disease_classifier/
│       ├── __init__.py
│       ├── config.py         # Paths, dataset settings, and hyperparameters
│       ├── model.py          # ResNet-18 transfer learning model builder
│       ├── transform.py      # Train and validation torchvision transforms
│       └── utils.py          # Seed setup, checkpointing, and Matplotlib plots
├── main.ipynb                # End-to-end training, evaluation, plotting & inference
├── pyproject.toml            # Dependencies and project metadata
└── README.md
```

---

## Setup & Usage

### 1. Install Dependencies

Using [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

### 2. Run the Notebook

Open `main.ipynb` in VS Code or Jupyter (select the `.venv` kernel) and run all cells to:

1. Load the dataset and inspect class distribution + sample images.
2. Build DataLoaders and fine-tune ResNet-18 while tracking train/validation loss and accuracy.
3. Save the best checkpoint (`outputs/checkpoints/best_resnet18.pth`) and metrics (`outputs/training_history.json`).
4. Plot loss/accuracy curves (`outputs/plots/loss_accuracy_curves.png`) and confusion matrix (`outputs/plots/confusion_matrix.png`).
5. Run inference on sample validation images.

---

## Results

- **Best Validation Accuracy**: `97.84%` (with validation loss dropping to `0.0939` by Epoch 4).
