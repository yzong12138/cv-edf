"""
EDF Concrete Wall Crack Detection
==================================
Binary classification (0 = no damage, 1 = crack/corrosion) using a
fine-tuned EfficientNet-B0 pretrained on ImageNet.

Run:
    uv run python main.py
"""

from tqdm import tqdm
import time
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import models, transforms
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B1_Weights, EfficientNet_B2_Weights,
    EfficientNet_B3_Weights, EfficientNet_B4_Weights, EfficientNet_B5_Weights,
    EfficientNet_B6_Weights, EfficientNet_B7_Weights,
)
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/zong/cv-dataset/test_data_scientist")
LABELS_CSV = DATA_DIR / "labels.csv"

MODEL_VARIANT = "b0"    # EfficientNet variant: b0 … b7
# Each variant expects a different input resolution
EFFICIENTNET_IMG_SIZE = {
    "b0": 224, "b1": 240, "b2": 260, "b3": 300,
    "b4": 380, "b5": 456, "b6": 528, "b7": 600,
}
BATCH_SIZE = 32
NUM_EPOCHS = 30
LR = 3e-4               # learning rate for fine-tuning head
LR_BACKBONE = 1e-5      # lower lr for pretrained backbone layers
WEIGHT_DECAY = 1e-4
PATIENCE = 7            # early-stopping patience (epochs)
NUM_WORKERS = 2

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

OUTPUT_DIR = Path("/home/zong/cv-dataset/output")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
def get_transforms(split: str, img_size: int = 224) -> transforms.Compose:
    """Return data transforms for train / val / test."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:  # val / test
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class CrackDataset(Dataset):
    """PyTorch Dataset for concrete wall crack images."""

    def __init__(self, df: pd.DataFrame, data_dir: Path, split: str, img_size: int = 224) -> None:
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.transform = get_transforms(split, img_size=img_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["name"]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = torch.tensor(int(row["label"]), dtype=torch.long)
        return image, label


def build_dataloaders(
    df: pd.DataFrame, data_dir: Path, batch_size: int = BATCH_SIZE, img_size: int = 224
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Stratified split → DataLoaders with balanced sampling for train set."""

    # --- split ------------------------------------------------------------
    df_train, df_temp = train_test_split(
        df, test_size=1 - TRAIN_RATIO, stratify=df["label"], random_state=SEED
    )
    relative_val = VAL_RATIO / (1 - TRAIN_RATIO)
    df_val, df_test = train_test_split(
        df_temp, test_size=1 - relative_val, stratify=df_temp["label"], random_state=SEED
    )

    print(
        f"Split sizes  — train: {len(df_train):,}  "
        f"val: {len(df_val):,}  test: {len(df_test):,}"
    )
    for name, subset in [("train", df_train), ("val", df_val), ("test", df_test)]:
        counts = subset["label"].value_counts().to_dict()
        print(f"  {name}: class-0={counts.get(0,0)}  class-1={counts.get(1,0)}")

    # --- datasets ---------------------------------------------------------
    ds_train = CrackDataset(df_train, data_dir, "train", img_size=img_size)
    ds_val = CrackDataset(df_val, data_dir, "val", img_size=img_size)
    ds_test = CrackDataset(df_test, data_dir, "test", img_size=img_size)

    # --- balanced sampler for training ------------------------------------
    class_counts = df_train["label"].value_counts().sort_index().values  # [n0, n1]
    class_weights = 1.0 / class_counts
    sample_weights = df_train["label"].map(
        {0: class_weights[0], 1: class_weights[1]}
    ).values
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    pin = torch.cuda.is_available()
    loader_train = DataLoader(
        ds_train, batch_size=batch_size, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=pin
    )
    loader_val = DataLoader(
        ds_val, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin
    )
    loader_test = DataLoader(
        ds_test, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin
    )

    return loader_train, loader_val, loader_test


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
_EFFICIENTNET_VARIANTS: dict[str, tuple] = {
    "b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1),
    "b1": (models.efficientnet_b1, EfficientNet_B1_Weights.IMAGENET1K_V1),
    "b2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1),
    "b3": (models.efficientnet_b3, EfficientNet_B3_Weights.IMAGENET1K_V1),
    "b4": (models.efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1),
    "b5": (models.efficientnet_b5, EfficientNet_B5_Weights.IMAGENET1K_V1),
    "b6": (models.efficientnet_b6, EfficientNet_B6_Weights.IMAGENET1K_V1),
    "b7": (models.efficientnet_b7, EfficientNet_B7_Weights.IMAGENET1K_V1),
}


def build_model(variant: str = "b0", num_classes: int = 2) -> nn.Module:
    """EfficientNet-B{0..7} pretrained on ImageNet, with a new classification head."""
    if variant not in _EFFICIENTNET_VARIANTS:
        raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(_EFFICIENTNET_VARIANTS)}")
    model_fn, weights = _EFFICIENTNET_VARIANTS[variant]
    model = model_fn(weights=weights)

    # Replace the classifier head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, num_classes),
    )
    return model


def get_optimizer(model: nn.Module, lr: float = LR, lr_backbone: float = LR_BACKBONE) -> torch.optim.Optimizer:
    """Use different learning rates for backbone vs classifier head.
    If lr_backbone == 0, backbone parameters are frozen (no gradients computed).
    """
    backbone_params = [p for n, p in model.named_parameters() if "classifier" not in n]
    head_params = list(model.classifier.parameters())

    if lr_backbone == 0:
        for p in backbone_params:
            p.requires_grad = False
        print("Backbone frozen (lr_backbone=0) — training head only.")
        return torch.optim.AdamW(head_params, lr=lr, weight_decay=WEIGHT_DECAY)

    return torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr},
        ],
        weight_decay=WEIGHT_DECAY,
    )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
class EarlyStopping:
    """Stop training when a metric stops improving."""

    def __init__(self, patience: int = PATIENCE, mode: str = "max") -> None:
        self.patience = patience
        self.mode = mode
        self.best = None
        self.counter = 0
        self.triggered = False

    def step(self, value: float) -> bool:
        improved = (
            self.best is None
            or (self.mode == "max" and value > self.best)
            or (self.mode == "min" and value < self.best)
        )
        if improved:
            self.best = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return improved  # True = new best


def compute_class_weights(df: pd.DataFrame, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weights for the loss function."""
    counts = df["label"].value_counts().sort_index().values.astype(float)
    weights = counts.sum() / (len(counts) * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    phase: str,
    scaler: torch.amp.GradScaler | None = None,
) -> tuple[float, list, list]:
    """Run one epoch with optional AMP. Returns (loss, all_labels, all_preds)."""
    is_train = phase == "train"
    model.train() if is_train else model.eval()
    use_amp = scaler is not None and device.type == "cuda"

    total_loss = 0.0
    all_labels, all_preds = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for images, labels in tqdm(loader, desc="No. batch"):
            images, labels = images.to(device), labels.to(device)

            if is_train:
                optimizer.zero_grad()

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, all_labels, all_preds


def train(
    model: nn.Module,
    loader_train: DataLoader,
    loader_val: DataLoader,
    device: torch.device,
    class_weights: torch.Tensor,
    output_dir: Path,
    n_epochs: int = NUM_EPOCHS,
    lr: float = LR,
    lr_backbone: float = LR_BACKBONE,
) -> dict:
    """Full training loop with AMP on CUDA. Returns history dict."""
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = get_optimizer(model, lr=lr, lr_backbone=lr_backbone)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    early_stopping = EarlyStopping(patience=PATIENCE, mode="max")
    # GradScaler enables AMP on CUDA; on CPU it is a no-op placeholder
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None
    if scaler is not None:
        print("AMP enabled (mixed precision)")

    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}
    best_model_path = output_dir / "best_model.pt"

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_loss, train_labels, train_preds = run_epoch(
            model, loader_train, criterion, optimizer, device, "train", scaler=scaler
        )
        val_loss, val_labels, val_preds = run_epoch(
            model, loader_val, criterion, None, device, "val", scaler=None
        )

        train_f1 = f1_score(train_labels, train_preds, average="weighted", zero_division=0)
        val_f1 = f1_score(val_labels, val_preds, average="weighted", zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        scheduler.step()

        is_best = early_stopping.step(val_f1)
        if is_best:
            torch.save(model.state_dict(), best_model_path)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{n_epochs} | "
            f"train loss {train_loss:.4f}  f1 {train_f1:.4f} | "
            f"val loss {val_loss:.4f}  f1 {val_f1:.4f} | "
            f"{'[BEST]' if is_best else '      '} {elapsed:.0f}s"
        )
        
        if early_stopping.triggered:
            print(f"Early stopping triggered after epoch {epoch}.")
            break

    print(f"\nBest val F1: {early_stopping.best:.4f}  →  {best_model_path}")
    return history


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    split_name: str,
    output_dir: Path,
) -> dict:
    """Evaluate model and return a metrics dict."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1_weighted": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1_macro": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs),
    }

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy       : {metrics['accuracy']:.4f}")
    print(f"  F1 (weighted)  : {metrics['f1_weighted']:.4f}")
    print(f"  F1 (macro)     : {metrics['f1_macro']:.4f}")
    print(f"  Precision (1)  : {metrics['precision']:.4f}")
    print(f"  Recall (1)     : {metrics['recall']:.4f}")
    print(f"  ROC-AUC        : {metrics['roc_auc']:.4f}")
    print(f"\n{classification_report(all_labels, all_preds, target_names=['no crack','crack'])}")

    # Confusion matrix plot
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, split_name, output_dir)

    return metrics


def _plot_confusion_matrix(cm: np.ndarray, split_name: str, output_dir: Path) -> None:
    _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    classes = ["No crack (0)", "Crack (1)"]
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted label",
        ylabel="True label",
        title=f"Confusion Matrix — {split_name}",
    )
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:,}",
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14,
            )
    plt.tight_layout()
    path = output_dir / f"confusion_matrix_{split_name}.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_training_history(history: dict, output_dir: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    _, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Val")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_f1"], label="Train")
    axes[1].plot(epochs, history["val_f1"], label="Val")
    axes[1].set(title="Weighted F1", xlabel="Epoch", ylabel="F1 Score")
    axes[1].legend()

    plt.tight_layout()
    path = output_dir / "training_history.png"
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="EDF Crack Detection — fine-tune EfficientNet")
    parser.add_argument("--model", default=MODEL_VARIANT, choices=list(_EFFICIENTNET_VARIANTS),
                        help="EfficientNet variant (default: b0)")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--lr-backbone", type=float, default=LR_BACKBONE)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    variant = args.model
    n_epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    lr_backbone = args.lr_backbone
    data_dir = args.data_dir
    labels_csv = data_dir / "labels.csv"
    img_size = EFFICIENTNET_IMG_SIZE[variant]

    # Encode key hyperparameters in the run folder name
    run_name = f"net_{variant}_bs{batch_size}_lr{lr:.0e}_lrbb{lr_backbone:.0e}_epochs{n_epochs}"
    output_dir = args.output_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run dir: {output_dir}")
    set_seed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    df = pd.read_csv(labels_csv)
    # Drop rows whose image file is missing from disk
    exists_mask = df["name"].apply(lambda n: (data_dir / n).exists())
    missing = (~exists_mask).sum()
    if missing:
        print(f"WARNING: {missing} entries in CSV have no matching file and will be skipped.")
        df = df[exists_mask].reset_index(drop=True)
    print(f"\nDataset: {len(df):,} images  |  class-0: {(df.label==0).sum():,}  class-1: {(df.label==1).sum():,}")

    best_model_path = output_dir / "best_model.pt"
    summary_path = output_dir / "metrics_summary.csv"

    # ── Train (skip if checkpoint already exists) ──────────────────────────
    if best_model_path.exists():
        print(f"\nCheckpoint found — skipping training. ({best_model_path})")
    else:
        loader_train, loader_val, _ = build_dataloaders(df, data_dir, batch_size=batch_size, img_size=img_size)

        model = build_model(variant=variant, num_classes=2).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\nModel: EfficientNet-{variant.upper()}  |  img_size: {img_size}  |  params: {total_params:,}")

        class_weights = compute_class_weights(df, device)
        print(f"Class weights: {class_weights.cpu().tolist()}")

        print(f"\nTraining for up to {n_epochs} epochs (early-stop patience={PATIENCE}) …\n")
        history = train(model, loader_train, loader_val, device, class_weights, output_dir, n_epochs=n_epochs, lr=lr, lr_backbone=lr_backbone)
        plot_training_history(history, output_dir)

    # ── Evaluate (skip if summary already exists) ──────────────────────────
    if summary_path.exists():
        print(f"\nResults found — skipping evaluation. ({summary_path})")
        print(pd.read_csv(summary_path, index_col=0).to_string())
    else:
        _, loader_val, loader_test = build_dataloaders(df, data_dir, batch_size=batch_size, img_size=img_size)

        model = build_model(variant=variant, num_classes=2).to(device)
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        val_metrics = evaluate(model, loader_val, device, "validation", output_dir)
        test_metrics = evaluate(model, loader_test, device, "test", output_dir)

        summary = pd.DataFrame([val_metrics, test_metrics], index=["validation", "test"])
        summary.to_csv(summary_path)
        print(f"\nMetrics saved to {summary_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
