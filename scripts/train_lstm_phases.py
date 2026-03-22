from __future__ import annotations

from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


# =========================
# Config
# =========================
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/lstm_phases")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

X_PATH = DATA_DIR / "X.npy"
Y_PATH = DATA_DIR / "y.npy"
META_PATH = DATA_DIR / "meta.csv"

SEED = 42
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
HIDDEN_SIZE = 128
NUM_LAYERS = 1
DROPOUT = 0.2

TRAIN_ATHLETES = {
    "abokhala_karim",
    "alipour_ali",
    "friedich_raphael",
    "moeini_sedeh_alireza",
}
VAL_ATHLETES = {"nasar_karlos"}
TEST_ATHLETES = {"rostami_kianoush"}

PHASE_NAMES = {
    1: "setup",
    2: "first_pull",
    3: "transition",
    4: "second_pull",
    5: "turnover",
    6: "catch",
    7: "recovery",
}


# =========================
# Utils
# =========================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PhaseDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class LSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        lstm_dropout = dropout if num_layers > 1 else 0.0

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        output, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]  # (batch, hidden_size)
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


def remap_labels(y: np.ndarray, class_ids: list[int]) -> np.ndarray:
    mapping = {cls_id: i for i, cls_id in enumerate(class_ids)}
    return np.array([mapping[int(v)] for v in y], dtype=np.int64)


def inverse_label_map(class_ids: list[int]) -> dict[int, int]:
    return {i: cls_id for i, cls_id in enumerate(class_ids)}


def compute_class_weights(y_train_remapped: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y_train_remapped, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    weights = counts.sum() / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def standardize_by_train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # media y std por feature, usando train solamente
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    std[std < 1e-8] = 1.0

    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std

    return X_train_std, X_val_std, X_test_std, mean, std


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro")

    return avg_loss, acc, f1, np.array(all_targets), np.array(all_preds)


# =========================
# Main
# =========================
def main() -> None:
    set_seed(SEED)

    if not X_PATH.exists() or not Y_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError("Faltan X.npy, y.npy o meta.csv en data/processed")

    X = np.load(X_PATH)
    y = np.load(Y_PATH)
    meta = pd.read_csv(META_PATH)

    if len(X) != len(y) or len(X) != len(meta):
        raise ValueError("X, y y meta no tienen la misma longitud")

    # Filtrar por atletas
    train_mask = meta["athlete"].isin(TRAIN_ATHLETES).to_numpy()
    val_mask = meta["athlete"].isin(VAL_ATHLETES).to_numpy()
    test_mask = meta["athlete"].isin(TEST_ATHLETES).to_numpy()

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    meta_train = meta[train_mask].reset_index(drop=True)
    meta_val = meta[val_mask].reset_index(drop=True)
    meta_test = meta[test_mask].reset_index(drop=True)

    print("[INFO] Split por atleta")
    print(f"Train: {X_train.shape}, atletas={sorted(meta_train['athlete'].unique())}")
    print(f"Val:   {X_val.shape}, atletas={sorted(meta_val['athlete'].unique())}")
    print(f"Test:  {X_test.shape}, atletas={sorted(meta_test['athlete'].unique())}")

    # Mantener solo clases presentes en train
    class_ids = sorted(np.unique(y_train).tolist())
    num_classes = len(class_ids)
    print(f"[INFO] Clases en train: {class_ids}")

    # Filtrar val/test a clases presentes en train
    val_keep = np.isin(y_val, class_ids)
    test_keep = np.isin(y_test, class_ids)

    X_val, y_val = X_val[val_keep], y_val[val_keep]
    X_test, y_test = X_test[test_keep], y_test[test_keep]
    meta_val = meta_val[val_keep].reset_index(drop=True)
    meta_test = meta_test[test_keep].reset_index(drop=True)

    # Remap labels a 0..C-1
    y_train_r = remap_labels(y_train, class_ids)
    y_val_r = remap_labels(y_val, class_ids)
    y_test_r = remap_labels(y_test, class_ids)

    idx_to_class = inverse_label_map(class_ids)
    target_names = [PHASE_NAMES[idx_to_class[i]] for i in range(num_classes)]

    # Normalización
    X_train, X_val, X_test, mean, std = standardize_by_train(X_train, X_val, X_test)

    # Datasets / loaders
    train_ds = PhaseDataset(X_train, y_train_r)
    val_ds = PhaseDataset(X_val, y_val_r)
    test_ds = PhaseDataset(X_test, y_test_r)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Modelo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Device: {device}")

    input_size = X_train.shape[-1]

    model = LSTMClassifier(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    class_weights = compute_class_weights(y_train_r, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_val_f1 = -1.0
    best_model_path = OUTPUT_DIR / "best_model.pt"

    history = []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, val_f1, _, _ = evaluate(
            model, val_loader, criterion, device
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "train_f1_macro": train_f1,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1_macro": val_f1,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_ids": class_ids,
                    "input_size": input_size,
                    "hidden_size": HIDDEN_SIZE,
                    "num_layers": NUM_LAYERS,
                    "dropout": DROPOUT,
                    "mean": mean,
                    "std": std,
                },
                best_model_path,
            )

    print(f"\n[INFO] Mejor val_f1_macro = {best_val_f1:.4f}")
    print(f"[INFO] Modelo guardado en {best_model_path}")

    # Cargar mejor modelo
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)

    best_model = LSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(checkpoint["class_ids"]),
        dropout=checkpoint["dropout"],
    ).to(device)
    best_model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_acc, test_f1, y_true, y_pred = evaluate(
        best_model, test_loader, criterion, device
    )

    print("\n[TEST RESULTS]")
    print(f"test_loss     = {test_loss:.4f}")
    print(f"test_acc      = {test_acc:.4f}")
    print(f"test_f1_macro = {test_f1:.4f}")

    report = classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(
        y_true,
        y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    ))

    print("[CONFUSION MATRIX]")
    print(cm)

    # Guardar outputs
    pd.DataFrame(history).to_csv(OUTPUT_DIR / "history.csv", index=False)
    pd.DataFrame(cm, index=target_names, columns=target_names).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )
    with open(OUTPUT_DIR / "classification_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    test_preds_df = meta_test.copy()
    test_preds_df["y_true_remap"] = y_true
    test_preds_df["y_pred_remap"] = y_pred
    test_preds_df["y_true_phase_id"] = [idx_to_class[i] for i in y_true]
    test_preds_df["y_pred_phase_id"] = [idx_to_class[i] for i in y_pred]
    test_preds_df["y_true_phase_name"] = [PHASE_NAMES[idx_to_class[i]] for i in y_true]
    test_preds_df["y_pred_phase_name"] = [PHASE_NAMES[idx_to_class[i]] for i in y_pred]
    test_preds_df.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print(f"\n[INFO] Resultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()