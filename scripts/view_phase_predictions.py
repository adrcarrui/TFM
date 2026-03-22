from __future__ import annotations

from pathlib import Path
import re
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# =========================
# Config
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[1]

VIDEO_PATH = PROJECT_ROOT / "data/raw_videos/rostami/snatch_-96kg_rostami_kianoush_i2_ok_000010.mp4"
KEYPOINTS_DIR = PROJECT_ROOT / "data/keypoints"
LABELS_CSV = PROJECT_ROOT / "data/annotations/master_frame_labels.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/lstm_phases/best_model.pt"

WINDOW_SIZE = 16
STRIDE = 4

SAVE_OUTPUT_VIDEO = True
OUTPUT_VIDEO_PATH = PROJECT_ROOT / "outputs/lstm_phases/prediction_overlay.mp4"

PHASE_NAMES = {
    0: "unlabeled",
    1: "setup",
    2: "first_pull",
    3: "transition",
    4: "second_pull",
    5: "turnover",
    6: "catch",
    7: "recovery",
}


# =========================
# Model
# =========================
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
        _, (h_n, _) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.dropout(last_hidden)
        logits = self.fc(out)
        return logits


# =========================
# Helpers
# =========================
def resolve_keypoint_csv(video_path: Path) -> Path:
    rel = video_path.relative_to(PROJECT_ROOT / "data/raw_videos")
    return (KEYPOINTS_DIR / rel).with_suffix(".csv")


def load_keypoints(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).sort_values("frame").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != "frame"]
    df[feature_cols] = df[feature_cols].interpolate(limit_direction="both")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df


def load_labels_for_video(labels_df: pd.DataFrame, video_path: Path) -> pd.DataFrame:
    rel_video = str(video_path.relative_to(PROJECT_ROOT / "data/raw_videos")).replace("\\", "/")
    video_name = video_path.name

    if "video_relpath" in labels_df.columns:
        match = labels_df[labels_df["video_relpath"] == rel_video]
        if not match.empty:
            return match.sort_values("frame").reset_index(drop=True)

    match = labels_df[labels_df["video"] == video_name]
    return match.sort_values("frame").reset_index(drop=True)


def standardize_pose_array(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def remap_phase_id_to_model_index(phase_id: int, class_ids: list[int]) -> int:
    mapping = {cls_id: i for i, cls_id in enumerate(class_ids)}
    return mapping[phase_id]


def inverse_model_index_to_phase_id(model_index: int, class_ids: list[int]) -> int:
    return class_ids[model_index]


def build_framewise_prediction(
    pose_array: np.ndarray,
    model: nn.Module,
    class_ids: list[int],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
    total_frames: int,
) -> np.ndarray:
    """
    Genera predicción por frame usando ventanas deslizantes.
    Si un frame recibe múltiples predicciones, votamos por mayoría.
    """
    num_classes = len(class_ids)
    votes = np.zeros((total_frames, num_classes), dtype=np.int32)

    for start in range(0, total_frames - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        center = start + WINDOW_SIZE // 2

        window = pose_array[start:end].astype(np.float32)
        window = standardize_pose_array(window, mean, std)

        x = torch.tensor(window[None, ...], dtype=torch.float32, device=device)

        with torch.no_grad():
            logits = model(x)
            pred_idx = int(torch.argmax(logits, dim=1).item())

        # votar sobre el frame central
        votes[center, pred_idx] += 1

    pred_phase_ids = np.zeros(total_frames, dtype=np.int64)

    # rellenar frames con voto directo
    for i in range(total_frames):
        if votes[i].sum() > 0:
            pred_phase_ids[i] = inverse_model_index_to_phase_id(int(np.argmax(votes[i])), class_ids)
        else:
            pred_phase_ids[i] = 0

    # rellenar huecos por propagación simple
    last_seen = 0
    for i in range(total_frames):
        if pred_phase_ids[i] == 0:
            pred_phase_ids[i] = last_seen
        else:
            last_seen = pred_phase_ids[i]

    # backward fill para huecos iniciales
    next_seen = 0
    for i in range(total_frames - 1, -1, -1):
        if pred_phase_ids[i] == 0:
            pred_phase_ids[i] = next_seen
        else:
            next_seen = pred_phase_ids[i]

    return pred_phase_ids


def draw_timeline(
    frame: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    current_frame: int,
) -> np.ndarray:
    h, w = frame.shape[:2]
    canvas = frame.copy()

    bar_w = w - 40
    x0 = 20
    y_true = h - 60
    y_pred = h - 30

    n = len(true_labels)
    if n <= 1:
        return canvas

    # color map simple
    colors = {
        0: (100, 100, 100),
        1: (255, 200, 0),
        2: (255, 120, 0),
        3: (180, 0, 255),
        4: (0, 180, 255),
        5: (0, 255, 255),
        6: (0, 255, 0),
        7: (255, 0, 0),
    }

    for i in range(n - 1):
        xa = x0 + int(i / (n - 1) * bar_w)
        xb = x0 + int((i + 1) / (n - 1) * bar_w)

        cv2.line(canvas, (xa, y_true), (xb, y_true), colors[int(true_labels[i])], 8)
        cv2.line(canvas, (xa, y_pred), (xb, y_pred), colors[int(pred_labels[i])], 8)

    xc = x0 + int(current_frame / (n - 1) * bar_w)
    cv2.line(canvas, (xc, y_true - 15), (xc, y_pred + 15), (255, 255, 255), 2)

    cv2.putText(canvas, "GT", (x0, y_true - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Pred", (x0, y_pred - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas


def main() -> None:
    if not VIDEO_PATH.exists():
        raise FileNotFoundError(f"No existe el video: {VIDEO_PATH}")

    keypoint_csv = resolve_keypoint_csv(VIDEO_PATH)
    if not keypoint_csv.exists():
        raise FileNotFoundError(f"No existe el CSV de keypoints: {keypoint_csv}")

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"No existe el CSV de labels: {LABELS_CSV}")

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"No existe el checkpoint: {CHECKPOINT_PATH}")

    labels_df = pd.read_csv(LABELS_CSV)
    pose_df = load_keypoints(keypoint_csv)
    label_df = load_labels_for_video(labels_df, VIDEO_PATH)

    if label_df.empty:
        raise RuntimeError(f"No se encontraron labels para {VIDEO_PATH.name}")

    n = min(len(pose_df), len(label_df))
    pose_df = pose_df.iloc[:n].copy()
    label_df = label_df.iloc[:n].copy()

    feature_cols = [c for c in pose_df.columns if c != "frame"]
    pose_array = pose_df[feature_cols].to_numpy(dtype=np.float32)
    true_phase_ids = label_df["phase_id"].to_numpy(dtype=np.int64)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)

    class_ids = list(checkpoint["class_ids"])
    mean = np.array(checkpoint["mean"], dtype=np.float32)
    std = np.array(checkpoint["std"], dtype=np.float32)

    model = LSTMClassifier(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        num_layers=checkpoint["num_layers"],
        num_classes=len(class_ids),
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pred_phase_ids = build_framewise_prediction(
        pose_array=pose_array,
        model=model,
        class_ids=class_ids,
        mean=mean,
        std=std,
        device=device,
        total_frames=n,
    )

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_OUTPUT_VIDEO:
        OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (width, height))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= n:
            break

        gt_id = int(true_phase_ids[frame_idx])
        pred_id = int(pred_phase_ids[frame_idx])

        gt_name = PHASE_NAMES.get(gt_id, "unknown")
        pred_name = PHASE_NAMES.get(pred_id, "unknown")

        match = gt_id == pred_id
        status_color = (0, 200, 0) if match else (0, 0, 255)

        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (500, 130), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        cv2.putText(frame, f"Frame: {frame_idx}", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"GT:   {gt_name}", (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Pred: {pred_name}", (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2, cv2.LINE_AA)

        frame = draw_timeline(frame, true_phase_ids, pred_phase_ids, frame_idx)

        cv2.imshow("Phase Prediction Viewer", frame)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(25) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            while True:
                paused_key = cv2.waitKey(0) & 0xFF
                if paused_key == ord("p"):
                    break
                if paused_key == ord("q"):
                    cap.release()
                    if writer is not None:
                        writer.release()
                    cv2.destroyAllWindows()
                    return

        frame_idx += 1

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()

    if SAVE_OUTPUT_VIDEO:
        print(f"[OK] Video guardado en: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()