from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd


KEYPOINTS_DIR = Path("data/keypoints")
LABELS_CSV = Path("data/annotations/master_frame_labels.csv")
OUTPUT_DIR = Path("data/processed")

WINDOW_SIZE = 16
STRIDE = 4

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


def extract_athlete_from_video_name(video_name: str) -> str:
    """
    Extrae algo razonable como athlete_id desde nombres tipo:
    snatch_-96kg_nasar_karlos_i1_ok_000011.mp4
    snatch_-96kg_moeini_sedeh_alireza_i2_ok_000016.mp4
    """
    stem = Path(video_name).stem
    parts = stem.split("_")

    # formato esperado:
    # snatch, -96kg, athlete_name..., iX, ok/fail, id
    if len(parts) < 6:
        return "unknown"

    try:
        i_idx = next(i for i, p in enumerate(parts) if re.fullmatch(r"i\d+", p))
    except StopIteration:
        return "unknown"

    athlete_parts = parts[2:i_idx]
    if not athlete_parts:
        return "unknown"

    # usa apellido o bloque completo simplificado
    return "_".join(athlete_parts)


def load_keypoints_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    if "frame" not in df.columns:
        raise ValueError(f"CSV sin columna 'frame': {csv_path}")

    # ordenar por frame
    df = df.sort_values("frame").reset_index(drop=True)

    # rellenar NaN con interpolación simple y luego fallback a 0
    feature_cols = [c for c in df.columns if c != "frame"]
    df[feature_cols] = df[feature_cols].interpolate(limit_direction="both")
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df


def resolve_video_key_for_pose(csv_path: Path) -> tuple[str, str]:
    """
    Devuelve:
    - video_name esperado en labels (con .mp4)
    - video_relpath relativo a data/keypoints, convertido a .mp4
    """
    rel = csv_path.relative_to(KEYPOINTS_DIR)
    video_relpath = str(rel.with_suffix(".mp4")).replace("\\", "/")
    video_name = csv_path.with_suffix(".mp4").name
    return video_name, video_relpath


def find_label_rows(labels_df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    video_name, video_relpath = resolve_video_key_for_pose(csv_path)

    candidates = []

    if "video_relpath" in labels_df.columns:
        candidates.append(labels_df[labels_df["video_relpath"] == video_relpath])

    if "video" in labels_df.columns:
        candidates.append(labels_df[labels_df["video"] == video_name])

    for cand in candidates:
        if not cand.empty:
            return cand.sort_values("frame").reset_index(drop=True)

    return pd.DataFrame()


def build_samples_for_video(
    pose_df: pd.DataFrame,
    label_df: pd.DataFrame,
    video_name: str,
    athlete_id: str,
) -> tuple[list[np.ndarray], list[int], list[dict]]:
    samples_x: list[np.ndarray] = []
    samples_y: list[int] = []
    samples_meta: list[dict] = []

    n = min(len(pose_df), len(label_df))
    if n < WINDOW_SIZE:
        return samples_x, samples_y, samples_meta

    pose_df = pose_df.iloc[:n].copy()
    label_df = label_df.iloc[:n].copy()

    # comprobar frames alineados de forma simple
    # si no lo están exactamente, nos quedamos con el orden
    feature_cols = [c for c in pose_df.columns if c != "frame"]
    pose_data = pose_df[feature_cols].to_numpy(dtype=np.float32)
    phase_ids = label_df["phase_id"].to_numpy(dtype=np.int64)

    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        center = start + WINDOW_SIZE // 2

        center_label = int(phase_ids[center])

        # descartar unlabeled
        if center_label == 0:
            continue

        window = pose_data[start:end]

        samples_x.append(window)
        samples_y.append(center_label)
        samples_meta.append(
            {
                "sample_id": None,  # se asigna luego
                "video": video_name,
                "athlete": athlete_id,
                "start_frame": int(start),
                "end_frame": int(end - 1),
                "center_frame": int(center),
                "phase_id": center_label,
                "phase_name": PHASE_NAMES.get(center_label, "unknown"),
            }
        )

    return samples_x, samples_y, samples_meta


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"No existe: {LABELS_CSV}")

    labels_df = pd.read_csv(LABELS_CSV)

    if "frame" not in labels_df.columns or "phase_id" not in labels_df.columns:
        raise ValueError("master_frame_labels.csv debe tener al menos 'frame' y 'phase_id'")

    keypoint_files = sorted(KEYPOINTS_DIR.rglob("*.csv"))
    if not keypoint_files:
        raise FileNotFoundError(f"No se encontraron CSVs en {KEYPOINTS_DIR}")

    all_x: list[np.ndarray] = []
    all_y: list[int] = []
    all_meta: list[dict] = []

    print(f"[INFO] CSVs de keypoints encontrados: {len(keypoint_files)}")

    for csv_path in keypoint_files:
        video_name, _video_relpath = resolve_video_key_for_pose(csv_path)
        athlete_id = extract_athlete_from_video_name(video_name)

        try:
            pose_df = load_keypoints_csv(csv_path)
        except Exception as e:
            print(f"[WARN] Error leyendo pose {csv_path}: {e}")
            continue

        label_df = find_label_rows(labels_df, csv_path)
        if label_df.empty:
            print(f"[WARN] No se encontraron labels para: {csv_path}")
            continue

        x_list, y_list, meta_list = build_samples_for_video(
            pose_df=pose_df,
            label_df=label_df,
            video_name=video_name,
            athlete_id=athlete_id,
        )

        for x in x_list:
            all_x.append(x)
        all_y.extend(y_list)
        all_meta.extend(meta_list)

        print(f"[OK] {video_name}: {len(x_list)} ventanas")

    if not all_x:
        raise RuntimeError("No se generaron muestras. Revisa nombres/rutas/alineación.")

    X = np.stack(all_x).astype(np.float32)
    y = np.array(all_y, dtype=np.int64)

    for i, row in enumerate(all_meta):
        row["sample_id"] = i

    meta_df = pd.DataFrame(all_meta)

    np.save(OUTPUT_DIR / "X.npy", X)
    np.save(OUTPUT_DIR / "y.npy", y)
    meta_df.to_csv(OUTPUT_DIR / "meta.csv", index=False)

    print("\n[INFO] Dataset generado")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"meta: {OUTPUT_DIR / 'meta.csv'}")

    print("\n[INFO] Distribución de clases:")
    print(meta_df["phase_name"].value_counts().sort_index())

    print("\n[INFO] Muestras por atleta:")
    print(meta_df["athlete"].value_counts().sort_index())


if __name__ == "__main__":
    main()