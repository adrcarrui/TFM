from __future__ import annotations

import csv
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import pandas as pd


VIDEO_DIR = Path("data/raw_videos")
ANNOTATIONS_DIR = Path("data/annotations")
FRAME_LABELS_DIR = ANNOTATIONS_DIR / "frame_labels"
SEGMENT_LABELS_DIR = ANNOTATIONS_DIR / "segment_labels"
MASTER_FRAME_CSV = ANNOTATIONS_DIR / "master_frame_labels.csv"
MASTER_SEGMENT_CSV = ANNOTATIONS_DIR / "master_segment_labels.csv"

PHASES: Dict[int, str] = {
    0: "unlabeled",
    1: "setup",
    2: "first_pull",
    3: "transition",
    4: "second_pull",
    5: "turnover",
    6: "catch",
    7: "recovery",
}


def list_videos(video_dir: Path) -> List[Path]:
    return sorted(video_dir.rglob("*.mp4"))


def load_video_frames(video_path: Path) -> List:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el vídeo: {video_path}")

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def get_relative_video_key(video_path: Path) -> str:
    return str(video_path.relative_to(VIDEO_DIR)).replace("\\", "/")


def get_frame_label_csv_path(video_path: Path) -> Path:
    relative = video_path.relative_to(VIDEO_DIR)
    return FRAME_LABELS_DIR / relative.with_suffix(".csv")


def get_segment_label_csv_path(video_path: Path) -> Path:
    relative = video_path.relative_to(VIDEO_DIR)
    return SEGMENT_LABELS_DIR / relative.with_suffix(".csv")


def load_existing_labels(video_path: Path, num_frames: int) -> List[int]:
    csv_path = get_frame_label_csv_path(video_path)

    if not csv_path.exists():
        return [0] * num_frames

    df = pd.read_csv(csv_path)
    labels = [0] * num_frames

    for _, row in df.iterrows():
        frame_idx = int(row["frame"])
        phase_id = int(row["phase_id"])
        if 0 <= frame_idx < num_frames:
            labels[frame_idx] = phase_id

    return labels


def save_frame_labels(video_path: Path, labels: List[int]) -> None:
    csv_path = get_frame_label_csv_path(video_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    relative_video = get_relative_video_key(video_path)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "video_relpath", "frame", "phase_id", "phase_name"])

        for frame_idx, phase_id in enumerate(labels):
            writer.writerow([
                video_path.name,
                relative_video,
                frame_idx,
                phase_id,
                PHASES[phase_id],
            ])

    print(f"[OK] Guardado frame labels en: {csv_path}")


def build_segments(labels: List[int]) -> List[Tuple[int, int, int]]:
    if not labels:
        return []

    segments: List[Tuple[int, int, int]] = []
    start = 0
    current = labels[0]

    for i in range(1, len(labels)):
        if labels[i] != current:
            segments.append((start, i - 1, current))
            start = i
            current = labels[i]

    segments.append((start, len(labels) - 1, current))
    return segments


def save_segment_labels(video_path: Path, labels: List[int]) -> None:
    csv_path = get_segment_label_csv_path(video_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    relative_video = get_relative_video_key(video_path)
    segments = build_segments(labels)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "video_relpath", "start_frame", "end_frame", "phase_id", "phase_name"])

        for start_frame, end_frame, phase_id in segments:
            writer.writerow([
                video_path.name,
                relative_video,
                start_frame,
                end_frame,
                phase_id,
                PHASES[phase_id],
            ])

    print(f"[OK] Guardado segment labels en: {csv_path}")


def assign_range(labels: List[int], start_idx: int, end_idx: int, phase_id: int) -> None:
    lo = min(start_idx, end_idx)
    hi = max(start_idx, end_idx)
    for i in range(lo, hi + 1):
        labels[i] = phase_id


def summarize_segments(labels: List[int], max_items: int = 12) -> List[str]:
    segments = build_segments(labels)
    lines = []

    for start, end, phase_id in segments[:max_items]:
        lines.append(f"{start:03d}-{end:03d}: {PHASES[phase_id]}")

    if len(segments) > max_items:
        lines.append("...")

    return lines


def push_history(history: List[List[int]], labels: List[int], max_history: int = 50) -> None:
    history.append(deepcopy(labels))
    if len(history) > max_history:
        history.pop(0)


def undo_last_action(history: List[List[int]], labels: List[int]) -> bool:
    if not history:
        return False

    previous = history.pop()
    labels[:] = previous
    return True


def fill_unlabeled_gaps(labels: List[int]) -> int:
    """
    Rellena huecos unlabeled (0) SOLO si están entre dos segmentos con la misma etiqueta.
    Ejemplo:
    1 1 0 0 1 1  -> rellena los 0 con 1

    Devuelve cuántos frames fueron rellenados.
    """
    filled = 0
    n = len(labels)
    i = 0

    while i < n:
        if labels[i] != 0:
            i += 1
            continue

        start = i
        while i < n and labels[i] == 0:
            i += 1
        end = i - 1

        left_label = labels[start - 1] if start - 1 >= 0 else None
        right_label = labels[i] if i < n else None

        if left_label is not None and right_label is not None and left_label == right_label and left_label != 0:
            for j in range(start, end + 1):
                labels[j] = left_label
                filled += 1

    return filled


def export_master_csvs() -> None:
    ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

    frame_files = sorted(FRAME_LABELS_DIR.rglob("*.csv"))
    segment_files = sorted(SEGMENT_LABELS_DIR.rglob("*.csv"))

    frame_dfs = []
    for csv_path in frame_files:
        try:
            df = pd.read_csv(csv_path)
            frame_dfs.append(df)
        except Exception as e:
            print(f"[WARN] No se pudo leer {csv_path}: {e}")

    segment_dfs = []
    for csv_path in segment_files:
        try:
            df = pd.read_csv(csv_path)
            segment_dfs.append(df)
        except Exception as e:
            print(f"[WARN] No se pudo leer {csv_path}: {e}")

    if frame_dfs:
        master_frame_df = pd.concat(frame_dfs, ignore_index=True)
        master_frame_df.to_csv(MASTER_FRAME_CSV, index=False)
        print(f"[OK] CSV maestro de frames exportado en: {MASTER_FRAME_CSV}")
    else:
        print("[INFO] No hay frame CSVs para exportar.")

    if segment_dfs:
        master_segment_df = pd.concat(segment_dfs, ignore_index=True)
        master_segment_df.to_csv(MASTER_SEGMENT_CSV, index=False)
        print(f"[OK] CSV maestro de segmentos exportado en: {MASTER_SEGMENT_CSV}")
    else:
        print("[INFO] No hay segment CSVs para exportar.")


def draw_overlay(
    frame,
    video_name: str,
    frame_idx: int,
    total_frames: int,
    current_phase_id: int,
    labels: List[int],
    range_start: Optional[int],
    range_end: Optional[int],
):
    display = frame.copy()
    h, w = display.shape[:2]

    assigned_count = sum(1 for x in labels if x != 0)
    current_phase_name = PHASES[current_phase_id]

    left_panel_lines = [
        f"Video: {video_name}",
        f"Frame: {frame_idx + 1}/{total_frames}",
        f"Phase actual: {current_phase_id} - {current_phase_name}",
        f"Frames etiquetados: {assigned_count}/{total_frames}",
        "",
        f"Inicio rango: {range_start if range_start is not None else '-'}",
        f"Fin rango: {range_end if range_end is not None else '-'}",
        "",
        "Controles:",
        "a/d -> prev/next",
        "j/l -> -10/+10",
        "p -> play/pause",
        "g -> go to frame",
        "i -> set range start",
        "f -> set range end",
        "u -> clear selected range",
        "h -> fill unlabeled gaps",
        "z -> undo",
        "s -> save current video",
        "m -> export master csvs",
        "q -> quit",
        "",
        "0 -> unlabeled",
        "1 -> setup",
        "2 -> first_pull",
        "3 -> transition",
        "4 -> second_pull",
        "5 -> turnover",
        "6 -> catch",
        "7 -> recovery",
    ]

    right_panel_lines = ["Segmentos actuales:"]
    right_panel_lines.extend(summarize_segments(labels, max_items=14))

    overlay = display.copy()
    cv2.rectangle(overlay, (10, 10), (500, 610), (0, 0, 0), -1)
    cv2.rectangle(overlay, (w - 340, 10), (w - 10, 360), (0, 0, 0), -1)
    alpha = 0.62
    cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)

    y = 35
    for line in left_panel_lines:
        cv2.putText(
            display,
            line,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 21

    y = 35
    for line in right_panel_lines:
        cv2.putText(
            display,
            line,
            (w - 330, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += 21

    bar_x1 = 20
    bar_x2 = w - 20
    bar_y = h - 25

    cv2.line(display, (bar_x1, bar_y), (bar_x2, bar_y), (200, 200, 200), 2)
    progress_x = int(bar_x1 + (frame_idx / max(total_frames - 1, 1)) * (bar_x2 - bar_x1))
    cv2.circle(display, (progress_x, bar_y), 6, (0, 255, 255), -1)

    if range_start is not None and range_end is not None:
        lo = min(range_start, range_end)
        hi = max(range_start, range_end)
        x1 = int(bar_x1 + (lo / max(total_frames - 1, 1)) * (bar_x2 - bar_x1))
        x2 = int(bar_x1 + (hi / max(total_frames - 1, 1)) * (bar_x2 - bar_x1))
        cv2.line(display, (x1, bar_y), (x2, bar_y), (0, 0, 255), 5)

    elif range_start is not None:
        x = int(bar_x1 + (range_start / max(total_frames - 1, 1)) * (bar_x2 - bar_x1))
        cv2.circle(display, (x, bar_y), 8, (0, 0, 255), -1)

    if range_start is not None and range_end is not None:
        lo = min(range_start, range_end)
        hi = max(range_start, range_end)
        if lo <= frame_idx <= hi:
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

    return display


def annotate_video(video_path: Path) -> None:
    print(f"\n[INFO] Cargando vídeo: {video_path}")
    frames = load_video_frames(video_path)
    total_frames = len(frames)

    if total_frames == 0:
        print("[ERROR] El vídeo no tiene frames.")
        return

    labels = load_existing_labels(video_path, total_frames)

    frame_idx = 0
    playing = False
    range_start: Optional[int] = None
    range_end: Optional[int] = None
    history: List[List[int]] = []

    window_name = "Snatch Phase Annotator - Final"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        frame = frames[frame_idx]
        phase_id = labels[frame_idx]

        display = draw_overlay(
            frame=frame,
            video_name=video_path.name,
            frame_idx=frame_idx,
            total_frames=total_frames,
            current_phase_id=phase_id,
            labels=labels,
            range_start=range_start,
            range_end=range_end,
        )

        cv2.imshow(window_name, display)

        wait_time = 30 if playing else 0
        key = cv2.waitKey(wait_time) & 0xFF

        if playing:
            frame_idx = min(frame_idx + 1, total_frames - 1)
            if frame_idx == total_frames - 1:
                playing = False

        if key == 255:
            continue

        if key == ord("q"):
            print("[INFO] Cerrando anotador.")
            break

        elif key == ord("d"):
            frame_idx = min(frame_idx + 1, total_frames - 1)

        elif key == ord("a"):
            frame_idx = max(frame_idx - 1, 0)

        elif key == ord("l"):
            frame_idx = min(frame_idx + 10, total_frames - 1)

        elif key == ord("j"):
            frame_idx = max(frame_idx - 10, 0)

        elif key == ord("p"):
            playing = not playing

        elif key == ord("g"):
            target = input(f"Ir al frame (0-{total_frames - 1}): ").strip()
            if target.isdigit():
                frame_idx = min(max(int(target), 0), total_frames - 1)

        elif key == ord("i"):
            range_start = frame_idx
            print(f"[INFO] Inicio de rango = {range_start}")

        elif key == ord("f"):
            range_end = frame_idx
            print(f"[INFO] Fin de rango = {range_end}")

        elif key == ord("u"):
            if range_start is not None and range_end is not None:
                push_history(history, labels)
                lo = min(range_start, range_end)
                hi = max(range_start, range_end)
                assign_range(labels, lo, hi, 0)
                print(f"[INFO] Rango limpiado: {lo}-{hi}")
            range_start = None
            range_end = None

        elif key == ord("h"):
            push_history(history, labels)
            filled = fill_unlabeled_gaps(labels)
            print(f"[INFO] Huecos rellenos: {filled} frames")

        elif key == ord("z"):
            ok = undo_last_action(history, labels)
            if ok:
                print("[INFO] Última acción deshecha.")
            else:
                print("[INFO] No hay acciones para deshacer.")

        elif key == ord("s"):
            save_frame_labels(video_path, labels)
            save_segment_labels(video_path, labels)

        elif key == ord("m"):
            export_master_csvs()

        elif key in [ord(str(n)) for n in range(8)]:
            selected_phase = int(chr(key))

            if range_start is not None and range_end is not None:
                push_history(history, labels)
                lo = min(range_start, range_end)
                hi = max(range_start, range_end)
                assign_range(labels, lo, hi, selected_phase)
                print(f"[INFO] Asignado rango {lo}-{hi} -> {PHASES[selected_phase]}")
                frame_idx = hi
                range_start = None
                range_end = None
            else:
                push_history(history, labels)
                labels[frame_idx] = selected_phase
                print(f"[INFO] Asignado frame {frame_idx} -> {PHASES[selected_phase]}")
                if frame_idx < total_frames - 1:
                    frame_idx += 1

    cv2.destroyAllWindows()


def main() -> None:
    videos = list_videos(VIDEO_DIR)

    if not videos:
        print(f"No se encontraron vídeos en {VIDEO_DIR}")
        return

    print("Vídeos encontrados:")
    for i, video in enumerate(videos):
        print(f"{i:02d} -> {video.relative_to(VIDEO_DIR)}")

    selected = input("\nIntroduce el índice del vídeo a anotar: ").strip()
    if not selected.isdigit():
        print("Índice no válido.")
        return

    idx = int(selected)
    if idx < 0 or idx >= len(videos):
        print("Índice fuera de rango.")
        return

    annotate_video(videos[idx])


if __name__ == "__main__":
    main()
