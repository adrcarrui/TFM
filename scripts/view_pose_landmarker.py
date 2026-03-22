from __future__ import annotations

from pathlib import Path
import cv2
import mediapipe as mp

VIDEO_PATH = Path("data/raw_videos/nasar/snatch_-96kg_nasar_karlos_i1_ok_000011.mp4")
MODEL_PATH = "models/pose_landmarker_full.task"

SAVE_OUTPUT_VIDEO = False
OUTPUT_VIDEO_PATH = Path("outputs/pose_preview.mp4")

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


# Conexiones tipo esqueleto para dibujar
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22),
    (11, 23), (12, 24),
    (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
]


def draw_landmarks(frame, landmarks, visibility_threshold: float = 0.3):
    h, w = frame.shape[:2]

    # Dibujar conexiones
    for start_idx, end_idx in POSE_CONNECTIONS:
        lm1 = landmarks[start_idx]
        lm2 = landmarks[end_idx]

        vis1 = getattr(lm1, "visibility", 1.0)
        vis2 = getattr(lm2, "visibility", 1.0)

        if vis1 < visibility_threshold or vis2 < visibility_threshold:
            continue

        x1, y1 = int(lm1.x * w), int(lm1.y * h)
        x2, y2 = int(lm2.x * w), int(lm2.y * h)

        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Dibujar puntos
    for i, lm in enumerate(landmarks):
        vis = getattr(lm, "visibility", 1.0)
        if vis < visibility_threshold:
            continue

        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        # opcional: índice del landmark
        # cv2.putText(frame, str(i), (x + 4, y - 4),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def main() -> None:
    if not VIDEO_PATH.exists():
        print(f"[ERROR] No existe el vídeo: {VIDEO_PATH}")
        return

    if not Path(MODEL_PATH).exists():
        print(f"[ERROR] No existe el modelo: {MODEL_PATH}")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        print(f"[ERROR] No se pudo abrir el vídeo: {VIDEO_PATH}")
        return

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

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    with PoseLandmarker.create_from_options(options) as landmarker:
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms = int((frame_idx / fps) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            display = frame.copy()

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                display = draw_landmarks(display, landmarks)

            cv2.putText(
                display,
                f"Frame: {frame_idx}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Pose Landmarker Preview", display)

            if writer is not None:
                writer.write(display)

            key = cv2.waitKey(20) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("p"):
                # pausa simple
                while True:
                    paused_key = cv2.waitKey(0) & 0xFF
                    if paused_key == ord("p"):
                        break
                    elif paused_key == ord("q"):
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
        print(f"[OK] Vídeo guardado en: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()