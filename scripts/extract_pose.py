from pathlib import Path
import cv2
import pandas as pd
import mediapipe as mp

VIDEO_DIR = Path("data/raw_videos")
OUTPUT_DIR = Path("data/keypoints")
MODEL_PATH = "models/pose_landmarker_full.task"

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def process_video(video_path):

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("error opening", video_path)
        return

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1
    )

    rows = []

    with PoseLandmarker.create_from_options(options) as landmarker:

        frame_id = 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25

        while True:

            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp = int(frame_id / fps * 1000)

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            row = {"frame": frame_id}

            if result.pose_landmarks:

                landmarks = result.pose_landmarks[0]

                for i, lm in enumerate(landmarks):
                    row[f"x{i}"] = lm.x
                    row[f"y{i}"] = lm.y
                    row[f"z{i}"] = lm.z

            else:

                for i in range(33):
                    row[f"x{i}"] = None
                    row[f"y{i}"] = None
                    row[f"z{i}"] = None

            rows.append(row)
            frame_id += 1

    cap.release()

    out_path = OUTPUT_DIR / (video_path.stem + ".csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)

    print("saved pose:", out_path)


def main():

    videos = list(VIDEO_DIR.rglob("*.mp4"))

    for video in videos:
        process_video(video)


if __name__ == "__main__":
    main()