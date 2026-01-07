import subprocess
from pathlib import Path

input_video = r"C:\Users\Adrian\Downloads\Men's -94kg 2025 World Weightlifting Championships _ Full Session.mp4"
out_dir = Path(r"C:\Users\Adrian\Downloads\wl_clips")
out_dir.mkdir(parents=True, exist_ok=True)

output_clip = out_dir / "snatch_-96kg_moeini_sedeh_alireza_i3_ok_000018.mp4"

subprocess.run([
    "ffmpeg",
    "-ss", "00:27:41",
    "-to", "00:27:50",
    "-i", str(input_video),
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "18",
    "-an",
    str(output_clip)
], check=True)
