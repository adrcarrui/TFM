import subprocess
from pathlib import Path

input_video = r"C:\Users\Adrian\Downloads\wl_videos\W-+86kg 2025 World Weightlifting Championships.mp4"
out_dir = Path(r"C:\Users\Adrian\OneDrive\Escritorio\UCJC\TFM\wl_clips\data\raw_videos")
out_dir.mkdir(parents=True, exist_ok=True)

output_clip = out_dir / "snatch_-+86kg_ri_song_gum_i3_ok_000163.mp4"

subprocess.run([
    "ffmpeg",
    "-ss", "00:45:58",
    "-to", "00:46:05",
    "-i", str(input_video),
    "-c:v", "libx264",
    "-preset", "veryfast",
    "-crf", "18",
    "-an",
    str(output_clip)
], check=True)
