from pathlib import Path
import re

ROOT = Path(r"C:\Users\Adrian\OneDrive\Escritorio\UCJC\TFM\wl_clips\data\raw_videos")

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}

pattern = re.compile(
    r".*_(\d+)$"   # captura el número final
)

clips = []

for file in ROOT.rglob("*"):
    if file.is_file() and file.suffix.lower() in VIDEO_EXTS:

        match = pattern.match(file.stem)

        if match:
            num = int(match.group(1))
            clips.append((num, file.name))

# ordenar por número global
clips.sort()

print("\nSECUENCIA GLOBAL DE CLIPS\n")

for num, name in clips:
    print(f"{num:06d}  {name}")