import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent  # project root
video_path = BASE_DIR / "simulation" / "test_clips" / "pathole_dashcam.mp4"

print("Base dir:", BASE_DIR)
print("Full video path:", video_path)
print("Exists?:", video_path.exists())

clips_dir = BASE_DIR / "simulation" / "test_clips"
print("\nContents of simulation/test_clips:")
if clips_dir.exists():
    print(os.listdir(clips_dir))
else:
    print("Folder does NOT exist")