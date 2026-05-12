import cv2
import os
import sys

input_dir = sys.argv[1] if len(sys.argv) > 1 else "resultatsim"
output_file = sys.argv[2] if len(sys.argv) > 2 else "simulation.mp4"
fps = 30

# Get all jpg files sorted numerically
files = [f for f in os.listdir(input_dir) if f.endswith(".jpg")]
files.sort(key=lambda x: int(x.split(".")[0]))

if not files:
    print(f"No .jpg files found in {input_dir}")
    sys.exit(1)

# Read first frame to get dimensions
first = cv2.imread(os.path.join(input_dir, files[0]))
h, w = first.shape[:2]

print(f"Creating video: {len(files)} frames, {w}x{h}, {fps} fps")
print(f"Output: {output_file}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(output_file, fourcc, fps, (w, h))

for i, f in enumerate(files):
    frame = cv2.imread(os.path.join(input_dir, f))
    writer.write(frame)
    if (i + 1) % 100 == 0:
        print(f"  {i + 1}/{len(files)} frames written")

writer.release()
print(f"Done! Video saved to {output_file}")
