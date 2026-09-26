#!/usr/bin/env bash
# Render the shard_map transpose explainer at 720p30 and join the scenes into
# one mp4. Needs Manim Community Edition (pip install manim), ffmpeg, and the
# fonts CMU Serif and JetBrains Mono (Debian/Ubuntu: fonts-cmu,
# fonts-jetbrains-mono). Scenes render one at a time: parallel Manim runs can
# race on the shared text cache.
set -euo pipefail
cd "$(dirname "$0")"

SCENES="S1Devices S2Psum S3WholePieces S4Pvary S5Toy S6OldSeam S7DesignSpace"
for s in $SCENES; do
  manim -qm --disable_caching --progress_bar none shmap_transpose.py "$s"
done

: > scenes.txt
for s in $SCENES; do
  echo "file 'media/videos/shmap_transpose/720p30/$s.mp4'" >> scenes.txt
done
ffmpeg -y -f concat -safe 0 -i scenes.txt -c:v libx264 -preset slow -tune animation \
  -crf 19 -pix_fmt yuv420p -profile:v high -level 4.0 -movflags +faststart -an \
  shmap_transpose.mp4
