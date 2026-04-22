# Blink & Facial Dimension Analysis

This project analyzes videos (recorded during a research study)ss to extract:
- Blink count and blink rate
- Basic facial geometry (head, eyes, nose, mouth dimensions)

It uses MediaPipe Face Landmarker for facial tracking and OpenCV for video processing and visualization.

---

## Project Structure

project/
│── all_videos/            #  .MP4 videos
│── face_landmarker.task    # MediaPipe model
│── main_blink_detection.ipynb   # main script

---
## Requirements

Install dependencies:

pip install opencv-python mediapipe numpy matplotlib

---