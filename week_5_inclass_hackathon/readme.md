# Eye Blink Rate Comparison Tool

This project compares spontaneous eye blink rates across two different activities: **Reading a Document** and **Watching a Video**.

## How It Works
The application uses a real-time Computer Vision pipeline to monitor your face while you interact with native media.
* **Detection:** Uses Google’s **MediaPipe Tasks API** (Neural Network Blendshapes) to detect eyelid closure confidence.
* **Context:** Launches the PDF and Video in their **native macOS viewers** (Preview/QuickTime) to ensure a natural reading and viewing experience.
* **Monitoring:** A small, compact webcam overlay tracks your blinks in the corner of the screen.
* **Analysis:** Automatically calculates total time, total blinks, and **Blinks Per Minute (BPM)** for each task.


## Prerequisites
Install the core computer vision and plotting libraries:
```bash
pip install opencv-python mediapipe numpy matplotlib