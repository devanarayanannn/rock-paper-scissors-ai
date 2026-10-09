# rock-paper-scissors-ai
A Rock Paper Scissors Ai game built with Python,OpenCV, and MediaPipe. It detects hand gestures through webcam, lets you play against the computer, determines the winner, and keeps score
## Model File

This project requires the `hand_landmarker.task` model file in the project folder to detect hand gestures.

Download a compatible model from the official [MediaPipe documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker/python).

Check the model's terms before redistributing it.

## How to Run

1. Install Python 3.13.
2. Install the required libraries by running:

   ```bash
   py -3.13 -m pip install opencv-python mediapipe
   ```

3. Place `hand_landmarker.task` in the same folder as `hand_detection.py`.
4. Run the game:

   ```bash
   py -3.13 hand_detection.py
   ```

## Controls

- **Q** — Quit the game.
- **R** — Reset the score.
