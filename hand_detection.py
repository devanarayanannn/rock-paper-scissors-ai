

import cv2
import mediapipe as mp
import time
import random

# 1. Set up MediaPipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)


# 2. Recognize hand gestures
def get_gesture(hand):
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    fingers_up = []

    for tip, pip in zip(finger_tips, finger_pips):
        fingers_up.append(hand[tip].y < hand[pip].y)

    if fingers_up == [False, False, False, False]:
        return "ROCK"
    elif fingers_up == [True, True, True, True]:
        return "PAPER"
    elif fingers_up == [True, True, False, False]:
        return "SCISSORS"
    else:
        return "UNKNOWN"


# 3. Computer randomly chooses a move
def get_computer_move():
    return random.choice(["ROCK", "PAPER", "SCISSORS"])


# 4. Decide the winner
def find_winner(player, computer):
    if player == computer:
        return "DRAW!"

    elif (
        (player == "ROCK" and computer == "SCISSORS")
        or (player == "PAPER" and computer == "ROCK")
        or (player == "SCISSORS" and computer == "PAPER")
    ):
        return "YOU WIN!"

    return "COMPUTER WINS!"


# 5. Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access webcam.")
    raise SystemExit

player_score = 0
computer_score = 0

computer_move = "---"
game_result = "Show ROCK, PAPER or SCISSORS"

last_gesture = "UNKNOWN"
last_play_time = 0.0
last_round_gesture = None

print("Rock Paper Scissors game started!")
print("Show your hand to play.")
print("Press R to reset the score.")
print("Press Q to quit.")

try:
    with HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                print("Could not read webcam.")
                break

            # Mirror the camera image
            frame = cv2.flip(frame, 1)

            # Convert image to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(
                frame, cv2.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp = time.monotonic_ns() // 1_000_000
            result = detector.detect_for_video(
                mp_image, timestamp
            )

            current_gesture = "UNKNOWN"

            # Detect hand and draw landmarks
            if result.hand_landmarks:
                hand = result.hand_landmarks[0]
                points = []

                for landmark in hand:
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    points.append((x, y))

                for connection in (
                    mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                ):
                    start = connection.start
                    end = connection.end

                    cv2.line(
                        frame,
                        points[start],
                        points[end],
                        (0, 255, 0),
                        2
                    )

                for x, y in points:
                    cv2.circle(
                        frame, (x, y), 4, (0, 0, 255), -1
                    )

                current_gesture = get_gesture(hand)

            # Play one round when a recognized gesture changes
            now = time.monotonic()

            if (
                current_gesture in ["ROCK", "PAPER", "SCISSORS"]
                and current_gesture != last_gesture
                and now - last_play_time >= 1.0
            ):
                computer_move = get_computer_move()
                game_result = find_winner(
                    current_gesture, computer_move
                )

                if game_result == "YOU WIN!":
                    player_score += 1
                elif game_result == "COMPUTER WINS!":
                    computer_score += 1

                last_play_time = now

            last_gesture = current_gesture

            # 6. Display the game information
            cv2.putText(
                frame,
                "Your move: " + current_gesture,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "Computer: " + computer_move,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 200, 0),
                2
            )

            cv2.putText(
                frame,
                "Result: " + game_result,
                (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"You: {player_score}  Computer: {computer_score}",
                (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "R: Reset Score | Q: Quit",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.imshow("Rock Paper Scissors AI Game", frame)

            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF

            if key == ord("r") or key == ord("R"):
                player_score = 0
                computer_score = 0
                computer_move = "---"
                game_result = "New game! Show your hand"
                last_gesture = "UNKNOWN"
                last_play_time = 0.0

            elif key == ord("q") or key == ord("Q"):
                break

finally:
    cap.release()
    cv2.destroyAllWindows()

print("Game closed.")
print("Final score:")
print("You:", player_score)
print("Computer:", computer_score)
