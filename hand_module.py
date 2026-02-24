import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class HandDetector:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame, rgb_frame):
        results = self.hands.process(rgb_frame)
        message = ""

        if results.multi_hand_landmarks and results.multi_handedness:

            for hand_landmarks, handedness in zip(
                    results.multi_hand_landmarks,
                    results.multi_handedness):

                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Detect if fingers are open
                landmarks = hand_landmarks.landmark

                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                ring_tip = landmarks[16]
                pinky_tip = landmarks[20]

                index_pip = landmarks[6]
                middle_pip = landmarks[10]
                ring_pip = landmarks[14]
                pinky_pip = landmarks[18]

                fingers_open = 0

                if index_tip.y < index_pip.y:
                    fingers_open += 1
                if middle_tip.y < middle_pip.y:
                    fingers_open += 1
                if ring_tip.y < ring_pip.y:
                    fingers_open += 1
                if pinky_tip.y < pinky_pip.y:
                    fingers_open += 1

                hand_label = handedness.classification[0].label

                if fingers_open == 4:
                    if hand_label == "Left":
                        message = "Left Hand Raised"
                    elif hand_label == "Right":
                        message = "Right Hand Raised"

        return message