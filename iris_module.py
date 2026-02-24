import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class IrisDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # MUST be True for iris
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        message = ""

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:

                landmarks = face_landmarks.landmark
                h, w, _ = frame.shape

                LEFT_IRIS = [474, 475, 476, 477]
                RIGHT_IRIS = [469, 470, 471, 472]

                for idx in LEFT_IRIS:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                for idx in RIGHT_IRIS:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                message = "Iris Detected"

        return message