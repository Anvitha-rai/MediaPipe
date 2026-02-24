import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

class FaceDetector:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect(self, frame, rgb_frame):
        results = self.face_mesh.process(rgb_frame)
        message = ""

        if results.multi_face_landmarks:
            h, w, _ = frame.shape

            for face_landmarks in results.multi_face_landmarks:
                for landmark in face_landmarks.landmark:
                    x = int(landmark.x * w)
                    y = int(landmark.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

            message = "Face Keypoints Detected"

        return message