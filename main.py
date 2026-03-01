import cv2
from face_module import FaceDetector
from hand_module import HandDetector
from pose_module import PoseDetector
from iris_module import IrisDetector

# Initialize all modules
face_detector = FaceDetector()
hand_detector = HandDetector()
pose_detector = PoseDetector()
iris_detector = IrisDetector()

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Flip the frame horizontally (mirror image)
    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Call each module
    face_msg = face_detector.detect(frame, rgb_frame)
    hand_msg = hand_detector.detect(frame, rgb_frame)
    pose_msg = pose_detector.detect(frame, rgb_frame)
    iris_msg = iris_detector.detect(frame, rgb_frame)

    # Store messages in list
    messages = list(filter(None, [
        face_msg,
        hand_msg,
        pose_msg,
        iris_msg
    ]))

    # Display messages line by line
    y_position = 40
    for msg in messages:
        cv2.putText(
            frame,
            msg,
            (10, y_position),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA
        )
        y_position += 30

    cv2.imshow("MediaPipe System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()