import mediapipe as mp
import math

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

class PoseDetector:
    def __init__(self):
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def distance(self, p1, p2):
        """Calculate Euclidean distance between two landmarks"""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def detect(self, frame, rgb_frame):
        results = self.pose.process(rgb_frame)
        message = ""
        
        if results.pose_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark
            
            # Get key landmarks
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            left_elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            right_elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            
            # Distance checks for arm crossing
            left_wrist_to_right_elbow = self.distance(left_wrist, right_elbow)
            right_wrist_to_left_elbow = self.distance(right_wrist, left_elbow)
            wrists_distance = self.distance(left_wrist, right_wrist)
            
            # Thresholds (adjust if needed)
            cross_threshold = 0.15
            wrist_close_threshold = 0.2
            hand_raised_margin = 0.05  # How much higher wrist needs to be above shoulder
            
            # Check if hands are raised
            left_hand_raised = left_wrist.y < (left_shoulder.y - hand_raised_margin)
            right_hand_raised = right_wrist.y < (right_shoulder.y - hand_raised_margin)
            
            # ------------------------- 
            # ARMS CROSSED
            # -------------------------
            # Check if wrists are close to opposite elbows AND wrists are close together
            if (left_wrist_to_right_elbow < cross_threshold and 
                right_wrist_to_left_elbow < cross_threshold and
                wrists_distance < wrist_close_threshold):
                message = "Arms Crossed"
            
            # -------------------------
            # VICTORY POSE (Both hands CLEARLY up)
            # -------------------------
            elif left_hand_raised and right_hand_raised:
                message = "Victory Pose"
            
            # -------------------------
            # LEFT HAND RAISED (and right hand NOT raised)
            # -------------------------
            elif left_hand_raised and not right_hand_raised:
                message = "Right Hand Raised"
            
            # -------------------------
            # RIGHT HAND RAISED (and left hand NOT raised)
            # -------------------------
            elif right_hand_raised and not left_hand_raised:
                message = "Left Hand Raised"
        
        return message