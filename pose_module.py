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
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def detect(self, frame, rgb_frame):
        results = self.pose.process(rgb_frame)
        message = ""
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )
            
            landmarks = results.pose_landmarks.landmark
            
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            right_wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            
            # Distances
            left_to_right_shoulder = self.distance(left_wrist, right_shoulder)
            right_to_left_shoulder = self.distance(right_wrist, left_shoulder)
            wrists_distance = self.distance(left_wrist, right_wrist)
            
            # Thresholds (bigger = easier detection)
            cross_threshold = 0.25
            wrist_close_threshold = 0.3
            hand_raised_margin = 0.05
            
            # Hands raised
            left_hand_raised = left_wrist.y < (left_shoulder.y - hand_raised_margin)
            right_hand_raised = right_wrist.y < (right_shoulder.y - hand_raised_margin)
            
            # -------------------------
            # ARMS CROSSED
            # -------------------------
            if (left_to_right_shoulder < cross_threshold and
                right_to_left_shoulder < cross_threshold and
                wrists_distance < wrist_close_threshold):
                
                message = "Arms Crossed"
            
            # -------------------------
            # VICTORY POSE
            # -------------------------
            elif left_hand_raised and right_hand_raised:
                message = "Victory Pose"
        
        return message