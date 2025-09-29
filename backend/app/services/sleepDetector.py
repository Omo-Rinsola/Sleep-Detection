"""
wraps up the sleep detection logic
"""

import numpy as np
import cv2
from .eye_landmark import EyeLandmark
from .earCalculator import EarCalculator


class Detect:
    def __init__(self):
        self.eye_landmark = EyeLandmark()
        self.ear_calculator = EarCalculator()
        self.threshold = 0.1

    def detect(self, frame):
        frame.flags.writeable = False
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.eye_landmark.face_mesh.process(frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_points, right_eye_points = self.eye_landmark.get_eye_landmarks(frame, face_landmarks)

                # Calculate the eye aspect ratio (EAR)
                left_ear = self.ear_calculator.calculate_ear(left_eye_points)
                right_ear = self.ear_calculator.calculate_ear(right_eye_points)

                # calculate average EAR
                average = self.ear_calculator.average_ear(left_ear, right_ear)

                # threshold
                if average < self.threshold:
                    return "Sleeping"
                else:
                    return "Awake"

        return "No face detected"
                                                                                                                                                                                 