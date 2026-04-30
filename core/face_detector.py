"""
face_detector.py
Detects faces using OpenCV's built-in Haar Cascade — no internet, no model download needed.
"""

import cv2
import numpy as np


class FaceDetector:
    """Face detector using OpenCV Haar Cascade (bundled with OpenCV)."""

    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 5):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

        self.detector = cv2.CascadeClassifier(cascade_path)
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, image: np.ndarray):
        """
        Detect faces in a BGR image.
        Returns list of dicts with keys: bbox (x, y, w, h), confidence
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )

        detections = []
        if len(faces) > 0:
            for (x, y, w, h) in faces:
                detections.append({
                    "bbox": (int(x), int(y), int(w), int(h)),
                    "confidence": 1.0,
                })
        return detections

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes on a copy of the image."""
        vis = image.copy()
        for d in detections:
            x, y, w, h = d["bbox"]
            cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 100), 2)
            cv2.putText(vis, "Face", (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 100), 2)
        return vis

    def close(self):
        pass
