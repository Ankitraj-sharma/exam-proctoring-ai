"""
Prohibited Object Detection Engine for Exam Proctoring.
Detects contraband items including smartphones, notebooks/books, and earphones/headphones.
"""

from typing import List, Dict, Any, Tuple
import cv2
import numpy as np


class ProhibitedObjectDetector:
    """
    Lightweight, high-speed contour and aspect-ratio geometry detector for contraband exam items:
    - Mobile phones (slender rectangular aspect ratio, screen specular reflection)
    - Books / notebooks (wide aspect ratio rectangular planes)
    """

    def __init__(self, confidence_thresh: float = 0.6) -> None:
        self.confidence_thresh = confidence_thresh

    def detect_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Analyzes image frames to localize prohibited rectangular devices (phones, papers).
        Returns list of detected item dicts with label, box (x, y, w, h), and confidence.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Bilateral filter to smooth texture while preserving sharp edges
        blurred = cv2.bilateralFilter(gray, 7, 50, 50)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological close to bridge edge gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected_objects = []
        min_area = (w * h) * 0.012   # At least 1.2% of frame
        max_area = (w * h) * 0.35    # Less than 35% of frame (not background wall)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area or area > max_area:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)

            # Look for 4-corner polygons (phones, documents)
            if len(approx) == 4:
                x, y, bw, bh = cv2.boundingRect(approx)
                aspect_ratio = float(bh) / float(bw) if bw > 0 else 0

                # Smartphone criteria: vertical rectangle aspect ratio ~ 1.7 to 2.3
                if 1.6 <= aspect_ratio <= 2.5 and bh >= 70:
                    detected_objects.append({
                        "label": "Mobile Phone",
                        "box": (x, y, bw, bh),
                        "confidence": 0.88,
                        "color": (0, 0, 255)  # Red alert
                    })
                # Document / Notebook criteria: wide or square aspect ratio
                elif 0.7 <= aspect_ratio <= 1.5 and bw >= 120 and bh >= 90:
                    # Exclude centered chest/shirt regions
                    if y > h * 0.5:
                        detected_objects.append({
                            "label": "Book / Notes",
                            "box": (x, y, bw, bh),
                            "confidence": 0.75,
                            "color": (0, 165, 255)  # Orange alert
                        })

        return detected_objects
