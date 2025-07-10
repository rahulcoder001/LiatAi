import cv2
import numpy as np
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):
        # Load the YOLO model directly from ultralytics
        self.model = YOLO(model_path)
        print("Model loaded successfully. Class names:", self.model.names)
    
    def detect(self, frame):
        # Run inference
        results = self.model(frame, conf=0.7, classes=[0])  # Only detect players (class 0)
        
        # Extract detections
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls = box.cls.item()
                # Only include players (class 0)
                if cls == 0:
                    detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections)