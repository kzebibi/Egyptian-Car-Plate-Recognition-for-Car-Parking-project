import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self):
        self.model = YOLO('yolov9.pt')  # Assuming YOLOv9 uses similar syntax to YOLOv8

    def detect(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image)

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]  # Get the first detected plate
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            plate_image = image[y1:y2, x1:x2]
            return plate_image
        return None
