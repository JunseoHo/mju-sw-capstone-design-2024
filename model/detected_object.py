import numpy as np
from PIL import Image


class DetectedObject:
    def __init__(self,
                 origin_image: list[list[list[int]]],
                 bbox_xyxy: list[float],
                 confidence: float,
                 label: int):
        self.bbox_xyxy = bbox_xyxy
        self.confidence = confidence
        self.label = label
        x1, y1, x2, y2 = map(int, bbox_xyxy)
        self.image = np.array(origin_image[y1:y2, x1:x2])

    def save(self, filename):
        Image.fromarray(self.image.astype('uint8')).save(filename)

