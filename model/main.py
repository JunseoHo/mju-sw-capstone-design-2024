from super_gradients.training import models
import torch

from model.detected_object import DetectedObject

# 테스트 이미지 경로
test_image_path = 'sample.jpg'

# YOLONAS로 바운딩 박스 예측
model = models.get(
    model_name='yolo_nas_s',
    checkpoint_path='./checkpoints/yolonas/ckpt_best.pth',
    num_classes=2
).to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

out = model.predict('sample.jpg')

detected_objects = []
for index in range(len(out.prediction.bboxes_xyxy)):
    detected_objects.append(DetectedObject(out.image,
                                           out.prediction.bboxes_xyxy[index],
                                           out.prediction.confidence[index],
                                           out.prediction.labels[index]))

for index, detected_object in enumerate(detected_objects):
    detected_object.save(f'detected_object_{index}.png')

