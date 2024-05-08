from super_gradients.training import models
import torch

# 테스트 이미지 경로
test_image_path = 'sample.jpg'

# YOLONAS로 바운딩 박스 예측
model = models.get(
    model_name='yolo_nas_s',
    checkpoint_path='./checkpoints/yolonas/ckpt_best.pth',
    num_classes=2
).to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))

out = model.predict('sample.jpg')

image = out.image  # YOLONAS에 입력된 이미지 픽셀 값
bboxes = out.prediction.bboxes_xyxy  # YOLONAS에서 예측된 바운딩 박스 값

print(bboxes)
