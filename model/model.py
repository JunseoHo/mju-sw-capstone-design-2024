import cv2
import numpy as np
import torch
import tensorflow as tf
from matplotlib import pyplot as plt
from super_gradients.training import models

classes = ['sp', 'cn', 'la', 'tr', 'lp']  # 클래스 리스트
image_path = 'sample_2.jpg'  # 추론할 이미지 경로
obj_dtt_name = 'yolo_nas_s'  # 객체 탐지 모델 이름
obj_pth = 'weights/object_detector.pth'  # 객체 탐지 모델 가중치 경로
sp_h5 = 'weights/tr_classifier.h5'  # 폴리머현수 분할 모델 가중치 경로
cn_h5 = 'weights/tr_classifier.h5'  # 접속개소 분할 모델 가중치 경로
la_h5 = 'weights/tr_classifier.h5'  # LA 분할 모델 가중치 경로
tr_h5 = 'weights/tr_classifier.h5'  # TR 분할 모델 가중치 경로
lp_h5 = 'weights/tr_classifier.h5'  # 폴리머LP 분할 모델 가중치 경로
margin = 300  # 바운딩 박스에 추가할 마진의 크기
confidence_threshold = 0.3  # 객체 탐지 컨피던스 스레숄드
segmentation_threshold = 0.2  # 이미지 분할 프로펜서티 스레숄드
input_size = (256, 256)  # 이미지 분할 모델의 입력 크기
mask_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

# 모델 초기화
obj_dtt = models.get(
    model_name=obj_dtt_name,
    checkpoint_path=obj_pth,
    num_classes=len(classes)
).to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
sp_clf = tf.keras.models.load_model(sp_h5)
cn_clf = tf.keras.models.load_model(cn_h5)
la_clf = tf.keras.models.load_model(la_h5)
tr_clf = tf.keras.models.load_model(tr_h5)
lp_clf = tf.keras.models.load_model(lp_h5)

# 이미지 불러오기
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 이미지의 객체 탐지
object_detection_result = obj_dtt.predict(image_path)

detected_object_list = object_detection_result.prediction
bounding_boxes = detected_object_list.bboxes_xyxy
labels = detected_object_list.labels
confidences = detected_object_list.confidence

# 인스턴스 분할
mask_infos = []
for bbox, label, confidence in zip(bounding_boxes, labels, confidences):
    if confidence < confidence_threshold:
        continue
    print(f"Object detected... {classes[label - 1]}, {bbox}, {confidence}")
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, image.shape[1])
    y2 = min(y2 + margin, image.shape[0])
    instance_image = cv2.resize(image[y1:y2, x1:x2], input_size)
    instance_image = np.expand_dims(instance_image, axis=0)
    if label == 4:
        propensity_mask = tr_clf.predict(instance_image)[0]
        binary_mask = (propensity_mask[:, :, 0] >= segmentation_threshold).astype(np.uint16)
        binary_mask = cv2.resize(binary_mask, (x2 - x1, y2 - y1))
        mask_infos.append(((x1, y1, x2, y2), binary_mask))

# 인스턴스 마스킹
mask_color_idx = 0
for mask_info in mask_infos:
    (x1, y1, x2, y2), mask = mask_info

    indices = np.where(mask == 1)
    indices_y = indices[0] + y1
    indices_x = indices[1] + x1

    valid_indices = (indices_y < image.shape[0]) & (indices_x < image.shape[1])
    indices_y = indices_y[valid_indices]
    indices_x = indices_x[valid_indices]

    mask_color = np.array(mask_colors[mask_color_idx], dtype=np.uint8)
    mask_color_idx = 0 if mask_color_idx == len(mask_colors) - 1 else mask_color_idx + 1
    for y, x in zip(indices_y, indices_x):
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            original_pixel = image[y, x].astype(np.float32)
            blended_pixel = (original_pixel * 0.5 + mask_color.astype(np.float32) * 0.5).astype(np.uint8)
            image[y, x] = blended_pixel

# 결과 이미지 표시
plt.imshow(image)
plt.show()
