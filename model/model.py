import cv2
import numpy as np
import torch
import tensorflow as tf
from matplotlib import pyplot as plt
from super_gradients.training import models

classes = ['sp', 'cn', 'la', 'tr', 'lp']


def init_model(object_detector_pkt, sp_h5, cn_h5, la_h5, tr_h5, lp_h5):
    object_detector = models.get(
        model_name='yolo_nas_s',
        checkpoint_path=object_detector_pkt,
        num_classes=len(classes)
    ).to(torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu"))
    sp_classifier = tf.keras.models.load_model(sp_h5)
    cn_classifier = tf.keras.models.load_model(cn_h5)
    la_classifier = tf.keras.models.load_model(la_h5)
    tr_classifier = tf.keras.models.load_model(tr_h5)
    lp_classifier = tf.keras.models.load_model(lp_h5)
    return object_detector, sp_classifier, cn_classifier, la_classifier, tr_classifier, lp_classifier


image = 'sample_3.jpg'  # 추론할 이미지 경로
object_detector_pth = 'weights/object_detector.pth'  # 객체 감지 모델 가중치 경로
sp_h5 = 'weights/tr_classifier.h5'  # 폴리머현수 분할 모델 가중치 경로
cn_h5 = 'weights/tr_classifier.h5'  # 접속개소 분할 모델 가중치 경로
la_h5 = 'weights/tr_classifier.h5'  # LA 분할 모델 가중치 경로
tr_h5 = 'weights/tr_classifier.h5'  # TR 분할 모델 가중치 경로
lp_h5 = 'weights/tr_classifier.h5'  # 폴리머LP 분할 모델 가중치 경로
margin = 300
object_detection_threshold = 0.7
segmenatation_threshold = 0.001
input_size = (256, 256)

object_detector, sp_classifier, cn_classifier, la_classifier, tr_classifier, lp_classifier = init_model(
    object_detector_pth,
    sp_h5,
    cn_h5,
    la_h5,
    tr_h5,
    lp_h5)

object_detection = object_detector.predict(image)
instance_masks = []


def mask_to_label(mask):
    print(mask)
    binary_mask = (mask >= segmenatation_threshold).astype(np.uint8)
    return binary_mask


for bbox, label, confidence in zip(object_detection.prediction.bboxes_xyxy, object_detection.prediction.labels,
                                   object_detection.prediction.confidence):
    if confidence < object_detection_threshold:
        continue
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(x1 - margin, 0)
    y1 = max(y1 - margin, 0)
    x2 = min(x2 + margin, object_detection.image.shape[1])
    y2 = min(y2 + margin, object_detection.image.shape[0])
    instance_image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
    instance_image = cv2.resize(instance_image[y1:y2, x1:x2], input_size)
    instance_image = np.expand_dims(instance_image, axis=0)
    instance_mask = mask_to_label(tr_classifier.predict(instance_image)[0])
    instance_mask = np.reshape(instance_mask, input_size)
    instance_mask_resized = cv2.resize(instance_mask, (x2 - x1, y2 - y1))

    mask_resized = cv2.resize(instance_mask, (x2 - x1, y2 - y1))
    mask_binary = mask_resized

    original_image = cv2.imread(image)
    overlay = original_image.copy()

    overlay[y1:y2, x1:x2, 0] = np.where(mask_binary == 1, 0, overlay[y1:y2, x1:x2, 0])  # B 채널
    overlay[y1:y2, x1:x2, 1] = np.where(mask_binary == 1, 0, overlay[y1:y2, x1:x2, 1])  # G 채널
    overlay[y1:y2, x1:x2, 2] = np.where(mask_binary == 1, 255, overlay[y1:y2, x1:x2, 2])  # R 채널

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, original_image, 1 - alpha, 0, original_image)

    plt.figure()
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.show()

    instance_masks.append(mask_resized)
