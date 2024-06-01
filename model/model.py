import torch
import tensorflow as tf
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


image = ''  # 추론할 이미지 경로
object_detector_pkt = ''  # 객체 감지 모델 가중치 경로
sp_h5 = ''  # 폴리머현수 분할 모델 가중치 경로
cn_h5 = ''  # 접속개소 분할 모델 가중치 경로
la_h5 = ''  # LA 분할 모델 가중치 경로
tr_h5 = ''  # TR 분할 모델 가중치 경로
lp_h5 = ''  # 폴리머LP 분할 모델 가중치 경로

object_detector, sp_classifier, cn_classifier, la_classifier, tr_classifier, lp_classifier = init_model(
    object_detector_pkt,
    sp_h5,
    cn_h5,
    la_h5,
    tr_h5,
    lp_h5)

bboxes = object_detector.predict(image)
