import os

from matplotlib import pyplot as plt
from pycocotools.coco import COCO

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")  # CUDA의 경로를 이곳에 입력
import tensorflow as tf
from tensorflow_datasets.core.features.image_feature import cv2
import numpy as np

h5_path = '../checkpoints/PoledeeplabV3plus_export.weights.h5'
images_dir = '../data/champions_league/dataset_pole/images/'
json_path = '../data/champions_league/dataset_pole/instances_default.json'
input_size = (512, 512)

coco = COCO(json_path)
images = []
for img_id in coco.getImgIds():
    img = coco.loadImgs(img_id)[0]
    file_name = images_dir + img['file_name']
    image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    image = cv2.resize(image, input_size)
    images.append(image)

images = np.array(images)

predictions = tf.keras.models.load_model(h5_path).predict(images)


def mask_to_label(mask):
    labels = np.argmax(mask, axis=-1)
    return labels

predicted_labels = mask_to_label(predictions)

# 이미지와 예측된 클래스 레이블을 시각화합니다.
for i in range(len(images)):
    # 원본 이미지를 표시합니다.
    plt.subplot(1, 2, 1)
    plt.imshow(images[i])
    plt.title('Original Image')
    plt.axis('off')

    # 예측된 클래스 레이블을 표시합니다.
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_labels[i], cmap='viridis')  # 클래스 레이블을 시각화합니다.
    plt.title('Predicted Labels')
    plt.axis('off')

    plt.show()
