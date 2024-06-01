import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")  # CUDA의 경로를 이곳에 입력

from common.data_loader import load_instances_coco_format_all
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

images_dir = '../test_dataset/images/'  # 전체 이미지가 저장되어 있는 디렉토리의 상대 주소
json_path = '../test_dataset/annotations/instances_default.json'  # COCO Json 파일의 상대 주소
h5_path = "../checkpoints/epoch-61-val_loss-030.h5"  # 학습된 모델의 저장 위치
input_size = (256, 256)  # 모델의 입력 크기
num_of_image = 100
cat_id = 4
threshold = 0.5
batch_size = 3

instance_images, instance_masks = load_instances_coco_format_all(images_dir, json_path, input_size, cat_id=cat_id,
                                                                 num_of_image=num_of_image)

instance_images = np.array(instance_images)
instance_masks = np.array(instance_masks)

model = tf.keras.models.load_model(h5_path)


def mask_to_label(mask):
    print(mask)
    binary_mask = (mask >= threshold).astype(int)
    return binary_mask


for i in range(0, len(instance_images), batch_size):
    batch_images = instance_images[i:i + batch_size]
    batch_masks = instance_masks[i:i + batch_size]

    predictions = model.predict(batch_images)

    predicted_labels = [mask_to_label(prediction) for prediction in predictions]

    for j in range(len(batch_images)):
        plt.figure(figsize=(10, 5))

        # 원본 이미지 표시
        plt.subplot(1, 2, 1)
        plt.imshow(batch_images[j])
        plt.title('Original Image')
        plt.axis('off')

        # 원본 이미지 위에 마스크를 겹침
        plt.subplot(1, 2, 2)
        plt.imshow(batch_images[j])
        plt.imshow(predicted_labels[j], cmap='viridis', alpha=0.5)  # alpha 값으로 투명도 조절
        plt.title('Original Image with Mask')
        plt.axis('off')

        plt.show()
