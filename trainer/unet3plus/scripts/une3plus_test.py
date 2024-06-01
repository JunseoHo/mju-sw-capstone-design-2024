import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")  # CUDA의 경로를 이곳에 입력

from common.data_loader import load_instances_coco_format_all
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

images_dir = '../test_dataset/images/'  # 전체 이미지가 저장되어 있는 디렉토리의 상대 주소
json_path = '../test_dataset/annotations/instances_default.json'  # COCO Json 파일의 상대 주소
h5_path = "../checkpoints/epoch-45-val_loss-029.h5"  # 학습된 모델의 저장 위치
input_size = (256, 256)  # 모델의 입력 크기
num_of_image = 100
cat_id = 4
threshold = 0.8
batch_size = 3

instance_images, instance_masks = load_instances_coco_format_all(images_dir, json_path, input_size, cat_id=cat_id,
                                                                 num_of_image=num_of_image)

instance_images = np.array(instance_images)
instance_masks = np.array(instance_masks)

model = tf.keras.models.load_model(h5_path)


def mask_to_label(mask):
    binary_mask = (mask >= threshold).astype(int)
    return binary_mask


def calculate_pixel_accuracy(truth, infer):
    total_pixels = truth.size
    correct_pixels = 0
    for i in range(len(truth)):
        for j in range(len(truth)):
            if truth[i][j] == infer[i][j]:
                correct_pixels += 1
    accuracy = correct_pixels / total_pixels
    return accuracy


for i in range(0, len(instance_images), batch_size):
    batch_images = instance_images[i:i + batch_size]
    batch_masks = instance_masks[i:i + batch_size]

    predictions = model.predict(batch_images)

    predicted_labels = [mask_to_label(prediction) for prediction in predictions]

    for i in range(0, len(instance_images), batch_size):
        batch_images = instance_images[i:i + batch_size]
        batch_masks = instance_masks[i:i + batch_size]

        predictions = model.predict(batch_images)

        predicted_labels = [mask_to_label(prediction) for prediction in predictions]

        for j in range(len(batch_images)):
            accuracy = calculate_pixel_accuracy(batch_masks[j], predicted_labels[j])

            plt.figure(figsize=(5, 5))

            # 원본 이미지 위에 마스크를 겹침
            plt.imshow(batch_images[j])
            plt.imshow(predicted_labels[j], cmap='viridis', alpha=0.5)  # alpha 값으로 투명도 조절
            plt.title(f'Inference result\nPixel Accuracy: {accuracy:.2f}')
            plt.axis('off')

            plt.show()
