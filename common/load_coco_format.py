import json
import os

from tqdm import tqdm

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")  # CUDA의 경로를 이곳에 입력
import cv2
from pycocotools.coco import COCO
import numpy as np
import tensorflow as tf

buffer_size = 2048


def load_coco_format(image_dir, json_name, input_size, train_batch_size, valid_batch_size, train_rate):
    if train_rate < 0.1 or train_rate > 1.0:
        raise Exception("Train rate must be between 0.1 and 1.0")
    with open(json_name, 'r', encoding='utf-8') as f:  # 인코딩 문제 해결을 위해 UTF-8로 다시 저장
        data = json.load(f)
    with open(json_name, 'w', encoding='utf-8') as f:
        json.dump(data, f)
    coco = COCO(json_name)
    img_ids = coco.getImgIds()[:50]
    cat_ids = coco.getCatIds()
    images = []
    segmentation_masks = []
    for img_id in tqdm(img_ids, 'Loading COCO format dataset...'):
        img = coco.loadImgs(img_id)[0]
        file_name = image_dir + img['file_name']
        image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        image = cv2.resize(image, input_size)
        images.append(image)
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        segmentation_mask = np.zeros(input_size)
        for ann in anns:
            if 'segmentation' in ann:
                pixel_value = cat_ids.index(ann['category_id']) + 1
                mask = cv2.resize(coco.annToMask(ann) * pixel_value, input_size)
                segmentation_mask = np.maximum(segmentation_mask, mask).astype(np.int32)
        segmentation_mask = segmentation_mask.reshape(input_size[0], input_size[1], 1)
        segmentation_masks.append(segmentation_mask)

    images = np.array(images)
    segmentation_masks = np.array(segmentation_masks)

    np.random.shuffle(images)
    np.random.shuffle(segmentation_masks)

    total_image_count = len(images)

    train_images = images[:int(total_image_count * train_rate)]
    train_masks = segmentation_masks[:int(total_image_count * train_rate)]

    valid_images = images[int(total_image_count * train_rate):]
    valid_masks = segmentation_masks[int(total_image_count * train_rate):]

    train_set = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    valid_set = tf.data.Dataset.from_tensor_slices((valid_images, valid_masks))

    train_batches = (  # 훈련용 데이터 전처리는 이곳에서 수행합니다.
        train_set
        .cache()
        .shuffle(buffer_size)
        .batch(train_batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    valid_batches = (  # 검증용 데이터 전처리는 이곳에서 수행합니다.
        valid_set
        .cache()
        .shuffle(buffer_size)
        .batch(valid_batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_batches, valid_batches
