import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")  # CUDA의 경로를 이곳에 입력

import json
import random
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from pycocotools.coco import COCO


def save_as_unicode(file_name):
    with open(file_name, 'r', encoding='utf-8') as f:  # 인코딩 문제 해결을 위해 UTF-8로 다시 저장
        data = json.load(f)
    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def load_instances_coco_format_all(image_dir, json_name, input_size,
                                   cat_id, num_of_image=None, shuffle=True, margin=200):
    save_as_unicode(json_name)
    coco = COCO(json_name)
    img_ids = coco.getImgIds()
    if num_of_image is not None:
        img_ids = img_ids[:num_of_image]
    if shuffle:
        random.shuffle(img_ids)
    cat_ids = coco.getCatIds()
    instance_images = []
    instance_masks = []
    for img_id in tqdm(img_ids, 'Loading instances from COCO format train_dataset...'):
        img = coco.loadImgs(img_id)[0]
        image = cv2.cvtColor(cv2.imread(image_dir + img['file_name']), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0  # [0.0, 1.0] 범위로 정규화
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if cat_id is not None and (cat_ids.index(ann['category_id']) + 1) != cat_id:
                continue
            x1, y1, w, h = map(int, ann['bbox'])
            x2, y2 = x1 + w, y1 + h
            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, image.shape[1])
            y2 = min(y2 + margin, image.shape[0])
            instance_images.append(cv2.resize(image[y1:y2, x1:x2], input_size))
            instance_masks.append(cv2.resize(coco.annToMask(ann)[y1:y2, x1:x2], input_size))
    return instance_images, instance_masks


def load_instances_coco_format(image_dir, json_name, input_size, train_batch_size, valid_batch_size, train_rate,
                               cat_id, num_of_image=None, shuffle=True, margin=200):
    save_as_unicode(json_name)
    coco = COCO(json_name)
    img_ids = coco.getImgIds()
    if num_of_image is not None:
        img_ids = img_ids[:num_of_image]
    if shuffle:
        random.shuffle(img_ids)
    cat_ids = coco.getCatIds()
    instance_images = []
    instance_masks = []
    for img_id in tqdm(img_ids, 'Loading instances from COCO format train_dataset...'):
        img = coco.loadImgs(img_id)[0]
        image = cv2.cvtColor(cv2.imread(image_dir + img['file_name']), cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0  # [0.0, 1.0] 범위로 정규화
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for ann in anns:
            if cat_id is not None and (cat_ids.index(ann['category_id']) + 1) != cat_id:
                continue
            x1, y1, w, h = map(int, ann['bbox'])
            x2, y2 = x1 + w, y1 + h
            x1 = max(x1 - margin, 0)
            y1 = max(y1 - margin, 0)
            x2 = min(x2 + margin, image.shape[1])
            y2 = min(y2 + margin, image.shape[0])
            instance_images.append(cv2.resize(image[y1:y2, x1:x2], input_size))
            instance_masks.append(cv2.resize(coco.annToMask(ann)[y1:y2, x1:x2], input_size))

    instance_images = np.array(instance_images)
    instance_masks = np.array(instance_masks)

    total_image_count = len(instance_images)

    train_images = instance_images[:int(total_image_count * train_rate)]
    train_masks = instance_masks[:int(total_image_count * train_rate)]

    valid_images = instance_images[int(total_image_count * train_rate):]
    valid_masks = instance_masks[int(total_image_count * train_rate):]

    train_set = tf.data.Dataset.from_tensor_slices((train_images, train_masks))
    valid_set = tf.data.Dataset.from_tensor_slices((valid_images, valid_masks))

    train_batches = (  # 훈련용 데이터 전처리는 이곳에서 수행합니다.
        train_set
        .cache()
        .shuffle(len(train_images) * 2)
        .batch(train_batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    valid_batches = (  # 검증용 데이터 전처리는 이곳에서 수행합니다.
        valid_set
        .cache()
        .shuffle(len(valid_images) * 2)
        .batch(valid_batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_batches, valid_batches
