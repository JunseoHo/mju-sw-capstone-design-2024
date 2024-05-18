import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")  # CUDA의 경로를 이곳에 입력
import cv2
from pycocotools.coco import COCO
import numpy as np
import tensorflow as tf


def load_coco_format(image_dir, json_name, input_size, batch_size):
    # Json 파일 로드
    coco = COCO(json_name)
    img_ids = coco.getImgIds()
    cat_ids = coco.getCatIds()
    images = []
    segmentation_masks = []
    for img_id in img_ids:
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

    dataset = tf.data.Dataset.from_tensor_slices((images, segmentation_masks))
    batches = (
        dataset
        .cache()
        .shuffle(1000)
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE))

    return batches
