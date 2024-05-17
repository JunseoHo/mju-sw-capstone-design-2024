import cv2
from pycocotools import mask as maskUtils
import tensorflow as tf
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
import numpy as np
from unet3plus import *

coco_images_dir = '../dataset/images/'
coco_json_path = '../dataset/COCO_Football Pixel.json'  # COCO 형식의 Json 파일 경로
input_size = [512, 512]

# Json 파일 로드
coco = COCO(coco_json_path)
img_ids = coco.getImgIds()
cat_ids = coco.getCatIds()
images = []
segmentation_masks = []
for img_id in img_ids:
    img = coco.loadImgs(img_id)[0]
    file_name = coco_images_dir + img['file_name']
    image = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    images.append(tf.image.resize(image, input_size))
    annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    segmentation_mask = np.zeros((input_size[0], input_size[1], 1))
    for ann in anns:
        if 'segmentation' in ann:
            rle = coco.annToRLE(ann)
            mask = maskUtils.decode(rle)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.convert_to_tensor(mask)
            mask = tf.image.convert_image_dtype(mask, tf.uint8)
            mask = tf.image.resize(mask, input_size)
            mask *= (cat_ids.index(ann['category_id']) + 1)
            segmentation_mask += mask
    segmentation_masks.append(segmentation_mask)

images = np.array(images).reshape((len(img_ids), input_size[0], input_size[1], 3))
segmentation_masks = np.array(segmentation_masks).reshape((len(img_ids), input_size[0], input_size[1], 1))


# def visualize_image_and_mask(image, mask):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#     # 이미지 출력
#     axes[0].imshow(image)
#     axes[0].set_title('Image')
#     axes[0].axis('off')
#
#     # 마스크 출력
#     axes[1].imshow(mask, cmap='gray')  # 마스크는 흑백으로 출력
#     axes[1].set_title('Segmentation Mask')
#     axes[1].axis('off')
#
#     plt.show()
#
#
# # 이미지와 세그멘테이션 마스크를 시각화
# visualize_image_and_mask(images[0], segmentation_masks[0])

model = unet3plus((input_size[0], input_size[1], 3), num_classes=len(coco.getCatIds()))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(images, segmentation_masks, epochs=3, batch_size=30)


