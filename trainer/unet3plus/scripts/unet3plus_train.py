import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")  # CUDA의 경로를 이곳에 입력
from common.coco_loader import load_coco_format
from tensorflow.python import keras
from keras.callbacks import ModelCheckpoint

from unet3plus import *

images_dir = '../dataset/images/'
json_path = '../dataset/COCO_Football Pixel.json'
checkpoint_path = "../checkpoints/unet3plus_ckpt.weights.h5"
export_path = "../checkpoints/unet3plus_export.weights.h5"
input_size = (128, 128)
epoch = 25
batch_size = 2
num_classes = 12  # 배경까지 포함한 개수여야 합니다.

keras.backend.clear_session()

train_batches = load_coco_format(images_dir, json_path, input_size, batch_size)
model = unet3plus((input_size[0], input_size[1], 3), num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_batches, epochs=epoch,
          callbacks=ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=False))
model.save(export_path)
