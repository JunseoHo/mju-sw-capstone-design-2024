import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.0/bin")  # CUDA의 경로를 이곳에 입력
from common.load_coco_format import load_coco_format
from tensorflow.python import keras
from keras.callbacks import ModelCheckpoint

from unet3plus import *

images_dir = '../data/champions_league/dataset_soccer/images/'  # 전체 이미지가 저장되어 있는 디렉토리의 상대 주소
json_path = '../data/champions_league/dataset_soccer/COCO_Football Pixel.json'  # COCO Json 파일의 상대 주소
checkpoint_path = "../checkpoints/unet3plus_ckpt.weights.h5"  # 체크 포인트의 저장 위치
export_path = "../checkpoints/unet3plus_export.weights.h5"  # 학습된 모델의 저장 위치
input_size = (128, 128)  # 초기128/ 모델의 입력 크기 : OOM 발생 시 줄여주세요.
epoch = 10  # 에폭의 횟수
train_batch_size = 10  # 초기5/ 훈련용 데이터 미니배치당 크기 : OOM 발생 시 줄여주세요.
valid_batch_size = 10  # 초기5/ 검증용 데이터 미니배치당 크기 : OOM 발생 시 줄여주세요.
train_rate = 0.7  # 전체 데이터 세트에서 훈련용으로 사용할 비율.
num_classes = 12  # 배경까지 포함한 개수여야 합니다.

train_batches, valid_batches = load_coco_format(images_dir, json_path, input_size, train_batch_size, valid_batch_size,
                                                train_rate)

print(f'Input size : {input_size}')
print(f'Epoch : {epoch}')

model = unet3plus((input_size[0], input_size[1], 3), num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_batches,
          epochs=epoch,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          callbacks=ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=False))
model.save(export_path)
