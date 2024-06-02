import os

os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8/bin")  # CUDA의 경로를 이곳에 입력

from common.callback import TrainCallback
from common.data_loader import load_instances_coco_format
from keras.callbacks import ModelCheckpoint
from unet3plus import *

images_dir = '../train_dataset/images/'  # 전체 이미지가 저장되어 있는 디렉토리의 상대 주소
json_path = '../train_dataset/annotations/instances_default.json'  # COCO Json 파일의 상대 주소
checkpoint_path = "../checkpoints/epoch-{epoch:02d}-val_loss-{val_loss:.2f}.h5"  # 체크 포인트의 저장 위치
export_path = "../checkpoints/model.weights.h5"  # 학습된 모델의 저장 위치
input_size = (256, 256)  # 모델의 입력 크기 : OOM 발생 시 줄여주세요.
epoch = 200  # 에폭의 횟수
train_batch_size = 2  # 훈련용 데이터 미니배치당 크기 : OOM 발생 시 줄여주세요.
valid_batch_size = 2  # 검증용 데이터 미니배치당 크기 : OOM 발생 시 줄여주세요.
train_rate = 0.9  # 전체 데이터 세트에서 훈련용으로 사용할 비율.
num_classes = 1  # 이진 분류이므로 출력 채널의 크기는 1입니다.
cat_id = 4  # 학습할 카테고리 아이디
callback_preq = 10  # 학습 중 콜백을 호출하는 에폭 주기

train_batches, valid_batches = load_instances_coco_format(images_dir, json_path, input_size, train_batch_size,
                                                          valid_batch_size, train_rate, cat_id=cat_id)

print(f'Input size : {input_size}')
print(f'Epoch : {epoch}')
print(f'Train batch size : {len(train_batches)}')
print(f'valid batch size : {len(valid_batches)}')

model = unet3plus((input_size[0], input_size[1], 3), num_classes)
model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_batches,
          epochs=epoch,
          shuffle=True,
          steps_per_epoch=len(train_batches),
          validation_data=valid_batches,
          validation_steps=len(valid_batches),
          workers=5,
          callbacks=[ModelCheckpoint(filepath=checkpoint_path, save_best_only=True),
                     TrainCallback(validation_data=valid_batches, freq=callback_preq)])
model.save(export_path)
