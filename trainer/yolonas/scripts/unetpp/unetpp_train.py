import fiftyone
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D
import tensorflow as tf
from tensorflow.keras.callbacks import *


def conv_block(inputs, filters, kernel_size=(3, 3), activation='relu', padding='same'):
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(inputs)
    conv = Conv2D(filters, kernel_size, activation=activation, padding=padding)(conv)
    return conv

def encoder_block(inputs, filters, pool_size=(2, 2), dropout=0.1):
    conv = conv_block(inputs, filters)
    pool = MaxPooling2D(pool_size)(conv)
    if dropout > 0:
        pool = tf.keras.layers.Dropout(dropout)(pool)
    return conv, pool

def decoder_block(inputs, conv_output, filters, upsample_size=(2, 2), dropout=0.1):
    upsample = UpSampling2D(upsample_size)(inputs)
    concat = concatenate([conv_output, upsample], axis=3)
    conv = conv_block(concat, filters)
    if dropout > 0:
        conv = tf.keras.layers.Dropout(dropout)(conv)
    return conv

def build_unetpp(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Encoder
    conv1, pool1 = encoder_block(inputs, 64)
    conv2, pool2 = encoder_block(pool1, 128)
    conv3, pool3 = encoder_block(pool2, 256)
    conv4, pool4 = encoder_block(pool3, 512)

    # Center
    center = conv_block(pool4, 1024)

    # Decoder
    deconv4 = decoder_block(center, conv4, 512)
    deconv3 = decoder_block(deconv4, conv3, 256)
    deconv2 = decoder_block(deconv3, conv2, 128)
    deconv1 = decoder_block(deconv2, conv1, 64)

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(deconv1)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Example usage
input_shape = (256, 256, 3)
num_classes = 2  # For binary segmentation
model = build_unetpp(input_shape, num_classes)
model.summary()


# Define your UNet++ model
input_shape = (256, 256, 3)
num_classes = 2
model = build_unetpp(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define checkpoint callback to save model weights during training
checkpoint_path = "model_checkpoint.weights.h5"

# import os
#
# # 이미지 파일과 마스크 파일이 있는 폴더 경로
# image_dir = "/path/to/image"
# mask_dir = "/path/to/mask"
#
# # 이미지 파일 및 마스크 파일 경로 목록 가져오기
# image_filenames = [os.path.join(image_dir, filename) for filename in os.listdir(image_dir)]
# mask_filenames = [os.path.join(mask_dir, filename) for filename in os.listdir(mask_dir)]
#
# # 이미지 파일과 마스크 파일 경로 목록 정렬
# image_filenames.sort()
# mask_filenames.sort()
#
# # 데이터셋 생성
# def parse_image(filename):
#     image = tf.io.read_file(filename)
#     image = tf.image.decode_image(image, channels=3)  # 이미지는 RGB 채널
#     image = tf.cast(image, tf.float32) / 255.0       # 이미지 정규화
#     return image
#
# def parse_mask(filename):
#     mask = tf.io.read_file(filename)
#     mask = tf.image.decode_image(mask, channels=1)   # 마스크는 단일 채널
#     mask = tf.cast(mask, tf.int32)                   # 마스크를 정수형으로 변환
#     return mask
#
# image_dataset = tf.data.Dataset.from_tensor_slices(image_filenames)
# image_dataset = image_dataset.map(parse_image)
#
# mask_dataset = tf.data.Dataset.from_tensor_slices(mask_filenames)
# mask_dataset = mask_dataset.map(parse_mask)
#
# # 이미지와 마스크를 합치기
# dataset = tf.data.Dataset.zip((image_dataset, mask_dataset))
#
# # 데이터셋 전처리 (크기 조정, 배치 등)
# dataset = dataset.shuffle(buffer_size=1000).batch(60).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

dataset = fiftyone.zoo.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    classes=["person", "car"],
    max_samples=50,
)

import tensorflow_datasets as tfds

dataset, info = tfds.load('coco/2017', split='validation[:1]', with_info=True)

print(dataset)

# 모델 훈련
model.fit(dataset, epochs=10)
checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True)

# Train the model

# Export the model with loaded weights
model.load_weights(checkpoint_path)
export_path = "/path/to/exported/model"
model.save(export_path)
