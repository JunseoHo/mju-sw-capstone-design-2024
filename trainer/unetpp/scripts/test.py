import tensorflow as tf
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import *
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt  # 데이터 시각화 라이브러리
from trainer.unetpp.scripts.unetpp import unetpp

# Constants
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
checkpoint_path = "unetpp_ckpt.weights.h5"
epochs = 3

# Load data from tensorflow datasets
dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

input_shape = (128, 128, 3)
num_classes = 3
model = unetpp(input_shape, num_classes)
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask

batch_size = 64
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

dataset = test_images.batch(batch_size)

model.load_weights('unetpp_ckpt.weights.h5')

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()

def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

if model is None:
    print("Model is not provided.")
if dataset:
    for image, mask in dataset.take(3):
        pred_mask = model.predict(image)
        display([image[0], mask[0], create_mask(pred_mask)])
else:
    print('No dataset provided')

