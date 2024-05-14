import fiftyone
from keras.src.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, UpSampling2D
import tensorflow as tf
from tensorflow.keras.callbacks import *
import tensorflow_datasets as tfds


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
input_shape = (128, 128, 3)
num_classes = 3  # For binary segmentation
model = build_unetpp(input_shape, num_classes)
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define checkpoint callback to save model weights during training
checkpoint_path = "model_checkpoint.weights.h5"

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def load_image(datapoint):
    input_image = tf.image.resize(datapoint['image'], (128, 128))
    input_mask = tf.image.resize(datapoint['segmentation_mask'], (128, 128))

    input_image, input_mask = normalize(input_image, input_mask)

    return input_image, input_mask


TRAIN_LENGTH = info.splits['train'].num_examples
BATCH_SIZE = 64
BUFFER_SIZE = 1000
STEPS_PER_EPOCH = TRAIN_LENGTH // BATCH_SIZE

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(BATCH_SIZE)

VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // BATCH_SIZE // VAL_SUBSPLITS
print(model)
model.fit(train_batches, epochs=20,
          steps_per_epoch=STEPS_PER_EPOCH,
          validation_steps=VALIDATION_STEPS,
          validation_data=test_batches)
# checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True)

# Train the model

# Export the model with loaded weights
# model.load_weights(checkpoint_path)
# export_path = "/path/to/exported/model"
# model.save(export_path)
