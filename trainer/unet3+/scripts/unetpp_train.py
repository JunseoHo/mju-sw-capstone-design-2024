from tensorflow.keras.callbacks import *
import tensorflow_datasets as tfds
from unet3plus import *

# Constants
input_shape = (128, 128, 3)
num_classes = 3
optimizer = 'adam'
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy']
checkpoint_path = "unetpp_ckpt.weights.h5"
epochs = 3

dataset, info = tfds.load('coco', with_info=True)

model = unet3plus(input_shape, num_classes)
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


train_length = info.splits['train'].num_examples
batch_size = 64
buffer_size = 1000
steps_for_epoch = train_length // batch_size

train_images = dataset['train'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
test_images = dataset['test'].map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

train_batches = (
    train_images
    .cache()
    .shuffle(buffer_size)
    .batch(batch_size)
    .repeat()
    .prefetch(buffer_size=tf.data.AUTOTUNE))

test_batches = test_images.batch(batch_size)

VAL_SUBSPLITS = 5
VALIDATION_STEPS = info.splits['test'].num_examples // batch_size // VAL_SUBSPLITS
model.fit(train_batches, epochs=epochs,
          steps_per_epoch=steps_for_epoch,
          validation_steps=VALIDATION_STEPS,
          validation_data=test_batches,
          callbacks=ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, save_best_only=True))

# save model weights
model.load_weights(checkpoint_path)
export_path = "export.h5"
model.save(export_path)
