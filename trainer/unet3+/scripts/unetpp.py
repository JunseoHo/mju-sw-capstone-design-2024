from keras import Input, Model
from keras.src.layers import Conv2DTranspose, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate
from tensorflow.keras.callbacks import *


def conv_block(inputs, filters, dropout_rate=0):
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(inputs)
    x = Dropout(dropout_rate)(x) if dropout_rate > 0 else x
    x = Conv2D(filters, (3, 3), padding='same', activation='relu')(x)
    return x


def unetpp(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1_1 = conv_block(inputs, 64)
    pool1 = MaxPooling2D((2, 2))(conv1_1)

    conv2_1 = conv_block(pool1, 128)
    pool2 = MaxPooling2D((2, 2))(conv2_1)

    conv3_1 = conv_block(pool2, 256)
    pool3 = MaxPooling2D((2, 2))(conv3_1)

    conv4_1 = conv_block(pool3, 512)
    pool4 = MaxPooling2D((2, 2))(conv4_1)

    # Center
    center = conv_block(pool4, 1024)

    # Decoder with nested skip connections
    conv4_2 = conv_block(concatenate([Conv2DTranspose(512, (2, 2), strides=2, padding='same')(center), conv4_1]), 512)
    conv3_2 = conv_block(concatenate([Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv4_2), conv3_1]), 256)
    conv2_2 = conv_block(concatenate([Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv3_2), conv2_1]), 128)
    conv1_2 = conv_block(concatenate([Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv2_2), conv1_1]), 64)

    conv3_3 = conv_block(
        concatenate([Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv4_2), conv3_1, conv3_2]), 256)
    conv2_3 = conv_block(
        concatenate([Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv3_3), conv2_1, conv2_2]), 128)
    conv1_3 = conv_block(
        concatenate([Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv2_3), conv1_1, conv1_2]), 64)

    conv2_4 = conv_block(
        concatenate([Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv3_3), conv2_1, conv2_2, conv2_3]), 128)
    conv1_4 = conv_block(
        concatenate([Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv2_4), conv1_1, conv1_2, conv1_3]), 64)

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv1_4)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
