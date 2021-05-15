from tensorflow.python.keras.layers import *


def conv_block_transpose(inputs, filters, concatenation_list):
    x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = concatenate(concatenation_list + [x])

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    return x
