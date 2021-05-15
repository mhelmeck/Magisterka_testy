from tensorflow.python.keras.layers import *

from models.building_blocks.conv_2d import conv_2d


def conv_block_transpose(inputs, features, concatenation_list):
    x = Conv2DTranspose(features, (2, 2), strides=(2, 2), padding='same')(inputs)
    x = concatenate(concatenation_list + [x])

    x = conv_2d(x, features, (3, 3))
    x = conv_2d(x, features, (3, 3))

    x = Dropout(0.2)(x)

    return x
