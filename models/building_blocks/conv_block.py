from tensorflow.python.keras.layers import *


def conv_block(inputs, filters, pool=True):
    x = Conv2D(filters, (3, 3), padding="same")(inputs)  # kernel_initializer='he_normal'
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)  # kernel_initializer='he_normal'
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    if pool:
        p = MaxPooling2D((2, 2))(x)
        return x, p
    else:
        return x
