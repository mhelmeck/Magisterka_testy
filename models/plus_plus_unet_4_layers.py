from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

def conv_block(inputs, filters, pool=True):
  x = Conv2D(filters, (3, 3), padding="same")(inputs) #kernel_initializer='he_normal'
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dropout(0.2)(x)

  x = Conv2D(filters, (3, 3), padding="same")(x) #kernel_initializer='he_normal'
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dropout(0.2)(x)

  if pool == True:
      p = MaxPooling2D((2, 2))(x)
      return x, p
  else:
      return x

def conv_block_transpose(inputs, filters, concatenation_list):
  x = Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(inputs)
  x = concatenate(concatenation_list + [x])
  
  x = Conv2D(filters, (3, 3), padding="same")(x) #kernel_initializer='he_normal'
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  
  x = Conv2D(filters, (3, 3), padding="same")(x) #kernel_initializer='he_normal'
  x = BatchNormalization()(x)
  x = Activation("relu")(x)
  x = Dropout(0.2)(x)
  
  return x

def build_model(img_width, img_height, channels_number, start_neurons=16):
    inputs = Input((img_width, img_height, channels_number))

    x00, p0 = conv_block(inputs, 32, pool=True)
    # x00 = BatchNormalization()(inputs)
    # x00 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x00)
    # x00 = Dropout(0.2)(x00)
    # x00 = BatchNormalization()(x00)
    # x00 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x00)
    # x00 = Dropout(0.2)(x00)
    # p0 = MaxPooling2D((2, 2))(x00)

    x10, p1 = conv_block(p1, 64, pool=True)
    # x10 = BatchNormalization()(p0)
    # x10 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x10)
    # x10 = Dropout(0.2)(x10)
    # x10 = BatchNormalization()(x10)
    # x10 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x10)
    # x10 = Dropout(0.2)(x10)
    # p1 = MaxPooling2D((2, 2))(x10)

    x01 = Conv2DTranspose(start_neurons * 1, kernel_size=(2, 2), strides=(2, 2), padding='same')(x10)
    x01 = concatenate([x00, x01])
    x01 = BatchNormalization()(x01)
    x01 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x01)
    x01 = BatchNormalization()(x01)
    x01 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x01)
    x01 = Dropout(0.2)(x01)

    x20 = BatchNormalization()(p1)
    x20 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x20)
    x20 = Dropout(0.2)(x20)
    x20 = BatchNormalization()(x20)
    x20 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x20)
    x20 = Dropout(0.2)(x20)
    p2 = MaxPooling2D((2, 2))(x20)

    x11 = Conv2DTranspose(start_neurons * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(x20)
    x11 = concatenate([x10, x11])
    x11 = BatchNormalization()(x11)
    x11 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x11)
    x11 = BatchNormalization()(x11)
    x11 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x11)
    x11 = Dropout(0.2)(x11)

    x02 = Conv2DTranspose(start_neurons * 1, kernel_size=(2, 2), strides=(2, 2), padding='same')(x11)
    x02 = concatenate([x00, x01, x02])
    x02 = BatchNormalization()(x02)
    x02 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x02)
    x02 = BatchNormalization()(x02)
    x02 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x02)
    x02 = Dropout(0.2)(x02)

    x30 = BatchNormalization()(p2)
    x30 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x30)
    x30 = Dropout(0.2)(x30)
    x30 = BatchNormalization()(x30)
    x30 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x30)
    x30 = Dropout(0.2)(x30)
    p3 = MaxPooling2D((2, 2))(x30)

    x21 = Conv2DTranspose(start_neurons * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')(x30)
    x21 = concatenate([x20, x21])
    x21 = BatchNormalization()(x21)
    x21 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x21)
    x21 = BatchNormalization()(x21)
    x21 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x21)
    x21 = Dropout(0.2)(x21)

    x12 = Conv2DTranspose(start_neurons * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(x21)
    x12 = concatenate([x10, x11, x12])
    x12 = BatchNormalization()(x12)
    x12 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x12)
    x12 = BatchNormalization()(x12)
    x12 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x12)
    x12 = Dropout(0.2)(x12)

    x03 = Conv2DTranspose(start_neurons * 1, kernel_size=(2, 2), strides=(2, 2), padding='same')(x12)
    x03 = concatenate([x00, x01, x02, x03])
    x03 = BatchNormalization()(x03)
    x03 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x03)
    x03 = BatchNormalization()(x03)
    x03 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x03)
    x03 = Dropout(0.2)(x03)

    x40 = BatchNormalization()(p3)
    x40 = Conv2D(start_neurons * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x40)
    x40 = Dropout(0.2)(x40)
    x40 = BatchNormalization()(x40)
    x40 = Conv2D(start_neurons * 16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x40)
    x40 = Dropout(0.2)(x40)

    x31 = Conv2DTranspose(start_neurons * 8, kernel_size=(2, 2), strides=(2, 2), padding='same')(x40)
    x31 = concatenate([x30, x31])
    x31 = BatchNormalization()(x31)
    x31 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x31)
    x31 = BatchNormalization()(x31)
    x31 = Conv2D(start_neurons * 8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x31)
    x31 = Dropout(0.2)(x31)

    x22 = Conv2DTranspose(start_neurons * 4, kernel_size=(2, 2), strides=(2, 2), padding='same')(x31)
    x22 = concatenate([x20, x21, x22])
    x22 = BatchNormalization()(x22)
    x22 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x22)
    x22 = BatchNormalization()(x22)
    x22 = Conv2D(start_neurons * 4, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x22)
    x22 = Dropout(0.2)(x22)

    x13 = Conv2DTranspose(start_neurons * 2, kernel_size=(2, 2), strides=(2, 2), padding='same')(x22)
    x13 = concatenate([x10, x11, x12, x13])
    x13 = BatchNormalization()(x13)
    x13 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x13)
    x13 = BatchNormalization()(x13)
    x13 = Conv2D(start_neurons * 2, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x13)
    x13 = Dropout(0.2)(x13)

    x04 = Conv2DTranspose(start_neurons * 1, kernel_size=(2, 2), strides=(2, 2), padding='same')(x13)
    x04 = concatenate([x00, x01, x02, x03, x04])
    x04 = BatchNormalization()(x04)
    x04 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x04)
    x04 = BatchNormalization()(x04)
    x04 = Conv2D(start_neurons * 1, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x04)
    x04 = Dropout(0.2)(x04)

    outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
