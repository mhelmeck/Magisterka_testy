from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

# ------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------

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

# ------------------------------------------------------------------------------------

def build_model(img_width, img_height, channels_number, start_neurons=16):
  inputs = Input((img_width, img_height, channels_number))

  x00, p0 = conv_block(inputs, 32, pool=True)
  
  x10, p1 = conv_block(p0, 64, pool=True)
  x01 = conv_block_transpose(x10, 32, [x00])

  x20, p2 = conv_block(p1, 128, pool=True)
  x11 = conv_block_transpose(x20, 32, [x10])
  x02 = conv_block_transpose(x11, 32, [x00, x01])
    
  x30, p3 = conv_block(p2, 256, pool=True)
  x21 = conv_block_transpose(x30, 32, [x20])
  x12 = conv_block_transpose(x21, 32, [x10, x11])
  x03 = conv_block_transpose(x12, 32, [x00, x01, x02])

  x40 = conv_block(p3, 512, pool=False)
  x31 = conv_block_transpose(x40, 256, [x30])
  x22 = conv_block_transpose(x31, 128, [x20, x21])
  x13 = conv_block_transpose(x22, 64, [x10, x11, x12])
  x04 = conv_block_transpose(x13, 32, [x00, x01, x02, x03])
 
  outputs = Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)

  model = Model(inputs=[inputs], outputs=[outputs])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model
