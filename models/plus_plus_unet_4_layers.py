from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Model

from models.building_blocks.conv_block import conv_block
from models.building_blocks.conv_block_transpose import conv_block_transpose

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
