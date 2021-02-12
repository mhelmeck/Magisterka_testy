import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow

from callbacks import get_callbacks
from models.unet_4_layers import build_model
from read_data import get_images_and_masks
from utils import get_save_model_path

img_width = 128
img_height = 128
channel_numbers=3
epochs = 200

model_save_path = get_save_model_path('unet_4_layers')
# model_save_path = 'training_unet_128_11022021_0844/cp.ckpt'

model = build_model(img_width, img_height, channel_numbers)
print("Model built")

x_train, y_train = get_images_and_masks(img_width, img_height, 100, 180, True)
print("Train data loaded")

results = model.fit(x_train, y_train, validation_split=0.1, batch_size=128, epochs=epochs, callbacks=get_callbacks(model_save_path))
print("Learning ended")

test_images, test_labels = get_images_and_masks(img_width, img_height, 181, 181, True)
print("Test data loaded")


loss, acc = model.evaluate(test_images, test_labels, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(model_save_path)
print("Model weights loaded")

preds_test = model.predict(test_images, verbose=1)
preds_test_t = (preds_test > 0.3).astype(np.uint8)

#Perform a sanity check on some random training samples
ix = 32
imshow(test_images[ix])
plt.show()
imshow(np.squeeze(test_labels[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()
imshow(np.squeeze(preds_test[ix]))
plt.show()
