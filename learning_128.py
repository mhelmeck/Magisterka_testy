import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow

from callbacks import get_callbacks
from consts import MODEL_SAVE_DIR
from models.unet_128 import build_model
from models.unet_128 import build_model_plus_plus
from read_data import get_images_and_masks

img_width = 128
img_height = 128
epochs = 200

get_callbacks("unet_128", 1)

model = build_model_plus_plus()
# model = build_model()
# model.summary()
# print("Model built")

# x_train, y_train = get_images_and_masks(img_width, img_height, 100, 180, True)
# results = model.fit(x_train, y_train, validation_split=0.1, batch_size=128, epochs=epochs, callbacks=get_callbacks("unet_128", 1))
# print("Learning ended")

# test_images, test_labels = get_images_and_masks(img_width, img_height, 180, 200, True)
# print("Test data loaded")
#
model.load_weights(MODEL_SAVE_DIR.format('unet_128', 1) + "cp.ckpt")
# loss, acc = model.evaluate(test_images, test_labels, verbose=1)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
#
# preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
# preds_test = model.predict(test_images, verbose=1)
#
# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
#
# Perform a sanity check on some random training samples
# ix = 33
# imshow(test_images[ix])
# plt.show()
# imshow(np.squeeze(test_labels[ix]))
# plt.show()
# imshow(np.squeeze(preds_test[ix]))
# plt.show()

test_images, test_labels = get_images_and_masks(img_width, img_height, 181, 181, True)
print("Test data loaded")

# loss, acc = model.evaluate(test_images, test_labels, verbose=1)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(MODEL_SAVE_DIR.format('unet_128', 1) + "cp.ckpt")
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