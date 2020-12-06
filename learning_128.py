import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow

from extracted_data.callbacks import get_callbacks
from extracted_data.consts import MODEL_SAVE_DIR
from extracted_data.models.unet_128 import build_model
from extracted_data.read_data import get_images_and_masks

img_width = 128
img_height = 128
epochs = 200

get_callbacks("unet_128", 1)

model = build_model()
print("Model built")

# x_train, y_train = get_images_and_masks(img_width, img_height, 0, 80, True)
# print("Train data loaded")

# results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=epochs, callbacks=get_callbacks("unet_128", 1))
# print("Learning ended")

test_images, test_labels = get_images_and_masks(128, 128, 81, 81, True)
print("Test data loaded")

model.load_weights(MODEL_SAVE_DIR.format('unet_128', 1) + "cp.ckpt")
loss, acc = model.evaluate(test_images, test_labels, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# preds_train = model.predict(X_train[:int(X_train.shape[0] * 0.9)], verbose=1)
# preds_val = model.predict(X_train[int(X_train.shape[0] * 0.9):], verbose=1)
preds_test = model.predict(test_images, verbose=1)

# preds_train_t = (preds_train > 0.5).astype(np.uint8)
# preds_val_t = (preds_val > 0.5).astype(np.uint8)
preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = 51
imshow(test_images[ix])
plt.show()
imshow(np.squeeze(test_labels[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()
