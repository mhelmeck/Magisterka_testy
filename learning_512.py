import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imshow

from extracted_data.read_data import get_images_and_masks
from extracted_data.models.unet_512 import build_model
from extracted_data.callbacks import get_callbacks

img_width = 512
img_height = 512
epochs = 200

model = build_model()

x_train, y_train = get_images_and_masks(img_width, img_height, 0, 80)

results = model.fit(x_train, y_train, validation_split=0.1, batch_size=16, epochs=epochs, callbacks=get_callbacks("unet_512", 1))

test_images, test_labels = get_images_and_masks(img_width, img_width, 81, 100)

# Re-evaluate the model
loss, acc = model.evaluate(test_images, test_labels, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

preds_test = model.predict(test_images, verbose=1)

preds_test_t = (preds_test > 0.5).astype(np.uint8)

# Perform a sanity check on some random training samples
ix = 51
imshow(test_images[ix])
plt.show()
imshow(np.squeeze(test_labels[ix]))
plt.show()
imshow(np.squeeze(preds_test_t[ix]))
plt.show()