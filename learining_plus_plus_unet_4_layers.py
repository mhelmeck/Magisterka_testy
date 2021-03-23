import numpy as np
from sklearn.metrics import f1_score

from callbacks import get_callbacks
from models.plus_plus_unet_4_layers import build_model
from read_data import get_images_and_masks
from utils import get_save_model_path

img_width = 128
img_height = 128
channel_numbers = 3
epochs = 200

print('Started')
model_save_path = get_save_model_path('plus_plus_unet_4_layers')

model = build_model(img_width, img_height, channel_numbers)
model.summary()
print("Model built")

# x_train, y_train = get_images_and_masks(img_width, img_height, 0, 180, True)
# print("Train data loaded")

# results = model.fit(x_train, y_train, validation_split=0.2, batch_size=32, epochs=epochs, callbacks=get_callbacks(model_save_path))
# print("Learning ended")

# test_images, test_labels = get_images_and_masks(img_width, img_height, 181, 209, True)
# print("Test data loaded")

# loss, acc = model.evaluate(test_images, test_labels, verbose=1)
# print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

# model.load_weights(model_save_path)
# print("Model weights loaded")

# print("Prediction started")
# preds_test = model.predict(test_images, verbose=1)
# preds_test_t = (preds_test > 0.6).astype(np.uint8)

# f1_score = f1_score(test_labels.flatten().flatten(), preds_test_t.flatten().flatten())
# print('F1 score: %f' % f1_score)
