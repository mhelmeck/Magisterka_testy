import numpy as np
from sklearn.metrics import f1_score

from models.plus_plus_unet_4_layers import build_model_plus
from utils.callbacks import get_callbacks
from utils.read_data import get_images_and_masks
from utils.utils import get_save_model_path, load_variables

print('Started')
(model_name, channel_numbers, img_size, epochs, batch_size, starts_neuron, start_case_index_train, end_case_index_train,
 start_case_index_test, end_case_index_test) = load_variables()
print('Variables loaded')

model_save_path = get_save_model_path(model_name)

model = build_model_plus(img_size, img_size, channel_numbers, starts_neuron)
print("Model built")

model.summary()
print("Model built")

model.summary()
print("Model summary")

x_train, y_train = get_images_and_masks(img_size, img_size, start_case_index_train, end_case_index_train, True)
print("Train data loaded")

results = model.fit(x_train, y_train, validation_split=0.2, batch_size=batch_size, epochs=epochs,
                    callbacks=get_callbacks(model_save_path))
print("Learning ended")

test_images, test_labels = get_images_and_masks(img_size, img_size, start_case_index_test, end_case_index_test, True)
print("Test data loaded")

loss, acc = model.evaluate(test_images, test_labels, verbose=1)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

model.load_weights(model_save_path)
print("Model weights loaded")

print("Prediction started")
preds_test = model.predict(test_images, verbose=1)
preds_test_t = (preds_test > 0.6).astype(np.uint8)

f1_score = f1_score(test_labels.flatten().flatten(), preds_test_t.flatten().flatten())
print('F1 score: %f' % f1_score)