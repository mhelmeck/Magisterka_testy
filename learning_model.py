import numpy as np
from sklearn.metrics import f1_score

from models_builders import get_model_builder
from utils.callbacks import get_callbacks
from utils.read_data import get_images_and_masks
from utils.utils import get_save_model_path, load_variables

print("[LOG] Starting application")

print('[LOG] Loading variables')
(
    model_name,
    channel_numbers,
    img_size,
    epochs,
    batch_size,
    starts_neuron,
    start_case_index_train,
    end_case_index_train,
    start_case_index_test,
    end_case_index_test
) = load_variables()
print('[LOG] Did load variables')

print('[LOG] Creating path for model: ', model_name)
model_save_path = get_save_model_path(model_name)

print('[LOG] Building model')
model = get_model_builder(model_name)(
    img_size,
    img_size,
    channel_numbers,
    starts_neuron
)

print('[LOG] Model summary')
model.summary()

print('[LOG] Loading training data for cases from: ', start_case_index_train, " to: ", end_case_index_train)
x_train, y_train = get_images_and_masks(
    img_size,
    img_size,
    start_case_index_train,
    end_case_index_train,
    True
)
print("[LOG] Did load training data")

print('[LOG] Learning on training data with batch_size: ', batch_size, ", epochs: ", epochs)
results = model.fit(
    x_train,
    y_train,
    validation_split=0.2,
    batch_size=batch_size,
    epochs=epochs,
    callbacks=get_callbacks(model_save_path))
print("[LOG] Did learn with success")

print('[LOG] Loading test data for cases from: ', start_case_index_test, " to: ", end_case_index_test)
test_images, test_labels = get_images_and_masks(
    img_size,
    img_size,
    start_case_index_test,
    end_case_index_test,
    True
)
print("[LOG] Did load test data")

print("[LOG] Start model evaluation with test data")
loss, acc = model.evaluate(
    test_images,
    test_labels,
    verbose=1
)
print("[LOG] Did evaluate model")
print("[LOG] Current model accuracy: {:5.2f}%".format(100 * acc))

# model.load_weights(model_save_path)
# print("Model weights loaded")
#
# print("Prediction started")
# preds_test = model.predict(test_images, verbose=1)
# preds_test_t = (preds_test > 0.6).astype(np.uint8)
#
# f1_score = f1_score(test_labels.flatten().flatten(), preds_test_t.flatten().flatten())
# print('F1 score: %f' % f1_score)
