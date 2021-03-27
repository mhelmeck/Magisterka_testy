from models.plus_plus_unet_4_layers import build_model_plus
from utils.utils import load_variables

print('Started')
(channel_numbers, img_size, epochs, batch_size, starts_neuron, start_case_index_train, end_case_index_train,
 start_case_index_test, end_case_index_test) = load_variables()
print('Variables loaded')

# model_save_path = get_save_model_path('plus_plus_unet_4_layers')

model = build_model_plus(img_size, img_size, channel_numbers, starts_neuron)
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
