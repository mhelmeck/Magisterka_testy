import os
import tensorflow as tf

from consts import LOGS_DIR


def get_callbacks(model_save_path):
    checkpoint_path = model_save_path

    checkpoint_dir = os.path.dirname(checkpoint_path)

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                     validation_split=0.1,
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    return [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
        tf.keras.callbacks.TensorBoard(log_dir=LOGS_DIR),
        cp_callback
    ]
