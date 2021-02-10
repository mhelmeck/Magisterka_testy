import tensorflow as tf

img_width = 128
img_height = 128
channels_number = 3


def build_model():
    inputs = tf.keras.layers.Input((img_width, img_height, channels_number))

    x00 = tf.keras.layers.BatchNormalization()(inputs)
    x00 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x00)
    x00 = tf.keras.layers.Dropout(0.2)(x00)
    x00 = tf.keras.layers.BatchNormalization()(x00)
    x00 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x00)
    x00 = tf.keras.layers.Dropout(0.2)(x00)
    p0 = tf.keras.layers.MaxPooling2D((2, 2))(x00)

    x10 = tf.keras.layers.BatchNormalization()(p0)
    x10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x10)
    x10 = tf.keras.layers.Dropout(0.2)(x10)
    x10 = tf.keras.layers.BatchNormalization()(x10)
    x10 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x10)
    x10 = tf.keras.layers.Dropout(0.2)(x10)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(x10)

    x20 = tf.keras.layers.BatchNormalization()(p1)
    x20 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x20)
    x20 = tf.keras.layers.Dropout(0.2)(x20)
    x20 = tf.keras.layers.BatchNormalization()(x20)
    x20 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x20)
    x20 = tf.keras.layers.Dropout(0.2)(x20)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(x20)

    x30 = tf.keras.layers.BatchNormalization()(p2)
    x30 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x30)
    x30 = tf.keras.layers.Dropout(0.2)(x30)
    x30 = tf.keras.layers.BatchNormalization()(x30)
    x30 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x30)
    x30 = tf.keras.layers.Dropout(0.2)(x30)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(x30)

    x40 = tf.keras.layers.BatchNormalization()(p3)
    x40 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x40)
    x40 = tf.keras.layers.Dropout(0.2)(x40)
    x40 = tf.keras.layers.BatchNormalization()(x40)
    x40 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x40)
    x40 = tf.keras.layers.Dropout(0.2)(x40)

    x31 = tf.keras.layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding='same')(x40)
    x31 = tf.keras.layers.concatenate([x30, x31])
    x31 = tf.keras.layers.BatchNormalization()(x31)
    x31 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x31)
    x31 = tf.keras.layers.BatchNormalization()(x31)
    x31 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x31)
    x31 = tf.keras.layers.Dropout(0.2)(x31)

    x22 = tf.keras.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding='same')(x31)
    x22 = tf.keras.layers.concatenate([x20, x22])
    x22 = tf.keras.layers.BatchNormalization()(x22)
    x22 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x22)
    x22 = tf.keras.layers.BatchNormalization()(x22)
    x22 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x22)
    x22 = tf.keras.layers.Dropout(0.2)(x22)

    x13 = tf.keras.layers.Conv2DTranspose(32, kernel_size=(2, 2), strides=(2, 2), padding='same')(x22)
    x13 = tf.keras.layers.concatenate([x10, x13])
    x13 = tf.keras.layers.BatchNormalization()(x13)
    x13 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x13)
    x13 = tf.keras.layers.BatchNormalization()(x13)
    x13 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x13)
    x13 = tf.keras.layers.Dropout(0.2)(x13)

    x04 = tf.keras.layers.Conv2DTranspose(16, kernel_size=(2, 2), strides=(2, 2), padding='same')(x13)
    x04 = tf.keras.layers.concatenate([x00, x04])
    x04 = tf.keras.layers.BatchNormalization()(x04)
    x04 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x04)
    x04 = tf.keras.layers.BatchNormalization()(x04)
    x04 = tf.keras.layers.Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x04)
    x04 = tf.keras.layers.Dropout(0.2)(x04)

    outputs = tf.keras.layers.Conv2D(1, kernel_size=(1, 1), activation='sigmoid')(x04)

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
