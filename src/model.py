import tensorflow as tf

def build_model():
    model = tf.keras.Sequential([
        tf.kears.layers.InputLayer(input_shape=(256, 256, 3)),
        tf.kears.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.kears.layers.MaxPooling2D((2, 2)),
        tf.kears.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.kears.layers.MaxPooling2D((2, 2)),
        tf.kears.layers.Conv2D(128, (3, 3), activation='relu'),
        tf.kears.layers.Flatten(),
        tf.kears.layers.Dense(128, activation='relu'),
        tf.kears.layers.Dense(10, activation='softmax')
    ])

return model
