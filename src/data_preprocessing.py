import tensorflow as tf
import tensorflow_datasets as tfds

def load_and_preprocess_image(image_path, target_size=(520, 520)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, target_size)
    image = image / 255.0
    return image

def preprocess_image(image, label):
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0
    return image, label

def load_mamah_dataset():
    dataset, info = tfds.load('mamah/2024', with_info=True, as_supervised=True)
    train_dataset = dataset['train']

train_dataset = train_dataset.map(preprocess_image)
train_dataset = train_dataset.batch(32).shuffle(1000)

return train_dataset, info

