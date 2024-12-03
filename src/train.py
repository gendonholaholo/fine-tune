import tensorflow as tf
from src.data_preprocessing import load_mamah_dataset
from src.model import build_model

train_dataset, info = load_mamah_dataset()

model = build_model()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_dataset, epochs=10)

model.save('models/fine_tuned_model/saved_model')
