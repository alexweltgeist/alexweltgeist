# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 11:08:26 2021

@author: alex
"""

# Install if missing
# ! pip install tensorflow_hub

import os, shutil
import matplotlib.pylab as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")

# Download data from Kaggle or Git to local directory ('/test')
# https://github.com/nicolas-gervais/predicting-car-price-from-scraped-data/tree/master/picture-scraper

# subfolder erstellen
sourcefile = 'C:/Users/alex/CAS_ML_local/B_Deeplearning/03_Project/Data'

# Filename auslesen und Orderstruktur anlegen
for filename in os.listdir(sourcefile): 
    brand = filename.rsplit('_', 17)[0]
    try:
        os.mkdir(os.path.join(sourcefile, brand))   # Ordner erstellen wenn nötig...
    except WindowsError:
        pass                                        # ...sonst weiter und Bild moven
    shutil.move(os.path.join(sourcefile, filename), os.path.join(sourcefile, brand, filename))

'''
def get_nr_files(sourcefile):
    file_count = 0
    for r, d, files in os.walk(sourcefile):
        file_count += len(files)
    return file_count
# shoud return 64467
'''

# Daten in TensorFlow laden und spliten
para_kwargs = dict(
    directory=sourcefile, 
    labels='inferred', 
    label_mode='int',
    class_names=None, 
    color_mode='rgb', 
    batch_size=32, 
    image_size=(224, 224), 
    shuffle=False, 
    seed=None, 
    validation_split=0.2,
    interpolation='bilinear', 
    follow_links=False)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    **para_kwargs, 
    subset='training')

valid_ds = tf.keras.preprocessing.image_dataset_from_directory(
    **para_kwargs, 
    subset='validation')

# Visualisieren der eingelesenen Daten
class_names = train_ds.class_names
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

# Pre-trained Model auswählen (aus TF-Hub)
model_name = "mobilenet_v2_100_224" 
model_handle_map = {"mobilenet_v2_100_224": "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4",}
model_handle = model_handle_map.get(model_name)
IMAGE_SIZE = (224, 224, 3)
BATCH_SIZE = 32
print(f"\nAusgewaehltes model: {model_name} : {model_handle}")
print(f"\nBild-Groesse {IMAGE_SIZE}")

# CNN zusammenstellen mit ANzahl Klassen wie im Datenset 
print("\nModell erstellen mit", model_handle)
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(IMAGE_SIZE)),
    hub.KerasLayer(model_handle, trainable=False),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(class_names), kernel_regularizer=tf.keras.regularizers.l2(0.0001))
])
model.build((None,)+IMAGE_SIZE)
model.summary()

# Trainieren des Modells 
model.compile(
  optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9), 
  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
  metrics=['accuracy'])

hist = model.fit(
    train_ds,
    epochs=5, 
    validation_data=valid_ds).history


def get_nr_files(sourcefile):
    file_count = 0
    for r, d, files in os.walk(sourcefile):
        file_count += len(files)
    return file_count


