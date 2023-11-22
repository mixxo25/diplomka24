import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

from sklearn.model_selection import train_test_split

# Cesty k priečinkom s dátami
train_dir = 'Dataset/train'
valid_dir = 'Dataset/valid'
test_dir = 'Dataset/test'

# Načítanie párových CSV súborov
train_df = pd.read_csv('Dataset/pairs_training.csv')
valid_df = pd.read_csv('Dataset/pairs_valid.csv')
test_df = pd.read_csv('Dataset/pairs_test.csv')


# Funkcia na načítanie obrázkov a ich príslušných anotácií
def load_images_and_labels(df, img_dir):
    images = []
    labels = []

    for index, row in df.iterrows():
        img_path = os.path.join(img_dir, row['image_name'])
        img = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
        img = tf.keras.preprocessing.image.img_to_array(img)

        label = row['label']

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


# Načítanie a predspracovanie trénovacích, validačných a testovacích dát
train_images, train_labels = load_images_and_labels(train_df, train_dir)
valid_images, valid_labels = load_images_and_labels(valid_df, valid_dir)
test_images, test_labels = load_images_and_labels(test_df, test_dir)

# Normalizácia dát
train_images /= 255.0
valid_images /= 255.0
test_images /= 255.0

# Voliteľné: Data augmentation pre trénovacie dáta
train_datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Použitie data augmentation
train_generator = train_datagen.flow(train_images, train_labels, batch_size=32)
