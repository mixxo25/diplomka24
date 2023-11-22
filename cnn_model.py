import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate
import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.utils import to_categorical


def create_unet_model(input_size=(256, 256, 1)):
    inputs = Input(input_size)

    # Down-sampling
    conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)



    conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    # Bottleneck
    conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)

    # Up-sampling
    up1 = UpSampling2D(size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation='relu', padding='same')(up1)
    merged1 = Concatenate()([conv2, conv4])

    up2 = UpSampling2D(size=(2, 2))(merged1)
    conv5 = Conv2D(64, 3, activation='relu', padding='same')(up2)
    merged2 = Concatenate()([conv1, conv5])

    output = Conv2D(1, 1, activation='sigmoid')(merged2)

    model = Model(inputs=inputs, outputs=output)

    return model

model = create_unet_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Základná cesta k priečinku s datasetom
base_dataset_dir = 'Dataset/'

# Načítanie párových CSV súborov
train_df = pd.read_csv(os.path.join(base_dataset_dir, 'pairs_training.csv'), delimiter=';', header=None)
valid_df = pd.read_csv(os.path.join(base_dataset_dir, 'pairs_valid.csv'), delimiter=';', header=None)
test_df = pd.read_csv(os.path.join(base_dataset_dir, 'pairs_test.csv'), delimiter=';', header=None)

# Funkcia na načítanie dvojíc obrázkov a ich labelov
def load_image_pairs_and_labels(df, img_dir):
    image_pairs = []
    labels = []

    for _, row in df.iterrows():
        # Zostavenie cesty k obrázkom podľa aktuálneho riadku v dataframe
        img_path_1 = os.path.join(img_dir, row[0].strip())
        img_path_2 = os.path.join(img_dir, row[1].strip())

        # Načítanie a predspracovanie obrázkov
        img_1 = load_img(img_path_1, target_size=(256, 256))
        img_1 = img_to_array(img_1)

        img_2 = load_img(img_path_2, target_size=(256, 256))
        img_2 = img_to_array(img_2)

        # Spojenie obrázkov a pridanie do zoznamu
        image_pairs.append(np.concatenate((img_1, img_2), axis=-1))

        # Pridanie labelu
        labels.append(row[2])

    return np.array(image_pairs), to_categorical(np.array(labels))

# Načítanie a predspracovanie trénovacích, validačných a testovacích dát
train_image_pairs, train_labels = load_image_pairs_and_labels(train_df, os.path.join(base_dataset_dir, 'train'))
valid_image_pairs, valid_labels = load_image_pairs_and_labels(valid_df, os.path.join(base_dataset_dir, 'valid'))
test_image_pairs, test_labels = load_image_pairs_and_labels(test_df, os.path.join(base_dataset_dir, 'test'))

# Normalizácia dát
train_image_pairs /= 255.0
valid_image_pairs /= 255.0
test_image_pairs /= 255.0

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
# Predpokladá sa, že train_image_pairs je numpy pole obsahujúce spojené dvojice obrázkov
train_generator = train_datagen.flow(train_image_pairs, train_labels, batch_size=32)

# Trénovanie modelu (predpokladáme, že máte vytvorenú a skompilovanú modelovú architektúru)
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_image_pairs) // 32,  # Prípadne upravte podľa veľkosti vášho batchu
    epochs=20,
    validation_data=(valid_image_pairs, valid_labels)
)

# Vytvorenie priečinka, ak neexistuje
model_directory = 'Model'
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

# Uloženie celého modelu na disk
model.save(os.path.join(model_directory, 'model_segmentacia_textu.h5'))

# Uloženie architektúry, váh, konfigurácie tréningu, stavu optimalizátora
model.save(os.path.join(model_directory, 'kompletny_model_segmentacia_textu'))

# Uloženie iba váh modelu
model.save_weights(os.path.join(model_directory, 'vahy_modelu_segmentacia_textu.h5'))
