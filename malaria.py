import os
import sys

import numpy as np
import tensorflow as tf
from efficientnet.tfkeras import (center_crop_and_resize, preprocess_input,
                                  EfficientNetB4)
from sklearn.model_selection import train_test_split
from tensorflow.keras.backend import set_image_data_format
from tensorflow.keras.preprocessing.image import (ImageDataGenerator, load_img,
                                                  img_to_array)
from tensorflow.keras.callbacks import (CSVLogger, EarlyStopping,
                                        ModelCheckpoint)
from tensorflow.keras.utils import to_categorical, Sequence

###
# Important paths
##a
DATA_PATH = 'data/cell_images/'
PARASITIZED_PATH = 'data/cell_images/Parasitized/'
UNINFECTED_PATH = 'data/cell_images/Uninfected/'
MODEL_PATH = 'malaria.hdf5'

###
# Model settings
###
DATA_FORMAT = 'channels_last'
BATCH_SIZE = 32
EPOCHS = 1000
IMAGE_SIZE = 224
print('BATCH SIZE', BATCH_SIZE, 'EPOCHS', EPOCHS, 'IMAGE SIZE', IMAGE_SIZE)

set_image_data_format('channels_last')

categories = {
    'parasitized': 1,
    'uninfected': 0,
}

with tf.device('/device:GPU:0'):
    model = EfficientNetB4(weights=None,
                           input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
                           classes=2)
print('INPUT SHAPE', model.input_shape[1])


def process_img(path, image_size):
    img = load_img(path, target_size=(image_size, image_size))
    img_arr = img_to_array(img)
    img_arr = center_crop_and_resize(img_arr,
                                     image_size=image_size,
                                     crop_padding=0)
    return img_arr


def load_data(parasitized_path=PARASITIZED_PATH,
              uninfected_path=UNINFECTED_PATH):
    labels = []
    images = []
    parasitized_imgs = [
        os.path.join(parasitized_path, p) for p in os.listdir(parasitized_path)
        if p.endswith('.png')
    ]
    labels += [np.float32(categories['parasitized'])] * len(parasitized_imgs)
    uninfected_imgs = [
        os.path.join(uninfected_path, p) for p in os.listdir(uninfected_path)
        if p.endswith('.png')
    ]
    labels += [np.float32(categories['uninfected'])] * len(uninfected_imgs)
    images = parasitized_imgs + uninfected_imgs
    assert len(labels) == len(images)

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=2)

    x_train, x_test, y_train, y_test = train_test_split(images,
                                                        labels,
                                                        test_size=0.5,
                                                        shuffle=True)

    return x_train, x_test, y_train, y_test


with tf.device('/device:GPU:0'):
    train_datagen = ImageDataGenerator(horizontal_flip=True,
                                       vertical_flip=True,
                                       rotation_range=90,
                                       data_format=DATA_FORMAT,
                                       validation_split=0.5,
                                       dtype=np.float32)
    validation_datagen = ImageDataGenerator(data_format=DATA_FORMAT,
                                            validation_split=0.5,
                                            dtype=np.float32)
    train_gen = train_datagen.flow_from_directory(DATA_PATH,
                                                  target_size=(IMAGE_SIZE,
                                                               IMAGE_SIZE),
                                                  batch_size=BATCH_SIZE,
                                                  shuffle=True,
                                                  class_mode='categorical',
                                                  subset='training')
    validation_gen = validation_datagen.flow_from_directory(
        DATA_PATH,
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        class_mode='categorical',
        subset='validation')

with tf.device('/device:GPU:0'):
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['categorical_accuracy'],
    )

imgs_train, imgs_test, labels_train, labels_test = load_data()

with tf.device('/device:GPU:0'):
    fitted_model = model.fit(
        train_gen,
        epochs=EPOCHS,
        verbose=1,
        callbacks=[
            ModelCheckpoint(MODEL_PATH,
                            monitor='val_loss',
                            save_best_only=True,
                            save_weights_only=False,
                            mode='min',
                            period=1),
            CSVLogger('malaria.log'),
        ],
        validation_data=validation_gen,
        shuffle=True,
        use_multiprocessing=True,
        workers=os.cpu_count(),
    )

    evaluated_model = model.evaluate(
        validation_gen,
        verbose=1,
        use_multiprocessing=True,
        workers=os.cpu_count(),
    )

print("---")
print(f"LOSS: {evaluated_model[0]}")
print(f"ACCURACY: {evaluated_model[1]}")
