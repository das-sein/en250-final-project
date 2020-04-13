import json
import os

import numpy as np
from efficientnet.keras import (
    EfficientNetB3,
    center_crop_and_resize,
    preprocess_input
)
from keras import callbacks
from keras.models import load_model
from matplotlib import pyplot as plt

from . import prepare_data as pd

BATCH_SIZE = 16
EPOCHS = 100


def train(cell_images_path):
    print('INSTANTIATING MODEL')
    if os.path.exists('malaria.hdf5'):
        print('LOADING MODEL')
        model = load_model('malaria.hdf5')
    else:
        print('CREATING MODEL')
        model = EfficientNetB3(classes=2, weights=None,
                               input_shape=(128, 128, 3))

    print('LOADING IMAGE PATHS AND LABELS')
    x_train, x_test, y_train, y_test = pd.prepare_data_lists(cell_images_path)

    model.compile(
        loss='categorical_crossentropy',
        optimizer='SGD',
        metrics=['accuracy']
    )

    print('TRAINING MODEL')
    history = model.fit_generator(
        pd.provide_training_data(
            x_train, y_train, BATCH_SIZE, 128
        ),
        epochs=EPOCHS,
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        verbose=1,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0.1,
                patience=3,
                mode='min',
                restore_best_weights=True),
            callbacks.ModelCheckpoint(
                'malaria.hdf5',
                monitor='loss',
                mode='min',
                save_best_only=True),
            callbacks.CSVLogger('malaria.log', append=True),
        ],
        validation_data=pd.provide_validation_data(
            x_test, y_test, BATCH_SIZE, 128
        ),
        validation_steps=len(x_test) // BATCH_SIZE,
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('accuracy.png', format='png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('loss.png', format='png')

    print('EVALUATING MODEL')
    predict = model.evaluate_generator(
        pd.provide_validation_data(
            x_test,
            y_test,
            BATCH_SIZE,
            model.input_shape[1]
        ),
        steps=len(x_test),
        verbose=1,
    )

    print(f'LOSS: {predict[0]}')
    print(f'ACCURACY: {predict[1]}')
