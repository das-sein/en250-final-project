import json
import os

import numpy as np
from efficientnet.tfkeras import (
    center_crop_and_resize,
    preprocess_input
)
from efficientnet.model import EfficientNet
from tensorflow.keras import (
    backend,
    callbacks,
    layers,
    models,
    utils,
)
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.models import load_model
from kerastuner.tuners import Hyperband
from matplotlib import pyplot as plt


backend.set_image_data_format('channels_last')

BATCH_SIZE = 32
EPOCHS = 100


def build_model(hp):
    model = EfficientNet(
        1.0,
        1.0,
        128,
        hp.Choice(
            'dropout_rate',
            values=[1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
            default=2e-1,
        ),
        backend=backend,
        classes=2,
        drop_connect_rate=hp.Choice(
            'drop_connect_rate',
            values=[1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
            default=2e-1,
        ),
        include_top=True,
        input_shape=(128, 128, 3),
        input_tensor=None,
        layers=layers,
        model_name='efficientnet-b0',
        models=models,
        pooling=None,
        utils=utils,
        weights=None,
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer=RMSprop(
            learning_rate=0.256,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0,
        ),
        metrics=['categorical_accuracy']
    )
    return model


def train(cell_images_path):
    print('INSTANTIATING MODEL')
    if os.path.exists('malaria.hdf5'):
        print('LOADING MODEL')
        model = load_model('malaria.hdf5')
    else:
        print('CREATING MODEL')

    print('LOADING IMAGE PATHS AND LABELS')
    x_train, x_test, y_train, y_test = pd.prepare_data_lists(cell_images_path)

    print('TUNING MODEL')
    tuner = Hyperband(
        build_model,
        objective='val_categorical_accuracy',
        max_epochs=10,
        directory='hyperband',
        project_name='malaria',
        hyperband_iterations=81,
    )
    print(tuner.search_space_summary())
    tuner.search(
        pd.provide_training_data(
            x_train,
            y_train,
            BATCH_SIZE,
            128
        ),
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        validation_data=pd.provide_validation_data(
            x_test,
            y_test,
            BATCH_SIZE,
            128
        ),
        validation_steps=len(x_test) // BATCH_SIZE,
        callbacks=[
            callbacks.EarlyStopping(
                monitor='val_loss',
                min_delta=0.1,
                patience=3,
                mode='min',
                restore_best_weights=True
            ),
        ]
    )
    print(tuner.results_summary())
    model = tuner.get_best_models(num_models=1)[0]

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
                restore_best_weights=True
            ),
            callbacks.ModelCheckpoint(
                'malaria.hdf5',
                monitor='loss',
                mode='min',
                save_best_only=True
            ),
            callbacks.CSVLogger('malaria.log', append=True),
        ],
        validation_data=pd.provide_validation_data(
            x_test, y_test, BATCH_SIZE, 128
        ),
        validation_steps=len(x_test) // BATCH_SIZE,
    )

    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
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
            128,
        ),
        steps=len(x_test),
        verbose=1,
    )

    print(f'LOSS: {predict[0]}')
    print(f'ACCURACY: {predict[1]}')
