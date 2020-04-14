from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical, Sequence
from efficientnet.keras import center_crop_and_resize
from . import prepare_data as pd
from keras.models import load_model
from keras import callbacks
from efficientnet.keras import (EfficientNetB3, center_crop_and_resize,
                                preprocess_input)
from keras.layers.core import Activation
from keras.models import Model
from efficientnet.keras import EfficientNetB0, center_crop_and_resize, preprocess_input
import os

import numpy as np


BATCH_SIZE = 8
EPOCHS = 100


class MalariaSequence(Sequence):
    """Sequence that represents our Malaria dataset."""

    def _get_imgs(self, path):
        """Get list of image paths for a given `path`."""
        return [
            os.path.join(path, img)
            for img in os.listdir(path)
            if img.lower().endswith('.png')
        ]

    def _prepare_imgs(self, img_paths, image_size):
        """Process image paths into prepared images."""
        return [
            center_crop_and_resize(
                img_to_array(
                    load_img(
                        img_path,
                        target_size=image_size,
                        color_mode="rgb",
                        interpolation="bicubic"
                    ),
                    data_format='channels_last'
                ),
                image_size
            )
            for img_path in img_paths
        ]

    def __init__(
        self,
        parasitized_path,
        uninfected_path,
        image_size,
        batch_size=32,
    ):
        parasitized_imgs = self._get_imgs(parasitized_path)
        uninfected_imgs = self._get_imgs(uninfected_path)
        labels = np.array(
            ([1] * len(parasitized_imgs)) +
            ([0] * len(uninfected_imgs))
        )
        self.batch_size = batch_size
        self.image_size = image_size
        self.x = np.array(parasitized_imgs + uninfected_imgs)
        self.y = labels

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        # NOTE: Shape issue here
        x = np.array(self._prepare_imgs(batch_x, self.image_size))
        y = to_categorical(batch_y, num_classes=2)

        return x, y


def train(cell_images_path):
    print('INSTANTIATING MODEL')
    if os.path.exists('malaria.hdf5'):
        print('LOADING MODEL')
        model = load_model('malaria.hdf5')
    else:
        print('CREATING MODEL')
        model = EfficientNetB3(classes=2, weights=None)

    print('LOADING IMAGE PATHS AND LABELS')
    x_train, x_test, y_train, y_test = pd.prepare_data_lists(cell_images_path)

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD',
                  metrics=['accuracy'])

    print('TRAINING MODEL')
    model.fit_generator(pd.provide_training_data(x_train, y_train, BATCH_SIZE,
                                                 model.input_shape[1]),
                        epochs=EPOCHS,
                        steps_per_epoch=len(x_train) // BATCH_SIZE,
                        verbose=1,
                        callbacks=[
                            callbacks.EarlyStopping(monitor='loss',
                                                    min_delta=0.01,
                                                    patience=3,
                                                    mode='min',
                                                    restore_best_weights=True),
                            callbacks.ModelCheckpoint('malaria.hdf5',
                                                      monitor='loss',
                                                      mode='min',
                                                      save_best_only=True),
    ])
    print('EVALUATING MODEL')
    predict = model.evaluate_generator(
        pd.provide_validation_data(x_test, y_test, BATCH_SIZE,
                                   model.input_shape[1]),
        steps=len(x_test) // BATCH_SIZE,
        verbose=1,
    )

    print(f'LOSS: {predict[0]}')
    print(f'ACCURACY: {predict[1]}')
    print(predict)
