import os

import numpy as np
import matplotlib as plt
from efficientnet.keras import EfficientNetB0, center_crop_and_resize, preprocess_input
from keras.models import Model
from keras.layers.core import Activation

from . import prepare_data as pd


from efficientnet.keras import center_crop_and_resize, preprocess_input
from keras.utils import to_categorical, Sequence
from keras.preprocessing.image import img_to_array, load_img


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
    model = EfficientNetB0(weights=None, classes=2)

    # x_train, x_test, y_train, y_test = pd.prepare_data(
    # cell_images_path, model.input_shape[1])

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD', metrics=['accuracy'])

    model.fit(MalariaSequence('data/cell_images/Parasitized/',
                              'data/cell_images/Uninfected/', model.input_shape[1], 32), epochs=1)
    predict = model.evaluate(MalariaSequence('data/cell_images/Parasitized/',
                                             'data/cell_images/Uninfected/', model.input_shape[1], 32))
    # trained = model.fit(x_train, y_train, epochs=1, batch_size=32,
    # verbose=1, workers=8, use_multiprocessing=True)
    # predict = model.evaluate(x_test, y_test)

    print(f'loss: {predict[0]}')
    print(f'accuracy: {predict[1]}')
