import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
from efficientnet.keras import center_crop_and_resize, preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical


def prepare_data_lists(
        cell_images_path: Path
) -> Tuple[List[Path], List[Path], List[int], List[int]]:
    data = []
    labels = []

    parasitized_dir = os.path.join(cell_images_path, 'Parasitized/')
    for path in os.listdir(parasitized_dir):
        if len(path) > 2 and path.endswith('.png'):
            data.append(os.path.join(parasitized_dir, path))
            labels.append(1)

    uninfected_dir = os.path.join(cell_images_path, 'Uninfected/')
    for path in os.listdir(uninfected_dir):
        if len(path) > 2 and path.endswith('.png'):
            data.append(os.path.join(uninfected_dir, path))
            labels.append(0)

    data_train, data_test, labels_train, labels_test = train_test_split(
        data,
        labels,
        test_size=0.5,
        random_state=int.from_bytes(os.urandom(4), 'little'))

    return (
        data_train,
        data_test,
        labels_train,
        labels_test,
    )


def provide_training_data(x_train, y_train, batch_size, image_size):
    assert len(x_train) == len(y_train), 'X and Y are different sizes'

    length = len(x_train)
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < length:
            limit = min([batch_end, length])

            data = []
            labels = []
            for image_path, label in zip(x_train[batch_start:limit],
                                         y_train[batch_start:limit]):
                img = img_to_array(load_img(image_path),
                                   data_format='channels_last')
                img = preprocess_input(
                    center_crop_and_resize(img, image_size=image_size))
                data.append(np.expand_dims(img, 0))
                labels.append(label)

            data = np.array(data)
            labels = np.array(labels)
            data_idx = np.arange(data.shape[0])
            np.random.shuffle(data_idx)
            # TODO: Figure out why we have an extra dimension
            data = np.squeeze(data[data_idx], 1)
            labels = labels[data_idx]

            yield (data, to_categorical(labels, num_classes=2))

            batch_start += batch_size
            batch_end += batch_size


def provide_validation_data(x_test, y_test, batch_size, image_size):
    assert len(x_test) == len(y_test), 'X and Y are different sizes'

    length = len(x_test)
    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < length:
            limit = min([batch_end, length])

            data = []
            labels = []
            for image_path, label in zip(x_test[batch_start:limit],
                                         y_test[batch_start:limit]):
                img = img_to_array(load_img(image_path),
                                   data_format='channels_last')
                img = preprocess_input(
                    center_crop_and_resize(img, image_size=image_size))
                data.append(np.expand_dims(img, 0))
                labels.append(label)

            data = np.array(data)
            labels = np.array(labels)
            data_idx = np.arange(data.shape[0])
            np.random.shuffle(data_idx)
            data = np.squeeze(data[data_idx], 1)
            labels = labels[data_idx]

            yield (data, to_categorical(labels, num_classes=2))

            batch_start += batch_size
            batch_end += batch_size
