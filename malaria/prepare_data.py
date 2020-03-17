import os
from pathlib import Path

import numpy as np
from efficientnet.keras import center_crop_and_resize, preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.utils import to_categorical


def prepare_data(cell_images_path: Path, image_size):
    data = []
    labels = []

    parasitized_dir = os.path.join(cell_images_path, 'Parasitized/')
    for path in os.listdir(parasitized_dir):
        if len(path) > 2 and path.endswith('.png'):
            img = load_img(os.path.join(parasitized_dir, path))
            data.append(
                np.expand_dims(
                    preprocess_input(
                        center_crop_and_resize(
                            img_to_array(
                                img,
                                data_format='channels_last'
                            ),
                            image_size
                        )
                    ),
                    0
                )
            )
            labels.append(1)

    uninfected_dir = os.path.join(cell_images_path, 'Uninfected/')
    for path in os.listdir(uninfected_dir):
        if len(path) > 2 and path.endswith('.png'):
            img = load_img(os.path.join(uninfected_dir, path))
            data.append(
                np.expand_dims(
                    preprocess_input(
                        center_crop_and_resize(
                            img_to_array(
                                img,
                                data_format='channels_last'
                            ),
                            image_size
                        )
                    ),
                    0
                )
            )
            labels.append(0)

    data = np.array(data)
    labels = np.array(labels)

    data_idx = np.arange(data.shape[0])
    np.random.shuffle(data_idx)
    data = data[data_idx]
    labels = labels[data_idx]

    data_train, data_test, labels_train, labels_test = train_test_split(
        data, labels, test_size=0.5, random_state=int.from_bytes(os.urandom(4), 'little'))

    return (
        data_train,
        data_test,
        to_categorical(labels_train, num_classes=2),
        to_categorical(labels_test, num_classes=2),
    )
