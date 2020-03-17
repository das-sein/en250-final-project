import numpy as np
from efficientnet.keras import EfficientNetB0, center_crop_and_resize, preprocess_input

from . import prepare_data as pd


def train(cell_images_path):
    model = EfficientNetB0(weights=None)

    x_train, x_test, y_train, y_test = pd.prepare_data(
        cell_images_path, model.input_shape[1])

    model.compile(loss='categorical_crossentropy',
                  optimizer='SGD', metrics=['accuracy'])

    trained = model.fit(x_train, y_train, epochs=10, batch_size=32,
                        verbose=1, workers=8, use_multiprocessing=True)
    predict = model.evaluate(x_test, y_test)

    print(f'loss: {predict[0]}')
    print(f'accuracy: {predict[1]}')
