import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from efficientnet.tfkeras import center_crop_and_resize, preprocess_input
from matplotlib import pyplot as plt

input_img = load_img('cell.png', target_size=(128, 128))
img = img_to_array(input_img, data_format='channels_last')
img = center_crop_and_resize(img, 128, crop_padding=0)
plt.imshow(img)
plt.show()
img = preprocess_input(np.expand_dims(img, 0))

model = load_model('2020-04-13_malaria_b32.hdf5')

print(img.shape)

print(model.predict(img))