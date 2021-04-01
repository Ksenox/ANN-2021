import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array



def predict_image(model, image_path):
    img = load_img(image_path, color_mode="grayscale", target_size=(28, 28))
    pixels_array = np.array([img_to_array(img)]) / 255
    prediction_vector = model.predict(pixels_array)
    prediction = np.argmax(prediction_vector, 1)[0]
    print("Number on the image is:")
    print(prediction)
    return prediction


if __name__ == '__main__':
    if len(sys.argv) != 2:
        exit("You must specify path to image.")
    model = load_model('models/mnist_model')
    try:
        predict_image(model, sys.argv[1])
    except Exception as e:
        print(e, file=sys.stderr)
