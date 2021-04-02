import tensorflow as tf

import numpy as np

from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from keras.models import load_model

from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-i", "-input", dest="filename",
                        help="set image", metavar="FILE")
    parser.add_argument("-m", "--model", dest="model", default="model.tf",
                        help="set model")
    args = parser.parse_args()
    return args





def get_model():


    model = Sequential()
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(10, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam()

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def build_and_fit():
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_images = train_images / 255.0
    test_images = test_images / 255.0

    from keras.utils import to_categorical

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    model = get_model()

    model.fit(train_images, train_labels, epochs=5, batch_size=128)

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    return model, test_acc, test_loss


def load_image(path, show = False):
    import cv2

    # image = cv2.imread(path)
    image = cv2.imread(path)
    # cv2.imshow('foo',image)
    # cv2.waitKey()
    image = cv2.resize(image, (28, 28))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    if (show):
        sh = cv2.resize(image, (800, 800))
        cv2.imshow("Image", sh)
        cv2.waitKey()

    # n_image = np.around(np.divide(image, 1.0))


    return image



if __name__ == "__main__":

    # Parse args

    args = parse_args()

    print(args)

    # Load or build and fit model

    model:Sequential

    try:
        model = load_model(args.model)
    except Exception:
        print("Failed to load model. Generating new model")
        model,acc ,loss = build_and_fit()
        model.save(args.model)
        print(f"Generated new model. Accuracy:{acc}")
    else:
        print(f"Loaded model {args.model}")

    # Load image
    img:np.ndarray

    if(args.filename):
        try:
            img = load_image(args.filename,show=True)
        except Exception:
            print("Failed to load image")
        else:
            # img =
            res = model(img[np.newaxis, :, :])
            res: np.ndarray = np.array(res[0])
            enum_res = list(enumerate(res))
            enum_res.sort(key=lambda x: x[1], reverse=True)
            for i in enum_res:
                print(i[0],":",i[1])

            # result = np.where(res == res.max())
            # print(result[0][0], res.max())

    else:
        print("Image not specified. Use '-i FILE'")
