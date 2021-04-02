import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from tensorflow.keras.models import load_model

def load_usr_img(usr_path):
    img = Image.open(usr_path).convert('L')      #преобразовать в черно-белый
    img = img.resize((28, 28))                   #изменить размер
    img = np.array(img)                          #конвертируем в массив numpy
    img = 1 - img / 255                          #нормализация и инверсия
    return np.expand_dims(img, axis=0)           #predict ожидает трехмерный тензон

def work_with_usr():
    print("Enter the path to the image")
    path = input()
    while not path:
        print("Try it again")
        path = input()
    img = load_usr_img(path)
    model = load_model("ins.h5")                #загрузка инс из файла
    prediction = model.predict(img)             #применение сети для распознавания

    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()
    print("Network prediction: ", np.argmax(prediction))

#считывание картинки
print("Want to upload your own image? Enter: y or n.")
answ = input()
if answ == 'y':
    work_with_usr()
