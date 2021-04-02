import pandas
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import plot_model

from tensorflow.keras import optimizers

from PIL import Image
import warnings

from tensorflow.keras.datasets import mnist


def image_2_tensor(image_name, save_name=None, resize_=False, 
                                 brightness=0.8, newsize=(28, 28), format='JPEG', 
                                 is_bw=False, norm=False, plot_num=False):
    try:
        img = Image.open(image_name)
    except FileNotFoundError:
        print('Error name file')
        return
   
    if resize_:
        warnings.warn('The image need to resize!')
        img = img.resize(newsize) 
    
    if is_bw == False: 
        separator = 255 / brightness / 2 * 3
        for x in range(img.size[0]):
            for y in range(img.size[1]):
                r, g, b = img.getpixel((x, y))
                total = r + g + b
                if total > separator:
                    img.putpixel((x, y), (0, 0, 0))
                else:
                    img.putpixel((x, y), (250, 250, 250))
    
    res_tensor = np.asarray(img, dtype='uint8')
    
    if len(res_tensor.shape) > 2:
        height, width, _ = res_tensor.shape
        res_tensor = np.round(np.sum(res_tensor/3, axis=2)).astype(np.uint8).reshape((height, width))
    
    if save_name:
        img2 = Image.fromarray(res_tensor)
        img2.save(save_name, format)
    
    if norm:
        res_tensor = res_tensor.astype('float32') / 255
    
    if plot_num:
        plt.imshow(res_tensor, cmap=plt.cm.binary)
        plt.show()
    return res_tensor 


def get_test_set(file_names, is_bw_=False, plot_num=False, resize_=True):
    test_set = []
    for i in range(len(file_names)):
        test_set.append(image_2_tensor(file_names[i], resize_=resize_, norm=True, is_bw=is_bw_, plot_num = plot_num))
    
    return np.asarray(test_set)
    


# загрузка набора данных mnist 60000 & 10000
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# подготовка данных [0, 1]
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255



# к категориальному вектору (5 == [0, 0, 0, 0, 0, 1, 0, ... 0])
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# создание модели сети
model = Sequential()
#model.add(Flatten())
model.add(Dense(256, activation='relu', input_shape=(28 * 28,)))
#model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

#optimizer='rmsprop'
#model.compile(optimizer= optimizers.Adam(learning_rate=0.001),  loss='categorical_crossentropy', metrics=['accuracy'])
model.compile(optimizer= optimizers.Adamax(learning_rate=0.2, beta_1=0.9, beta_2=0.999),  loss='categorical_crossentropy', metrics=['accuracy'])

# обучение сети
epochs_num = 5
history = model.fit(train_images, train_labels, epochs=epochs_num, batch_size=128, validation_data=(test_images, test_labels))


# проверка на контрольном наборе
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc:', test_acc) 


# схема модели
plot_model(model, to_file='model.png', show_shapes=True)
print(model.summary())

# Провека модели на изображениях
images_names_1 = ['images/1/0.png', 
                                'images/1/1.png', 
                                'images/1/2.png', 
                                'images/1/3.png', 
                                'images/1/4.png', 
                                'images/1/5.png', 
                                'images/1/6.png',
                                'images/1/7.png',
                                'images/1/8.png',
                                'images/1/9.png']
                                
images_names_2 = ['images/2/0.jpg', 
                                'images/2/1.jpg', 
                                'images/2/2.jpg', 
                                'images/2/3.jpg', 
                                'images/2/4.jpg', 
                                'images/2/5.jpg', 
                                'images/2/6.jpg']


def test_on_images(model, images_names):
    test_set = get_test_set(images_names, is_bw_=False, resize_=True, plot_num=False)
    test_set = test_set.reshape((len(images_names), 28 * 28))

    result_img = model.predict(test_set)
    print(result_img[0])
    for i in range(len(result_img)):
        print("I think number: ", list(result_img[i]).index(max(result_img[i])))
    print(result_img)
                                

test_on_images(model, images_names_2)
