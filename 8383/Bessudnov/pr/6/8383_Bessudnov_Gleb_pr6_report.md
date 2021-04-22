# Практическое задание №5
## Задание

Необходимо построить сверточную нейронную сеть, которая будет классифицировать черно-белые изображения с простыми геометрическими фигурами на них.

К каждому варианту прилагается код, который генерирует изображения.

Для генерации данных необходимо вызвать функцию gen_data, которая возвращает два тензора:

- Тензор с изображениями ранга 3
- Тензор с метками классов

Обратите внимание:

- Выборки не перемешаны, то есть наблюдения классов идут по порядку
- Классы характеризуются строковой меткой
- Выборка изначально не разбита на обучающую, контрольную и тестовую
- Скачивать необходимо оба файла. Подключать файл, который начинается с var (в нем и находится функция gen_data)

### Вариант 3
Классификация изображений с горизонтальной или вертикальной линией

## Выполнение

Для начала в работу были подключены все необходимые зависимости и файл *Var3.py*
```Python
import var3
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.python.keras import utils
from sklearn.preprocessing import LabelEncoder
```

Было решено использовать малый набор данных и провести перекресную проверку по К-блокам. Для реализации этого метода, была написана функция, которая создает модель. В задании поставлена задача бинарной классификации, поэтому выходной слой имеет два нейрона и функцию активации *sigmoid*, функция потерь - бинарная кроссэнтропия. Ниже представлена функция *build_model*, предназначенная для построения модели:

```Python
def build_model():
    inp = Input(shape=(image_size, image_size, 1))

    # Слои Свертки и пуллинга
    conv_1 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(inp)
    conv_2 = Convolution2D(conv_depth_1, (kernel_size, kernel_size), padding='same', activation='relu')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

    # Слой Dropout
    drop_1 = Dropout(drop_conv)(pool_1)

    # Еще слои свертки и пуллинга
    conv_3 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(drop_1)
    conv_4 = Convolution2D(conv_depth_2, (kernel_size, kernel_size), padding='same', activation='relu')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)

    # Еще слой Dropout
    drop_2 = Dropout(drop_conv)(pool_2)


    flat = Flatten()(drop_2)
    hidden = Dense(hidden_size, activation='relu')(flat)
    drop_3 = Dropout(drop_dense)(hidden)
    out = Dense(2, activation='sigmoid')(drop_3)

    model = Model(inputs=inp, outputs=out) 
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
```
Далее были заданы некоторые гипер параметры модели, в частности размер набора данных, размер изображений, количество К-блоков, размер батча, количество эпох, размерность ядра свертки, размерность слоя пуллинга, вероятности слоя *Dropout* и количество нейронов на разных слоях.

```Python
# Гипер параметры
data_size = 1000
image_size = 50

K = 4
batch_size = 10
epochs = 40
kernel_size = 3
pool_size = 2
drop_conv = 0.1
drop_dense = 0.5
conv_depth_1 = 32 
conv_depth_2 = 64
hidden_size = 32
```

Затем были сгенерированны данные, подготовлены метки. Набор был перемещан.

```Python
(data, label) = var3.gen_data(size=data_size, img_size=image_size)

encoder = LabelEncoder()
encoder.fit(label)
label = encoder.transform(label) 
label = utils.np_utils.to_categorical(label, 2)

# Перемещивание данных
rand_index = np.random.permutation(len(label))
train_data = data[rand_index]
train_targets = label[rand_index]

num_val_samples = len(train_data) // K
all_scores = []
```
Далее был применен метод проверки по К-блокам. В результате точность модели составила 89.25%.

```Python
for i in range(K):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples], 
                                        train_data[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], 
                                           train_targets[(i + 1) * num_val_samples:]], axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, epochs=epochs, batch_size=batch_size, validation_split=0, verbose=0)
    val_bc, val_acc = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_acc)

print(np.mean(all_scores))
```
