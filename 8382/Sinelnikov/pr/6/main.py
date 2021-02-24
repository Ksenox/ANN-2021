import gens
import numpy as np
import keras
from keras import layers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""


val_fract = 0.2
test_fract = 0.2

batch_size = 16
epochs = 50

def gen_data(size=500, img_size=50):
    c1 = size // 2
    c2 = size - c1

    label_c1 = np.full([c1, 1], 'Horizontal')
    data_c1 = np.array([gens.gen_h_line(img_size) for i in range(c1)])
    label_c2 = np.full([c2, 1], 'Vertical')
    data_c2 = np.array([gens.gen_v_line(img_size) for i in range(c2)])

    data = np.vstack((data_c1, data_c2))
    label = np.vstack((label_c1, label_c2))

    return data, label

def splitData(data, label):
    digit_labels = np.ones((label.shape))
    bool_mask = label == 'Horizontal'
    digit_labels[bool_mask] = 1
    digit_labels[~bool_mask] = 0
    train_val_indixes = np.random.choice(data.shape[0], int((1-test_fract)*data.shape[0]),replace=False)
    train_val_data = data[train_val_indixes]
    train_val_labels = digit_labels[train_val_indixes]
    mask = np.ones(data.shape[0], bool)
    mask[train_val_indixes] = False
    test_data = data[mask]
    test_labels = digit_labels[mask]
    train_indixes = np.random.choice(train_val_data.shape[0], int((1 - val_fract) * train_val_data.shape[0]), replace=False)
    train_data = train_val_data[train_indixes]
    train_labels = train_val_labels[train_indixes]
    mask = np.ones(train_val_data.shape[0], bool)
    mask[train_indixes] = False
    val_data = train_val_data[mask]
    val_labels = train_val_labels[mask]
    return train_data, val_data, test_data, train_labels, val_labels, test_labels

def testEvaluation(model, test_data, test_labels):
    print("test accuracy",model.evaluate(test_data, test_labels))


def createModel(train_data, val_data, train_labels, val_labels):
    inputs = keras.Input(shape=(int(train_data.shape[1]),int(train_data.shape[0])))
    hidden_layer1 = layers.Reshape((int(train_data.shape[1]),int(train_data.shape[2]) , 1))(inputs)
    hidden_layer1 = layers.Conv2D(64,(3,3))(hidden_layer1)
    hidden_layer1 = layers.BatchNormalization()(hidden_layer1)
    hidden_layer1 = layers.Activation("relu")(hidden_layer1)
    hidden_layer1 = layers.Dropout(0.5)(hidden_layer1)
    hidden_layer1 = layers.MaxPool2D((2,2))(hidden_layer1)

    hidden_layer2 = layers.Conv2D(128, (3, 3))(hidden_layer1)
    hidden_layer2 = layers.BatchNormalization()(hidden_layer2)
    hidden_layer2 = layers.Activation("relu")(hidden_layer2)
    hidden_layer2 = layers.Dropout(0.5)(hidden_layer2)
    hidden_layer2 = layers.MaxPool2D((2, 2))(hidden_layer2)

    hidden_layer3 = layers.Conv2D(64, (3, 3))(hidden_layer2)
    hidden_layer3 = layers.BatchNormalization()(hidden_layer3)
    hidden_layer3 = layers.Activation("relu")(hidden_layer3)
    hidden_layer3 = layers.Dropout(0.5)(hidden_layer3)
    hidden_layer3 = layers.MaxPool2D((2, 2))(hidden_layer3)

    hidden_layer4 = layers.Flatten()(hidden_layer3)
    hidden_layer4 = layers.Dense(500)(hidden_layer4)
    hidden_layer4 = layers.BatchNormalization()(hidden_layer4)
    hidden_layer4 = layers.Activation("relu")(hidden_layer4)
    hidden_layer4 = layers.Dropout(0.5)(hidden_layer4)

    output_layer = layers.Dense(1)(hidden_layer4)
    res = layers.Activation("sigmoid")(output_layer)

    model = keras.Model(inputs, res)
    model.summary()

    model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001, decay=1e-3 / epochs), metrics=["accuracy"])

    H = model.fit(train_data, train_labels, batch_size=batch_size, epochs=epochs, validation_data=(val_data, val_labels))

    loss = H.history['loss']
    val_loss = H.history['val_loss']
    acc = H.history['accuracy']
    val_acc = H.history['val_accuracy']
    epochs_list = range(1, len(loss) + 1)

    plt.plot(epochs_list, loss, 'bo', label='Training loss')
    plt.plot(epochs_list, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.png")
    plt.show()

    # Построение графика точности

    plt.clf()
    plt.plot(epochs_list, acc, 'bo', label='Training acc')
    plt.plot(epochs_list, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("acc.png")
    plt.show()

    testEvaluation(model, test_data, test_labels)


data, label = gen_data()
train_data, val_data, test_data, train_labels, val_labels, test_labels = splitData(data, label)
createModel(train_data, val_data, train_labels, val_labels)