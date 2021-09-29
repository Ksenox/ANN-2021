"""
    Individual project ZOO
    Animals classification by attributes using ANN
    Authors:
        @Mikhail Ershov
        @Gordiyenko Alexandr
"""
import matplotlib.pyplot as plt
import pandas
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# reading data
csv_file = pandas.read_csv('zoo.csv', header=None)
dataset = csv_file.values
X = dataset[:, 1:17].astype(bool)
Y = dataset[:, 17]

# setting categories
# categories = {
#     0: ['aardvark', 'antelope', 'bear', 'boar', 'buffalo', 'calf', 'cavy', 'cheetah', 'deer', 'dolphin',
#         'elephant', 'fruitbat', 'giraffe', 'girl', 'goat', 'gorilla', 'hamster', 'hare', 'leopard', 'lion',
#         'lynx', 'mink', 'mole', 'mongoose', 'opossum', 'oryx', 'platypus', 'polecat', 'pony', 'porpoise',
#         'puma', 'pussycat', 'raccoon', 'reindeer', 'seal', 'sealion', 'squirrel', 'vampire', 'vole',
#         'wallaby', 'wolf'],
#     1: ['chicken', 'crow', 'dove', 'duck', 'flamingo', 'gull', 'hawk', 'kiwi', 'lark', 'ostrich', 'parakeet',
#         'penguin', 'pheasant', 'rhea', 'skimmer', 'skua', 'sparrow', 'swan', 'vulture', 'wren'],
#     2: ['pitviper', 'seasnake', 'slowworm', 'tortoise', 'tuatara'],
#     3: ['bass', 'carp', 'catfish', 'chub', 'dogfish', 'haddock', 'herring', 'pike', 'piranha', 'seahorse',
#         'sole', 'stingray', 'tuna'],
#     4: ['frog', 'frog', 'newt', 'toad'],
#     5: ['flea', 'gnat', 'honeybee', 'housefly', 'ladybird', 'moth', 'termite', 'wasp'],
#     6: ['clam', 'crab', 'crayfish', 'lobster', 'octopus', 'scorpion', 'seawasp', 'slug', 'starfish', 'worm']
# }

# setting Y dimension to be equal to X dimension
# Y = [i for category in Y for i in range(len(categories)) if category in categories[i]]

# setting categories
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = to_categorical(encoded_Y)

# building model
model = Sequential()
model.add(Dense(16, input_dim=16, activation='relu'))
model.add(Dense(48, activation='relu'))
model.add(Dense(7, activation='softmax'))

# setting training parameters
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# training model
h = model.fit(X, dummy_y, epochs=25, batch_size=5, validation_split=0.1)

# plotting training and validation accuracies and losses
h_dict = h.history

loss = h_dict['loss']
accuracy = h_dict['accuracy']
val_loss = h_dict['val_loss']
val_accuracy = h_dict['val_accuracy']

epochs = range(1, len(loss)+1)

plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'b-', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b-', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
