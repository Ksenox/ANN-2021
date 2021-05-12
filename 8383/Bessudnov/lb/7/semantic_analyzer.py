import numpy as np
from keras.models import load_model
from pathlib import Path
from tensorflow.keras.datasets import imdb 
from keras.preprocessing import sequence

def predictReview(filename):
    path = Path(filename)
    if (not path.exists()):
        print("Oops! Can't find file", filename)
        return

    reviewStr = ""
    with open(path.absolute()) as f:
        reviewStr = f.read()

    reviewStr = "".join(char for char in reviewStr if char.isalpha() or char.isspace() or char == "'").lower().strip().split()
    indicies = []
    wordsIndicies = imdb.get_word_index()

    for word in reviewStr:
        i = wordsIndicies.get(word)
        if i is not None and i < 10000:
            indicies.append(i + 3)

    indicies = np.array([indicies])
    indicies = sequence.pad_sequences(indicies, maxlen=500)

    scores = np.zeros(len(models))
    for i in range(len(models)):
        scores[i] = models[i].predict(indicies)
    
    return scores
    

models = []
for i in range(4):
    model = load_model("model_lb7_" + str(i % 2) + "_" + str(i // 2) + ".h5")
    models.append(model)
    
print("Enter file name with review: ")
inp = str(input())

res = predictReview(inp)
print(res)

answer = np.mean(res)
print(answer)

if answer >= 0.5:
    print("good")
else:
    print("bad")