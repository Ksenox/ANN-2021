import numpy as np
from keras.models import load_model
from pathlib import Path
from tensorflow.keras.datasets import imdb 
from main import vectorize

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

    reviewVector = vectorize(np.asarray([indicies]))
    res = model.predict(reviewVector)
    return res
    


model = load_model("model_lb6.h5")
print("Enter file name with review: ")
inp = str(input())

res = predictReview(inp)
if res >= 0.5:
    print("good")
else:
    print("bad")
   