from keras.models import load_model
from PIL import Image
import numpy as np
from pathlib import Path

def predict_all():
    for i in range(10):
        path = Path(str(i) + ".png")
        img = Image.open(path.absolute())
        img = np.array(img) / 255.0
        pred = model.predict(np.array([img]))
        print("Prediction for numder", i, ":", np.argmax(pred))

def predict_one(filename):
    path = Path(filename)
    if (not path.exists()):
        print("Oops! Can't find file", filename)
        return
    img = Image.open(path.absolute())
    img = np.array(img) / 255.0
    pred = model.predict(np.array([img]))
    print("Prediction for file", filename, ":", np.argmax(pred)) 
    


model = load_model("model_lb4.h5")
print("Predict all files?(y/n)")
inp = str(input())
while inp != "y" and inp != "n":
    print("Wrong input. Predict all files?(y/n)")
    inp = input()
if inp == "y":
    predict_all()
else:
    print("Enter file name:")
    filename = input()
    predict_one(filename)    
