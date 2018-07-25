import pandas as pd
import numpy as np
import sys
import pickle
from keras.models import Sequential, Model, load_model

def normalize(x):
    x = x/255.
    return x

# load testing data
test_path = sys.argv[1]
output_path = sys.argv[2]

test = pd.read_csv(test_path)
test_X = np.array([row.split(" ") for row in test["feature"].tolist()], dtype=float32)
test_X = normalize(test_X.reshape(-1,48,48,1))

print("Load Model.....")
model = load_model("model/model1-?????-0.?????.h5")
print("Predicting.....")
## 注意predict出來不是one-hot vector
p = model.predict(test_X)

# load sample.csv
sample = pd.read_csv("sample.csv")
sample["label"] = p
sample.to_csv(output_path, index=None)