from sklearn.utils import Bunch
import numpy as np
import math
import random, string, os
from keras.utils import plot_model
from IPython.display import Image


# sigmoid
def sigmoid(x):
    return 1.0 / (1 + math.exp(-x))

def tanh(x):
    return math.tanh(x)

# Read the vocabulary file
def read_vocab(filename="vocab.txt"):
    """
    Load the vocab file in to the vocabulary dictionary
    """
    vocab = {}
    with open(filename, 'r') as file:
        for line in file:
            cols = line.rstrip().split("\t")
            vocab[cols[0]] = int(cols[1])

    return vocab

# Read data from file
def load_data(filename):
    """
    Load input data and return sklearn.utils.Bunch 
    """
    target, text = [], []
    with open(filename, encoding="utf8") as file:
        for line in file:
            cols = line.split("\t")
            target.append(1 if cols[0] == "pos" else 0)
            text.append(cols[1].rstrip())

    return Bunch(text=text, target=np.array(target))

# Display Keras Model
def show_model(model):
    filename = "".join(random.choices(string.ascii_uppercase, k=10)) + ".png"
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)
    image = Image(filename)
    os.remove(filename)
    return image

def save_prediction(arr, filename="prediction.csv"):
    """
    Save the prediction into file
    """
    out = open(filename, "w", encoding="utf8")
    for idx, val in enumerate(arr):
        pred = "pos" if val == 1 else "neg"
        out.write("%s,%s\n" % (idx, pred))
    out.close()