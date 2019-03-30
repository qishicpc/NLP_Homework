import numpy as np
from keras.utils import plot_model
from IPython.display import Image
import pandas as pd
import random, string
import os

# Read data from file
def load_data(filename='plot_summaries_tokenized.txt'):
    text = []
    with open(filename, encoding="utf8") as file:
        for line in file:
            tokens = line.strip().split(" ")
            text.append([t.lower() for t in tokens if t != ''])

    return text

# Display Keras Model
def show_model(model, show_shapes=True, show_layer_names=False):
    filename = "".join(random.choices(string.ascii_uppercase, k=10)) + ".png"
    plot_model(model, to_file=filename, show_shapes=show_shapes, show_layer_names=show_layer_names)
    image = Image(filename)
    os.remove(filename)
    return image

