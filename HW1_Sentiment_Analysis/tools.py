import sys, argparse
from scipy import sparse
from sklearn import linear_model
from collections import Counter
import numpy as np
import re
from sklearn.utils import Bunch

# Read data from file
def load_data(filename):
    """
    Load input data and return sklearn.utils.Bunch 
    """
    target, text = [], []
    with open(filename, encoding="utf8") as file:
        for line in file:
            cols = line.split("\t")
            target.append(cols[0])
            text.append(cols[1].rstrip())

    return Bunch(text=text, target=target)

