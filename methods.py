import numpy as np


def get_wanted(data, index, U):
    return np.array(U[:, int(data[index, 2])-1])

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def diff(h):
    diff = h * (1 - h)
    return diff