import numpy as np


def get_wanted(data, index, U):
    return np.array(U[:, int(data[index, 2])-1])