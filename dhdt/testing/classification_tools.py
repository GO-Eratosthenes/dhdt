import numpy as np


def calculate_accuracy(Predicted, Truth):
    acc = 1 - np.divide(np.sum(np.logical_xor(Predicted, Truth)),
                        Truth.size)
    return acc
