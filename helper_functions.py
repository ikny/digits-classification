import numpy as np


def vector_sigmoid(v: np.array) -> np.array:
    """ Aplikuje sigmoid na každou složku vektoru v """
    return 1/(1+np.exp(-v))


def softmax(v: np.array) -> np.array:
    """ Aplikuje softmax na vektor v """
    return np.exp(v)/np.sum(np.exp(v))
