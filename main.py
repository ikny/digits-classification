import sklearn.datasets
import numpy as np
from helper_functions import *
import matplotlib.pyplot as plt
from neural_network import NeuralNetwork
from constants import *


def format_y(targets):
    """ štítky naformátuje na vektory """
    formatted = [np.zeros(K_LAYER) for i in range(len(targets))]
    for i, n in enumerate(targets):
        formatted[i][n] = 1
    return formatted


def visualize(image, target, neural_network: "NeuralNetwork", name):
    """ Uloží jednoduchý obrázek, který obsahuje pixely číslice, true label a predicted label """
    predicted = np.argmax(neural_network.forward_pass(image))
    plt.gray()
    plt.matshow(image.reshape([8, 8]))
    plt.title(f"pravý štítek: {target}    predikovaný štítek: {predicted}")
    plt.savefig(f"{name}.png")


if __name__ == "__main__":
    TEST_SET_LEN = 50
    NUM_NNS = 1
    N_EPOCHS = 100
    GEN_IMGS = False

    # format data
    digits = sklearn.datasets.load_digits()
    len_data = len(digits["data"])
    train_set_len = len_data - TEST_SET_LEN

    formatted_targets = format_y(digits["target"])

    train_xes = digits["data"][:train_set_len]
    train_ys = formatted_targets[:train_set_len]
    test_xes = digits["data"][train_set_len:]
    test_ys = formatted_targets[train_set_len:]

    # mainloop
    my_nns = []
    for n in range(NUM_NNS):
        nn = NeuralNetwork()
        for epoch in range(N_EPOCHS):
            for i in range(train_set_len):
                x, y_hat = train_xes[i], train_ys[i]
                nn.backward_pass(x=x, y_hat=y_hat)

            print(
                f"testovací přesnost po {epoch} epochách: {round(nn.accuracy(test_xes, test_ys), 3)}    trénovací přesnost: {round(nn.accuracy(train_xes, train_ys), 3)}")
        my_nns.append(nn)

    acc = []
    for i, nn in enumerate(my_nns):
        acc.append(nn.accuracy(test_xes, test_ys))
        print(
            f"testovací přesnost neuronky {i}: {round(nn.accuracy(test_xes, test_ys), 3)}    trénovací přesnost: {round(nn.accuracy(train_xes, train_ys), 3)}")

    if GEN_IMGS:
        for i in range(10):
            visualize(test_xes[i], np.argmax(test_ys[i]), my_nns[0], f"img_{i}")
    
