import numpy as np
from helper_functions import *
import pickle
from constants import *



class NeuralNetwork():
    def __init__(self) -> None:
        """ Založí H a K vrstvy neuronové sítě, weights jsou náhodné a biases jsou nulové. """
        self.h_weights = np.subtract(np.random.rand(D_LAYER, H_LAYER), 0.5)
        self.h_biases = np.zeros(H_LAYER)
        self.y_weights = np.subtract(np.random.rand(H_LAYER, K_LAYER), 0.5)
        self.y_biases = np.zeros(K_LAYER)

    def forward_pass(self, x) -> np.array:
        """ Z hodnot pixelů jednoho obrázku X vypočítá excitaci neuronů v K vrstvě, kterou vrátí v podobě vektoru """
        h_excitation = self.compute_h(x=x)
        y_excitation = self.compute_y(h=h_excitation)
        return y_excitation

    def backward_pass(self, x, y_hat) -> None:
        """ Updatuje svoje parametry (weights a biases) na základě trénovacích dat"""
        h = self.compute_h(x)
        y = self.compute_y(h)
        dL_by = (2/K_LAYER)*(y - y_hat)*y*(1-y)
        dL_wy = np.einsum("k,j -> jk", dL_by, h)
        dL_bh = np.einsum("m,jm,j,j -> j", dL_by, self.y_weights, h, 1-h)
        dL_wh = np.einsum("j,i -> ij", dL_bh, x)

        self.y_biases -= ALPHA*dL_by
        self.h_biases -= ALPHA*dL_bh
        self.y_weights -= ALPHA*dL_wy
        self.h_weights -= ALPHA*dL_wh

    def compute_h(self, x) -> np.array:
        """ Vypočítá excitaci neuronů v H vrstvě z H parametrů a dat X """
        return vector_sigmoid(np.einsum("ij,i->j", self.h_weights, x) + self.h_biases)

    def compute_y(self, h) -> np.array:
        """ Vypočítá excitaci neuronů v H vrstvě z H parametrů a dat X """
        return softmax(np.einsum("ij,i->j", self.y_weights, h) + self.y_biases)

    def accuracy(self, train_xes, train_ys) -> float:
        """ vrátí procento úspěšnosti na daných testovacích datech"""
        correct = 0
        for i in range(len(train_xes)):
            answer = np.argmax(self.forward_pass(train_xes[i]))
            if answer == np.argmax(train_ys[i]):
                correct += 1
        return correct/len(train_xes)

    def export(self, name: str) -> None:
        """ Exportuje svoje parametry do pickle souboru """
        theta = [self.h_weights, self.h_biases, self.y_weights, self.y_biases]
        pickle.dump(theta, open(f"theta_{name}.p", "wb"))