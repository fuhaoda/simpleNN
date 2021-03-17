import random
import numpy as np

class MSE:
    def __init__(self):
        pass

    @staticmethod
    def loss_function(yhat, y):
        assert y.ndim == yhat.ndim == 1
        diff = y - yhat
        return 0.5*sum(diff*diff)

    @staticmethod
    def loss_derivative(yhat, y):
        return yhat - y


class SequentialNetwork:
    def __init__(self, loss = None):
        print("Initialize Network ...")
        self.layers =[]
        if loss is None:
            self.loss = MSE()

    def add(self, layer):
        self.layers.append(layer)
        layer.describe()
        if len(self.layers) > 1:
            self.layers[-1].connect(self.layers[-2])

    def train(self, training_data, epochs, mini_batch_size, learning_rate, test_data = None):
        n = len(training_data)
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches =[training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.train_batch(mini_batch, learning_rate)
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}:{1}/{2}".format(epoch, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(epoch))

