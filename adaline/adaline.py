import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class Adaline(object):
    
    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        logging.info("STARTING TRAINING ...")

        self.w_ = np.zeros(1 + X.shape[1])
        logging.info("	W: {}".format(self.w_))

        self.cost_ = []
        for i in range(self.epochs):
            logging.info("  EPOCH: {}".format(i))
            
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0

            logging.info("	W: {}".format(self.w_))
            logging.info("      COST: {}".format(cost))
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)