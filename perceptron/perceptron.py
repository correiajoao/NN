import logging
import numpy as np

logging.basicConfig(level=logging.DEBUG)

class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta #Learning rate
        self.epochs = epochs #Learning iterations

    def train(self, X, y):
        logging.info("STARTING TRAINING ...")

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.epochs):
            logging.info("  EPOCH: {}".format(i))
            errors = 0

            for xi, target in zip(X, y):
                logging.info("      TRAINING INSTANCE: {} -- TARGET: {}".format(xi, target))
                update = self.eta * (target - self.predict(xi))
                
                self.w_[1:] +=  update * xi
                self.w_[0] +=  update

                logging.info("          WEIGHT: {}".format(self.w_))
                logging.info("          UPDATE: {}".format(update))


                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)