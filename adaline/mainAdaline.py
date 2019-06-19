import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mlxtend.plotting import plot_decision_regions
from adaline import Adaline

logging.basicConfig(level=logging.DEBUG)

df = pd.read_csv('../data/iris.data', header=None)

# setosa and versicolor
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# sepal length and petal length
X = df.iloc[0:100, [0,2]].values

#
ada = Adaline(epochs=10, eta=0.01).train(X, y)
plt.plot(range(1, len(ada.cost_)+1), np.log10(ada.cost_), marker='o')
plt.xlabel('Iterations')
plt.ylabel('log(Sum-squared-error)')
plt.title('Adaline - Learning rate 0.01')
plt.show()

#
ada = Adaline(epochs=10, eta=0.0001).train(X, y)
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.title('Adaline - Learning rate 0.0001')
plt.show()

#
plot_decision_regions(X, y, clf=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.show()


# FOR 40 EPOCHS
ada = Adaline(epochs=40, eta=0.01).train(X, y)
plt.plot(range(1, len(ada.cost_)+1), np.log10(ada.cost_), marker='o')
plt.xlabel('Iterations')
plt.ylabel('log(Sum-squared-error)')
plt.title('Adaline - Learning rate 0.01')
plt.show()

#
ada = Adaline(epochs=40, eta=0.0001).train(X, y)
plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Sum-squared-error')
plt.title('Adaline - Learning rate 0.0001')
plt.show()

#
plot_decision_regions(X, y, clf=ada)
plt.title('Adaline - Gradient Descent')
plt.xlabel('sepal length')
plt.ylabel('petal length')
plt.show()


# SPLITING FORMATION
for i in range(0,40):
	ada = Adaline(epochs=i, eta=0.0001).train(X, y)

	plot_decision_regions(X, y, clf=ada)
	plt.title('Adaline - Gradient Descent')
	plt.xlabel('sepal length')
	plt.ylabel('petal length')
	plt.show()