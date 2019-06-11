#Demonstrating how L2-Regularization can improve ML - model
#in case of outliers

import numpy as np
import matplotlib.pyplot as plt 

N = 50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)

#making outliers
Y[-1] += 30
Y[-2] += 30

X = np.vstack((X, np.ones(N))).T

#Find the line of maximum likelihood of Y
W_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Y_ml = X.dot(W_ml)

plt.scatter(X[:, 0], Y)
plt.plot(X[:,0], Y_ml)
plt.show()

l = 1000
W_l2 = np.linalg.solve(l*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_l2 = X.dot(W_l2)

plt.scatter(X[:, 0], Y)
plt.plot(X[:, 0], Y_ml, label="maximum likelihood")
plt.plot(X[:, 0], Y_l2, label="L2-Regularization")
plt.legend()
plt.show()


