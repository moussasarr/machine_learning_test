import numpy as np
import matplotlib.pyplot as plt 

N = 50
X = np.linspace(0, 10, N)

Y = 0.5*X + np.random.randn(N)

Y[-1] += 30
Y[-2] += 30

X = np.vstack((X, np.ones(N))).T

plt.scatter(X[:, 0], Y)
plt.show()

#Maximum Likelihood 
W_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat_ml = X.dot(W_ml)

#L2 Regularization
l = 1000.0
W_l2 = np.linalg.solve(l*np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_l2 = X.dot(W_l2)

plt.scatter(X[:, 0], Y)
plt.plot(X[:, 0], Yhat_ml, label="Maximum Likelihood")
plt.plot(X[:, 0], Y_l2, label="L2 Regularization")
plt.legend()
plt.show()





