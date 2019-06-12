import numpy as np
import matplotlib.pyplot as plt 

#Using our new method of Gradient Descent
N = 10
D = 3

X = np.zeros((N, D))

X[:,0] = 1
X[:5,1] = 1
X[5:10, 2] = 1

Y = [0]*5 + [1]*5

print(X)
print(Y)
#Because one column is a linear combination of the two others
#Specifically here, we have X[:, 0] == X[:, 1] + X[:, 2]
# So we cannot take the inverse of X.T.dot(X)
# Because X.T.dot(X) is a singular matrix
print(X.T.dot(X)) 
# We notice a singular matrix, therefore we cannot apply our usual 
# formula np.linalg.inv(SingularMatrix)

# So let ' s use Gradient Descent Technique instead to predict a close e
# enough to optimal value of w
# Let' s guess an initial value of W
cost = []
W = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001

for t in range(1000):
	Yhat = X.dot(W)
	delta = Yhat - Y
	mse = delta.T.dot(delta)/N
	cost.append(mse)
	W = W - learning_rate*X.T.dot(delta)
plt.plot(cost)
plt.show()

plt.plot(Yhat, label='prediction')
plt.plot(Y, label='target')
plt.legend()

plt.show()
