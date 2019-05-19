import numpy as np 
import matplotlib.pyplot as plt 

X = []
Y = []

for line in open("./data/data_poly.csv"):
	x, y = line.split(",")

	x = float(x)
	X.append([1, x, x*x])
	y = float(y)
	Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X[:, 1], Y)
plt.show()

W = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
Yhat = X.dot(W.T) 

plt.scatter(sorted(X[:, 1]), sorted(Y))
plt.plot(sorted(X[:, 1]), sorted(Yhat))
plt.show()

dt1 =  Y - Yhat
dt2 =  Y - Y.mean()

R2 = 1 - (dt1.dot(dt1)).sum() / dt2.dot(dt2).sum()
print("R-square value is ", R2)


