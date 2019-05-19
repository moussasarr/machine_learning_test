import numpy as np 
import matplotlib as mplt
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

X = []
Y = []

for line in open("./data/data_2d.csv"):
	x0, x1, y = line.split(",")
	X.append([float(x0), float(x1), 1])
	Y.append(float(y))
#print(X)
#print(Y)

#Transforming the lists to numpy arrays
X = np.array(X)
Y = np.array(Y)

#Solve the multi-d linear regression using the explicit formulas
W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
print(W)

#Predicting Y[0], the first row's value of y
#Y = WX
#Y[0] = WX.Transpose[0]
#Means that Y[0] is the actual Dot product of X[0] with W[0]
X0 = X[0]
Y0 = (X0 * W).sum()
print(Y0)

#Predicting value of W using np.linalg.solve
W0 = np.linalg.solve(X.T.dot(X), X.T.dot(Y))
print(W0)

print(mplt.__version__)
# Gives same result as W
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:,0], X[:,1], Y)
plt.show()

#Let 's check how good our model is by computing the Rsquare
Yhat = X.dot(W)
SSres = (Y - Yhat).dot(Y - Yhat)
SStot = (Y - Y.mean()).dot(Y - Y.mean())

R2 = 1 - SSres/SStot
print("R-square:", R2)

















