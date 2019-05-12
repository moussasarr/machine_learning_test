#Demonstrating linear regression formulas through code

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Load data/data_1d.csv
D = pd.read_csv("./data/data_1d.csv", header=None)
A = np.array(D)

X = A[:, 0]
Y = A[:, 1]
#Plotting the data



#Plot is in a line shape (Yi = mXi + b)
#Goal is to derive m and b using linear regression formulas

#Square of X
Xsq = (X**2)
#Mean of square of X
meanXsq = Xsq.mean()

#Mean of X
meanX = X.mean()
#Square of Mean of X
SqMeanX = meanX ** 2

#Denominator formula for calc a and b
denominator = meanXsq - SqMeanX

#Mean of Y
meanY = Y.mean()

#Calc...mean XY
XY = X * Y
meanXY = XY.mean()


#Slope m or a
m = (meanXY - (meanX * meanY))/denominator

#Calculating b the simple way
b = meanY - m*meanX
#Calculating b the another way
b2 =  (meanY*meanXsq - meanX*meanXY)/ denominator

#b1 and b2 should be about equal
print(b2)
print(b)


#Plotting the calculated line of best fit
A = np.linspace(0, 110)
bestFit = m*X + b2
plt.scatter(X, Y)
plt.plot(X, bestFit)
plt.show()





