#Verifying linear regression formulas through code
#Using the summation formulas

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
#Goal is to derive m and b using the linear regression summation formulas
denominator = X.dot(X) - (X.mean() * X.sum())
m =  (Y.dot(X) - Y.mean() * X.sum())/ denominator
b = (Y.mean() * X.dot(X) - X.mean() * Y.dot(X))/ denominator
# calculated line of best fit
bestFit = m*X + b

# Calculating R-squared
ERRi = Y - bestFit
SSTOTi = Y - Y.mean()
SSresidual = ERRi.dot(ERRi)
SStotal =  SSTOTi.dot(SSTOTi)
Rsquare = 1 - SSresidual/SStotal
print(Rsquare)





plt.scatter(X, Y)
plt.plot(X, bestFit)
plt.show()

