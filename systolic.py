# Applying Linear Regression in Python
# need to sudo pip install xlrd to use pd.read_excel
# data is from:
# http://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/frames/mlr02.html

# The data (X1, X2, X3) are for each patient.
# X1 = systolic blood pressure
# X2 = age in years
# X3 = weight in pounds
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

dt = pd.read_excel("./data/mlr02.xls")
Y = dt['X1']
#Relationship between age and systolic blood pressure
plt.scatter(dt['X2'], Y)
plt.show()
# we find a more less linear relationship


#Relationship between weight and systolic blood pressure
plt.scatter(dt['X3'], Y)
plt.show()
# a more or less linear relationship

def getR2(X, Y):
	W = np.linalg.inv(X.T.dot(X)).dot(X.T.dot(Y))
	Yhat = X.dot(W)
	d1 = Y - Yhat
	d2 = Y - Y.mean()
	R2 =  1 - (d1.dot(d1))/(d2.dot(d2))
	return R2


dt['ones'] = 1
X2Only = dt[["X2", "ones"]]
X3Only = dt[["X3", "ones"]]
X2X3 = dt[["X2", "X3", "ones"]]

print("R2 of X1Only", getR2(X2Only, Y))
print("R2 of X1Only", getR2(X3Only, Y))
print("R2 of X1Only", getR2(X2X3, Y))

#Conclusion: Combining X1 and X2 gives better R2, So the more variables, the more accurate the prediction






