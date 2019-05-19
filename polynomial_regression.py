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


	print(X)

