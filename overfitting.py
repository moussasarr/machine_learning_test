import numpy as np
import matplotlib.pyplot as plt


# Plotting the sin(x) function
N = 100
X = np.linspace(0, 6*np.pi, N)
Y = np.sin(X)

plt.plot(X, Y)
plt.show()