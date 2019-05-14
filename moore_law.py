import re
import numpy as np
import matplotlib.pyplot as plt

X = []
Y = []

# some numbers show up as 1,170,000,000 (commas)
# some numbers have references in square brackets after them
non_decimal = re.compile(r'[^\d]+')

for line in open('./data/moore.csv'):
    r = line.split('\t')

    x = int(non_decimal.sub('', r[2].split('[')[0]))
    y = int(non_decimal.sub('', r[1].split('[')[0]))
    X.append(x)
    Y.append(y)

X = np.array(X)
Y = np.array(Y)

plt.scatter(X, Y)
plt.show()


Y = np.log(Y)
plt.scatter(X, Y)
plt.show()

denominator = X.dot(X) - X.mean()*(X.sum())
a = (Y.dot(X) - Y.mean()*X.sum())/denominator
b = (Y.mean()*(X.dot(X)) - X.mean()*(Y.dot(X)))/denominator

bestFit = a*X + b
plt.scatter(X, Y)
plt.plot(X, bestFit)

SSres = (Y - bestFit).dot(Y - bestFit)
SStot = (Y - Y.mean()).dot(Y - Y.mean())
R2 = 1 - (SSres/SStot)
print("a:", a, "b:", b)
print("the r-squared is:", R2)
plt.show()

#How long does it take for the transistor count to double ?
# log(tc) = a*year + b
#Taking exponentials at both sides
# tc =  exp(a*year + b)
# tc = exp(a*year)*exp(b)
# 2*tc = 2*exp(a*year)*exp(b)
# 2tc = exp(ln2)*exp(b)*exp(a*year)
# 2tc = exp(b)*exp(ln2 + a*year)
# log(2tc) = b + ln2 + a*year
# ln2 + a*t1 = a*t2.
# a(t2 - t1) = ln2
# t2 - t1 = ln2/a
print("time it takes to double: ", np.log(2)/a)








