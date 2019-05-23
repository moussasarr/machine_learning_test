import numpy as np 
import matplotlib.pyplot as plt 

#Plot the sine function. We will be using it for overfitting
N = 100
X = np.linspace(0, 6*np.pi, N)
Y = np.sin(X)

plt.plot(X, Y)
plt.show()

#Make a plynomial of deg from the Input Matrix X
def make_poly(X, deg):
	n = len(X)
	dt= [np.ones(n)]
	for i in range(deg):
		dt.append(X**(i+1))
	dt = np.array(dt).T
	return dt

#Compute W
def fit(X, Y):
	return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

def fit_and_display(X, Y, samples, deg):
	N = len(X)
	train_idx = np.random.choice(N, samples)
	X_train = X[train_idx]
	Y_train = Y[train_idx]


	X_train_poly = make_poly(X_train, deg)
	W = fit(X_train_poly, Y_train)

	X_poly = make_poly(X, deg)
	Yhat = X_poly.dot(W)
	plt.scatter(X_train, Y_train)
	plt.plot(X, Yhat)
	plt.plot(X, Y)
	plt.title("Deg= %d" %deg)
	plt.show()

def plot_train_vs_test_curves(X, Y, sample=20, max_deg=20):
    N = len(X)
    train_idx = np.random.choice(N, sample)
    Xtrain = X[train_idx]
    Ytrain = Y[train_idx]

    test_idx = [idx for idx in range(N) if idx not in train_idx]
    # test_idx = np.random.choice(N, sample)
    Xtest = X[test_idx]
    Ytest = Y[test_idx]

    mse_trains = []
    mse_tests = []
    for deg in range(max_deg+1):
        Xtrain_poly = make_poly(Xtrain, deg)
        w = fit(Xtrain_poly, Ytrain)
        Yhat_train = Xtrain_poly.dot(w)
        mse_train = get_mse(Ytrain, Yhat_train)

        Xtest_poly = make_poly(Xtest, deg)
        Yhat_test = Xtest_poly.dot(w)
        mse_test = get_mse(Ytest, Yhat_test)

        mse_trains.append(mse_train)
        mse_tests.append(mse_test)

    plt.plot(mse_trains, label="train mse")
    plt.plot(mse_tests, label="test mse")
    plt.legend()
    plt.show()

    plt.plot(mse_trains, label="train mse")
    plt.legend()
    plt.show()


for deg in (5, 6, 7, 8, 9):
    fit_and_display(X, Y, 10, deg)

def get_mse(Y, Yhat):
	d = Y - Yhat
	return d.dot(d)/len(d)
	
plot_train_vs_test_curves(X, Y)


