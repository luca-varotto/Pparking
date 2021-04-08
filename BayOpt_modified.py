import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel, PairwiseKernel)
from warnings import catch_warnings
from warnings import simplefilter
from matplotlib import pyplot as plt
import math
from scipy.optimize import minimize_scalar
from tqdm import tqdm

############################################################################################################################
 
# objective function
def objective(x,E_pD,pD_meas):
	return -np.log10( abs(E_pD-pD_meas) ) if abs(E_pD-pD_meas) < 1 else np.log10( abs(E_pD-pD_meas) ) 

# surrogate or approximation for the objective function
def surrogate(model, X):
	# catch any warning generated when making a prediction
	with catch_warnings():
		# ignore generated warnings
		simplefilter("ignore")
		return model.predict(X, return_std=True)
 
# probability of improvement acquisition function
def acquisition(X, Xsamples, model,pf_est,flag):
	# calculate the best surrogate score found so far
	yhat, _ = surrogate(model, X)
	best = max(yhat)
	best_x = X[np.argmax(yhat)]
	# calculate mean and stdev via surrogate function
	mu, std = surrogate(model, Xsamples)
	mu = mu[:, 0]
	# calculate the probability of improvement
	PI = norm.cdf((mu - best) / (std+1E-9))
	# calculate the expected improvement
	EI = (mu - best)*norm.cdf((mu - best)/ (std+1E-9)) \
		  + std*norm.pdf((mu - best)/ (std+1E-9))
	# calculate the UCB
	beta = 1
	gamma = 0 if flag else 1
	ucb = mu-beta*std
	scores = flag*(1-gamma)*ucb - gamma*abs(Xsamples-np.asscalar(pf_est)).reshape(np.shape(ucb))
	return scores,best_x
 
# optimize the acquisition function
def opt_acquisition(X, y, model,pf_est,flag):
	# random search, generate random samples
	Xsamples = np.random.uniform(-np.pi,np.pi,100)
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	# calculate the acquisition function for each sample
	scores,best_x = acquisition(X, Xsamples, model,pf_est,flag)
	# locate the index of the largest scores
	ix = np.argmax(scores)
	return Xsamples[ix, 0],best_x
 
# plot real observations vs surrogate function
def plot(X, y, model):
	# scatter plot of inputs and real objective function
	plt.scatter(X, y,s=2)
	# line plot of surrogate function across domain
	Xsamples = np.asarray(np.arange(min(X), max(X), 0.001))
	Xsamples = Xsamples.reshape(len(Xsamples), 1)
	ysamples, _ = surrogate(model, Xsamples)
	plt.plot(Xsamples, np.maximum(ysamples,0.0))

############################################################################################################################

def test():
	# sample the domain sparsely with noise
	X = np.random.uniform(-np.pi,np.pi,1)
	x = X[0]
	y = np.asarray([objective(x) for x in X])
	# reshape into rows and cols
	X = X.reshape(len(X), 1)
	y = y.reshape(len(y), 1)
	# define the model
	model = GaussianProcessRegressor()
	# fit the model
	model.fit(X, y)
	# perform the optimization process
	for i in tqdm(range(300)):
		# select the next point to sample
		x = opt_acquisition(X, y, model,x)
		# sample the point
		actual = objective(x)
		# summarize the finding
		# est, _ = surrogate(model, [[x]])
		# print('>x=%.3f, f()=%3f, actual=%.3f' % (x, est, actual))
		# add the data to the dataset
		X = np.vstack((X, [[x]]))
		y = np.vstack((y, [[actual]]))
		# update the model
		model.fit(X, y)
	# best result
	ix = np.argmax(y)
	print('Best Result: x=%.3f, y=%.3f' % (X[ix], y[ix]))

	# plot all samples and the final surrogate function
	plot(X, y, model)

	plt.figure()
	plt.plot(X)
	plt.show()

