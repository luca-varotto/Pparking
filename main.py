import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (RBF, Matern, RationalQuadratic,
                                              ExpSineSquared, DotProduct,
                                              ConstantKernel, WhiteKernel, PairwiseKernel)
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
############################################################################################################################



############################################################################################################################

def p_park_est(traffic_density,gp_model=None):
    # start with a simple linear model
    if not gp_model:
        return 1-traffic_density
    # then use the learn GP model
    else:
        return gp_model.predict(traffic_density,return_std=False)

def p_park_true(traffic_density):
    return np.arccos(traffic_density)/(np.pi/2)

def car_speed(traffic_density_loc):
    v_max = 90 # [km/h]
    return v_max*np.exp(-traffic_density_loc)

############################################################################################################################

# road length 
L = 20 # [km]
x = np.linspace(0, L, 200) # domain

# traffic density as piecewise constant function
nb_segments = 40
segments = np.array([0])
segments = np.hstack((segments,np.sort(np.random.uniform(0,L,size=nb_segments-2))))
segments = np.hstack((segments,L) )# np.empty(nb_segments+1) # array defining the segments with constant density
bolleans_segments = list()
mu = np.empty(nb_segments) # array providing density values
for i in range(len(segments)):
    # if i ==0:
    #     segments[i] = 0
    if i==len(segments)-1:
        # segments[i] = L
        mu[i-1] = np.random.uniform()
        bolleans_segments.append( (x <= segments[i])*(x >= segments[i-1]) )
    else: 
        # segments[i] = np.random.uniform(segments[i-1],L)
        mu[i-1] = np.random.uniform()
        bolleans_segments.append( (x < segments[i])*(x >= segments[i-1]) )
traffic_density = np.piecewise(x, bolleans_segments, mu)

pP_true = p_park_true(traffic_density)

# car position 
s = 0 

# walking speed 
v_walk = 3 # [km/h]
# car speed
v = car_speed(mu[np.searchsorted(segments,s)]) # [km/h]

# prior
# define the model
gp_model = GaussianProcessRegressor()# Matern() + WhiteKernel()
X = traffic_density
y = p_park_est(traffic_density)
X = X.reshape(len(X),1)
y = y.reshape(len(y),1)
gp_model.fit(X,y)

fig = plt.figure(figsize=(9,6))
plt.ion()
plt.show()

# simulation
rmse = []
DeltaT = 1/500
s_list=[]
while s < L:

    s_list.append(s)
    actual = np.asscalar( 
        p_park_true(mu[np.searchsorted(segments,s)]) +np.random.uniform(0,0.1)
        )
    # add the data to the dataset
    try:
        X = np.vstack((X, [[mu[np.searchsorted(segments,s)]]]))
    except ValueError:
        X = np.vstack((X, [[np.asscalar(mu[np.searchsorted(segments,s)])]]))
    y = np.vstack((y, [[actual]]))
    # fit the model
    gp_model.fit(X, y)
    pred = p_park_est(traffic_density.reshape(len(traffic_density), 1),gp_model)[:,0]
    rmse.append( np.sqrt(np.mean((pred-pP_true)**2)))
    v = car_speed(mu[np.searchsorted(segments,s)])
    s += v*DeltaT

    sub1=plt.subplot(2,1,1)
    plt.scatter(s,mu[min(np.searchsorted(segments,s),len(mu)-1)])
    plt.plot(x,traffic_density)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu(x)$')
    plt.title(r'$v= $' +'%4.2f' %(v))
    sub2=plt.subplot(2,1,2)
    plt.plot(s_list,100*np.asarray(rmse)/rmse[0])
    plt.xlabel(r'$x$')
    plt.ylabel(r'$rmse_x/rmse_0 \; \%$')
    plt.title('Learning curve')
    plt.xlim([0,x[-1]])
    plt.ylim([0,None])
    plt.tight_layout()

    if plt.waitforbuttonpress(0.1):
        break 
    plt.pause(0.001)
    sub1.cla()
    sub2.cla()
plt.ioff()
plt.close()

s = L-1.0E-3
for _ in range(50):
    actual = np.asscalar( 
        p_park_true(mu[np.searchsorted(segments,s)]) +np.random.uniform(0,0.1)
        )
    # add the data to the dataset
    try:
        X = np.vstack((X, [[mu[np.searchsorted(segments,s)]]]))
    except ValueError:
        X = np.vstack((X, [[np.asscalar(mu[np.searchsorted(segments,s)])]]))
    y = np.vstack((y, [[actual]]))
    # fit the model
    gp_model.fit(X, y)
    pred = p_park_est(traffic_density.reshape(len(traffic_density), 1),gp_model)[:,0]
    rmse.append( np.sqrt(np.mean((pred-pP_true)**2)))

# plot final prediction
plt.figure()
# plt.plot(x, traffic_density,label=r'$\mu(x)$',color='r')
# plt.plot(x, p_park_est(traffic_density),label=r'$\hat{p}_{P,0}(x)$',color='g')
plt.plot(x, pP_true,label=r'$p_P(x)$',color='b')
plt.plot(x, p_park_est(traffic_density.reshape(len(traffic_density), 1),gp_model),label=r'$\hat{p}_P(x)$',color='k')
plt.legend()
plt.show()

# plot learnt function
plt.figure()
dens = np.linspace(0.0,1.0,100)
plt.plot(dens,p_park_true(dens),label=r'$p_P(\mu)$',color='r')
plt.plot(dens, p_park_est(dens.reshape(len(dens), 1),gp_model),label=r'$\hat{p}_P(\mu)$',color='k')
plt.legend()
plt.show()

# plot learning improvement
plt.figure() 
plt.plot(rmse)
plt.show()
