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
import time

############################################################################################################################

import BayOpt_modified
from kalman_filter import Kalman_filter

############################################################################################################################
        
# underlying parking probability attenuation due to traffic density          
def lambda_true(traffic_density):
    return 1 / (1+np.exp(20*(traffic_density-0.5)) )  
    # return  2/(1+np.exp(4*traffic_density)) # np.sinh( np.arccos(traffic_density)/(np.pi/2) )

# estimated parking probability attenuation due to traffic density
def lambda_est(traffic_density,gp_model=None, return_std=False):
    # start with a simple linear model
    if not gp_model:
        return 1-traffic_density
    # then use the learn GP model
    else:
        return gp_model.predict(traffic_density,return_std=return_std)

# car speed as function of the traffic density
def car_speed(traffic_density_loc):
    v_max = 90 # [km/h]
    return v_max*np.exp(-traffic_density_loc)

# compute expected time of arrival, given the actual car position, the traffic density 
# and the parking probability estimate
# def E_TOA(pos, s, traffic_density, pred):
#     E_TOA = 0
#     idx = np.searchsorted(x,s) 
#     while x[idx] < pos:
#         E_TOA += (segments[np.searchsorted(segments,x[idx])]-segments[np.searchsorted(segments,x[idx])-1]) \
#                 /car_speed(mu[np.searchsorted(segments,x[idx])-1])
#         idx +=1 
#     E_wait = 0
#     for k in range(idx,len(x)):
#         E_wait += (k+1)*pred[k]*np.prod(1-pred[:k])
#     print(E_wait)
#     E_TOA += (x[idx+int(E_wait)]-x[idx])/v_parking
#     idx = np.searchsorted(x,x[idx+int(E_wait)]) 
#     while x[idx] < x[-1]:
#         E_TOA += (segments[np.searchsorted(segments,x[idx])]-segments[np.searchsorted(segments,x[idx])-1]) \
#                 /v_walk
#         idx +=1 
#     return E_TOA

############################################################################################################################

# road length 
L = 100 # [km]

# traffic density as piecewise constant function
nb_segments = 50
x = np.linspace(0, L, nb_segments) # domain
segments = np.array([0])
segments = np.hstack((segments,np.sort(np.random.uniform(0.1,L,size=nb_segments-1))))
segments = np.hstack((segments,L) ) # array defining the spatial segments with constant density
bolleans_segments = list()
mu = np.empty(nb_segments) # density values per each segment
for i in range(1,len(segments)):
    if i==len(segments):
        mu[i-1] = np.random.uniform()
        bolleans_segments.append( (x <= segments[i])*(x >= segments[i-1]) )
    else: 
        mu[i-1] = np.random.uniform()
        bolleans_segments.append( (x < segments[i])*(x >= segments[i-1]) )
traffic_density = np.piecewise(x, bolleans_segments, mu) # traffic density over x

# prior parking prob (given by roads characteristics and presence of parking slots)
prior = [0.5]
# Kalman filter to smooth the prior
x0 = prior[-1] # initial guess
kf = Kalman_filter(1.0,0.0,1.0,0.1,0.1,x0,0.1) # A, B, C, Q, R, P
for i in range(len(x)-1):
    kf.predict()
    prior.append(np.clip(prior[-1] + np.random.uniform(-0.05,0.15),0,1 ))
    kf.update( prior[-1] ) # number of detections / number of frames (proportion of successes)
    prior[-1] = np.clip(kf.x,0,1)
prior = np.array(prior)

# underlying parking prob over x
pP_true = prior* lambda_true(traffic_density)

s = 1.0E-3 # (initial) car position 
s_list=[]
# walking speed 
v_walk = 3 # [km/h]
# parking speed 
v_parking = 10 # [km/h]
# car speed
v = car_speed(mu[np.searchsorted(segments,s)-1]) # [km/h]

fig = plt.figure(figsize=(9,6))
plt.ion()
plt.show()

rmse = [] # prediction rmse
rmse_extensive = []
rmse_IH = [] 
pred_time = [] # predition time
pred_time_IH = [] 
DeltaT = 1/500 # sampling interval
# define the model
kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
    + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e+1))
gp_model = GaussianProcessRegressor(Matern()) # cooperative
gp_model_extensive = GaussianProcessRegressor(Matern()) # non-cooperative
gp_model_IH = GaussianProcessRegressor(kernel=kernel) # cooperative Information-Hungry
while s < L:

    if s!=1.0E-3 and np.random.binomial(1,0.2) :
        idx = np.random.randint(0,len(mu))
        mu[idx] = np.random.uniform()
        # update traffic density over x
        traffic_density = np.piecewise(x, bolleans_segments, mu)
        # update underlying parking probability
        pP_true = prior* lambda_true(traffic_density)
        delete_idx = []
        for i in range(np.shape(X)[0]):
            if X[i] <= segments[idx+1] and X[i] >= segments[idx]:
                delete_idx.append(i)
        X = np.delete(X,delete_idx).reshape(-1,1)
        y = np.delete(y,delete_idx).reshape(-1,1)
        delete_idx = []
        for i in range(np.shape(X_extensive)[0]):
            if X_extensive[i] <= segments[idx+1] and X_extensive[i] >= segments[idx]:
                delete_idx.append(i)
        X_extensive = np.delete(X_extensive,delete_idx).reshape(-1,1)
        y_extensive = np.delete(y_extensive,delete_idx).reshape(-1,1)
        delete_idx = []
        # for i in range(np.shape(X_IH)[0]):
        #     if X_IH[i] <= segments[idx+1] and X_IH[i] >= segments[idx]:
        #         delete_idx.append(i)
        # X_IH = np.delete(X_IH,delete_idx).reshape(-1,1)
        # y_IH = np.delete(y_IH,delete_idx).reshape(-1,1)

    s_list.append(s)

    # where next datum is collected
    sample_pos = s if s==1.0E-3 else next_sample 

    # measurement
    actual = np.asscalar( 
            prior[np.searchsorted(x,sample_pos)-1]* \
            np.clip(lambda_true(mu[np.searchsorted(segments,sample_pos)-1]) +np.random.uniform(-0.1,0.1),0,1 )
        )
    # actual = actual/prior[np.searchsorted(x,sample_pos)-1]

    actual_extensive = np.asscalar( 
        prior[np.searchsorted(x,s)-1]* \
        np.clip(lambda_true(mu[np.searchsorted(segments,s)-1]) +np.random.uniform(-0.1,0.1),0,1 )
        )
    # actual_extensive = actual_extensive/prior[np.searchsorted(x,s)-1]

    # add the data to the dataset
    if s==1.0E-3:
        # X =  np.array([mu[np.searchsorted(segments,sample_pos)-1]]).reshape(1,1)
        X =  np.array([sample_pos]).reshape(1,1)
        y =  np.array([actual]).reshape(1,1)
        # X_extensive =  np.array([mu[np.searchsorted(segments,s)-1]]).reshape(1,1)
        X_extensive = np.array([s]).reshape(1,1)
        y_extensive =  np.array([actual_extensive]).reshape(1,1)
        # X_IH =  np.array([sample_pos]).reshape(1,1)
        # y_IH =  np.array([actual]).reshape(1,1)
    else: 
        try:
            # X = np.vstack((X, [[mu[np.searchsorted(segments,sample_pos)-1]]]))
            X = np.vstack((X, [[sample_pos]]))
        except ValueError:
            # X = np.vstack((X, [[np.asscalar(mu[np.searchsorted(segments,sample_pos)-1])]]))
            X = np.vstack((X, [[np.asscalar(sample_pos)]]))
        y = np.vstack((y, [[actual]]))
        
        try:
            # X_extensive = np.vstack((X_extensive, [[mu[np.searchsorted(segments,s)-1]]]))
            X_extensive = np.vstack((X_extensive, [[s]]))
        except ValueError:
            # X_extensive = np.vstack((X_extensive, [[np.asscalar(mu[np.searchsorted(segments,s)-1])]]))
            X_extensive = np.vstack((X_extensive, [[np.asscalar(s)]]))
        y_extensive = np.vstack((y_extensive, [[actual_extensive]]))

        # max_samples = 1
        # for count, sample in enumerate(Xsamples):
        #     if count > max_samples:
        #         break
        #     sample = sample[0]
        #     try:
        #         X_IH = np.vstack((X_IH, [[sample]]))
        #     except ValueError:
        #         X_IH = np.vstack((X_IH, [[np.asscalar(sample)]]))
        #     actual_IH = np.asscalar( 
        #         prior[np.searchsorted(x,sample)-1]* \
        #     np.clip(lambda_true(mu[np.searchsorted(segments,sample)-1]) +np.random.uniform(-0.1,0.1),0,1 )
        #     )
        #     y_IH = np.vstack((y_IH, [[actual_IH]]))

    start_time = time.time()    
    # fit the model
    gp_model.fit(X, y)
    # predict over x
    # pred,std = lambda_est(traffic_density.reshape(len(traffic_density), 1),gp_model,return_std=True)
    pred,std =  gp_model.predict(x.reshape(len(x),1), return_std=True)
    end_time = time.time()
    pred_time.append(end_time-start_time)
    # store the rmse
    # rmse.append( np.sqrt(np.mean((prior*pred[:,0]-pP_true)**2)))
    rmse.append( np.sqrt(np.mean((pred[:,0]-pP_true)**2)))

    # fit the model
    gp_model_extensive.fit(X_extensive, y_extensive)
    # predict over x
    # pred_extensive,std_extensive = lambda_est(traffic_density.reshape(len(traffic_density), 1),gp_model_extensive,return_std=True)
    pred_extensive,std_extensive = gp_model_extensive.predict(x.reshape(len(x),1),return_std=True) 
    # store the rmse
    # rmse_extensive.append( np.sqrt(np.mean((prior*pred_extensive[:,0]-pP_true)**2)))
    rmse_extensive.append( np.sqrt(np.mean((pred_extensive[:,0]-pP_true)**2)))

    # start_time = time.time()
    # # fit the model
    # gp_model_IH.fit(X_IH, y_IH)
    # # predict over x
    # pred_IH,std_IH = gp_model_IH.predict(x.reshape(len(x),1),return_std=True)
    # end_time = time.time()
    # pred_time_IH.append(end_time-start_time)
    # # store the rmse
    # rmse_IH.append( np.sqrt(np.mean((pred_IH-pP_true)**2)))
    
    # update car position
    v = car_speed(mu[np.searchsorted(segments,s)-1])
    s += v*DeltaT

    # choose next best point where to sample
    # Xsamples = np.sort(np.random.uniform(0,1,10))
    nb_friends = np.random.randint(0,10)
    Xsamples = np.sort(np.random.uniform(0,L,nb_friends))
    if nb_friends > 0:
        Xsamples = Xsamples.reshape(len(Xsamples), 1)
        # calculate mean and stdev via surrogate function
        mu_samples, std_samples = gp_model.predict(Xsamples, return_std=True)
        scores = std_samples	
        # locate the index of the largest scores
        # idx_next = ( np.abs(traffic_density - Xsamples[np.argmax(scores), 0])).argmin()
        # next_sample = x[idx_next]
        next_sample = Xsamples[ np.argmax(scores),0] 
    else: 
        next_sample = s

    # visualization
    sub1=plt.subplot(3,1,1)
    plt.scatter(s,mu[min(np.searchsorted(segments,s)-1,len(mu)-1)])
    plt.scatter(next_sample,mu[min(np.searchsorted(segments,next_sample)-1,len(mu)-1)])
    plt.plot(x,traffic_density)
    plt.xlabel(r'$x$')
    plt.ylabel(r'$\mu(x,t)$')
    plt.title(r'$v= $' +'%4.2f' %(v))
    sub2=plt.subplot(3,1,2)
    plt.plot(s_list,100*np.asarray(rmse)/rmse[0],color='k',label='C')
    plt.plot(s_list,100*np.asarray(rmse_extensive)/rmse_extensive[0],color='b',label='NC')
    # plt.plot(s_list,100*np.asarray(rmse_IH)/rmse_IH[0],color='g',label='IH')
    plt.xlabel(r'$x$')
    plt.ylabel(r'$rmse_x/rmse_0 \; \%$')
    plt.title('Learning curve')
    plt.xlim([0,x[-1]])
    plt.ylim([0,None])
    plt.legend()
    sub3 = plt.subplot(3,1,3)
    # dens = np.linspace(0.0,1.0,100)
    # plt.plot(dens,lambda_true(dens),label=r'$p_P(\mu)$',color='r')
    # pP_est, std = lambda_est(dens.reshape(len(dens), 1),gp_model,return_std=True)
    # plt.plot(dens,pP_est,label=r'$\hat{p}_P(\mu)$',color='k')
    # plt.plot(dens,lambda_est(dens.reshape(len(dens), 1),gp_model_extensive),color='b')
    plt.plot(x,pP_true,color='r', linewidth=2,label=r'$p_P(x)$')
    plt.plot(x,prior,color='r', linestyle='--', linewidth=0.5,label=r'$\pi(x)$')
    plt.plot(x,pred,color='k',label=r'$\hat{p}_P^{C}(x)$')
    plt.plot(x,pred_extensive,color='b',label=r'$\hat{p}_P^{NC}(x)$')
    # plt.plot(x,pred_IH,color='g',label=r'$\hat{p}_P^{IH}(x)$')
    plt.ylim([-0.5,1.05])
    plt.fill_between(np.squeeze(x),\
        np.squeeze(pred) - std,\
        np.squeeze(pred) + std,\
        alpha=0.1, facecolor='k')
    plt.fill_between(np.squeeze(x),\
        np.squeeze(pred_extensive) - std_extensive,\
        np.squeeze(pred_extensive) + std_extensive,\
        alpha=0.1, facecolor='b')
    plt.legend()
    # plt.scatter(X,y, alpha=0.5,marker='o',s=10, c='k') # observations
    # sub4 = plt.subplot(4,1,4)
    # plt.plot(Xsamples, std_samples)
    # plt.plot(s_list, np.asarray(pred_time)/np.asarray(pred_time_IH),label='time')
    # plt.plot(s_list, np.asarray(rmse)/np.asarray(rmse_IH),label='rmse')
    # plt.legend()
    # plt.xlim([0,x[-1]])
    plt.tight_layout()

    if plt.waitforbuttonpress(0.1):
        break 
    plt.pause(0.001)
    sub1.cla()
    sub2.cla()
    sub3.cla()
    # sub4.cla()

plt.ioff()
plt.close()

# # plot final prediction
# plt.figure()
# plt.plot(x, pP_true,label=r'$p_P(x)$',color='b')
# plt.plot(x, prior*lambda_est(traffic_density.reshape(len(traffic_density), 1),gp_model)[:,0],label=r'$\hat{p}_P(x)$',color='k')
# plt.ylim([0,1])
# plt.legend()
# plt.show()

# # plot underlyng and learnt attenuation function
# plt.figure()
# dens = np.linspace(0.0,1.0,100)
# plt.plot(dens,lambda_true(dens),label=r'$p_P(\mu)$',color='r')
# plt.plot(dens, lambda_est(dens.reshape(len(dens), 1),gp_model),label=r'$\hat{p}_P(\mu)$',color='k')
# plt.plot(dens, lambda_est(dens.reshape(len(dens), 1),gp_model_extensive),color='b')
# plt.legend()
# plt.show()

# # plot learning improvement
# plt.figure() 
# plt.plot(rmse,color='k')
# plt.plot(rmse_extensive,color='b')
# plt.show()
