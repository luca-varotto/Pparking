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
from statsmodels.distributions.empirical_distribution import ECDF
import pickle

############################################################################################################################

from kalman_filter import Kalman_filter

############################################################################################################################
        
# true parking probability attenuation due to traffic density          
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

# vehicle speed as function of the traffic density
def car_speed(traffic_density_loc):
    v_max = 90 # [km/h]
    return v_max*np.exp(-traffic_density_loc)

# compute the a-priori parking availability according to the paper definition
def prior_compute(x, W=100, D=5):
    pass
    # return prior

############################################################################################################################

# MC experiments parameters
nb_MC_tests = 1

# road length 
L = 10 # [km]
# sampling interval of the platform
DeltaT = 1/500 
# constant speed of the car
v_const = 50

# performance for the MC simulation
performance = np.empty((3,nb_MC_tests,int(L/(v_const*DeltaT))))
rmse_MC = np.empty((3,nb_MC_tests,int(L/(v_const*DeltaT))))

for MC_test in range(nb_MC_tests):

    # traffic density as piecewise constant function
    nb_segments = 5 # number of segments where the density is constant
    pts_per_segment = 50 # number of evaluation points per segment
    x = np.linspace(0, L, nb_segments*pts_per_segment) # space domain
    segments = np.linspace(0, L, nb_segments+1) # array defining the spatial segments with constant density
    mu = np.empty(nb_segments) # density values per each segment
    boolean_segments = list()
    for i in range(1,len(segments)):
        mu[i-1] = np.random.uniform()
        if i==len(segments):
            boolean_segments.append( (x <= segments[i])*(x >= segments[i-1]) )
        else: 
            boolean_segments.append( (x < segments[i])*(x >= segments[i-1]) )
    traffic_density = np.piecewise(x, boolean_segments, mu) # traffic density over x
    p_change = 0.2 # changing probability 

    # prior parking prob. (given by roads structural characteristics and presence of parking slots)
    prior = [0.5]
    # Kalman filter to smooth the prior
    x0 = prior[-1] # initial guess
    kf = Kalman_filter(1.0,0.0,1.0,0.01,0.1,x0,0.1) # A, B, C, Q, R, P
    for i in range(len(x)-1):
        kf.predict()
        prior.append(np.clip(prior[-1] + np.random.uniform(-0.15,0.15),0,1 )) # prior value at x[i+1]
        kf.update( prior[-1] ) 
        prior[-1] = np.clip(kf.x,0,1) # smooth
    prior = np.array(prior)

    # true parking prob
    pP_true = prior* lambda_true(traffic_density)

    # vehicle
    s = 1.0E-3 # (initial) vehicle position 
    s_list=[] # list of vehicle positions
    # vehicle speed (according to traffic conditions)
    v = v_const# car_speed(mu[np.searchsorted(segments,s)-1]) # [km/h]
    meas_noise = 0.1 # measurement noise (std. dev. of the additive Gaussian noise)
    max_nb_vehicles = 10 # maximum number of connected vehicles 
    max_samples = min(max_nb_vehicles,1) # maximum number of vehicles considered at a time 

    # GP models
    kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3)) \
        + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))
    gp_model = GaussianProcessRegressor(kernel=kernel) # cooperative selective
    gp_model_extensive = GaussianProcessRegressor(kernel=Matern()) # non-cooperative
    gp_model_nonSel = GaussianProcessRegressor(kernel=kernel) # cooperative non-selective

    # visualization
    if nb_MC_tests==1:
        fig = plt.figure(figsize=(15,5))
        plt.ion()
        plt.show()
    # prediction rmse buffers
    rmse = [] 
    rmse_extensive = []
    rmse_nonSel = [] 
    # predition time buffers
    pred_time = [] 
    pred_time_nonSel = [] 
    # while s < L:
    for t in tqdm(range(int(L/(v_const*DeltaT)))):

        # *** CHANGE TRAFFIC DENSITY (thus also Pparking)
        if s!=1.0E-3 and np.random.binomial(1,p_change) :
            idx = np.random.randint(0,len(mu)) # randomly choose a road segment
            mu[idx] = np.random.uniform() # randomly modify the segment density
            traffic_density = np.piecewise(x, boolean_segments, mu) # update traffic density function over x
            pP_true = prior* lambda_true(traffic_density) # update underlying parking probability
            # remove obsolete values from the dataset (X,y)
            # cooperative selective
            delete_idx = []
            for i in range(np.shape(X)[0]):
                if X[i] <= segments[idx+1] and X[i] >= segments[idx]:
                    delete_idx.append(i)
            X = np.delete(X,delete_idx).reshape(-1,1)
            y = np.delete(y,delete_idx).reshape(-1,1)
            # non-cooperative
            delete_idx = []
            for i in range(np.shape(X_extensive)[0]):
                if X_extensive[i] <= segments[idx+1] and X_extensive[i] >= segments[idx]:
                    delete_idx.append(i)
            X_extensive = np.delete(X_extensive,delete_idx).reshape(-1,1)
            y_extensive = np.delete(y_extensive,delete_idx).reshape(-1,1)
            # cooperative non-selective
            delete_idx = []
            for i in range(np.shape(X_nonSel)[0]):
                if X_nonSel[i] <= segments[idx+1] and X_nonSel[i] >= segments[idx]:
                    delete_idx.append(i)
            X_nonSel = np.delete(X_nonSel,delete_idx).reshape(-1,1)
            y_nonSel = np.delete(y_nonSel,delete_idx).reshape(-1,1)

        # *** DATA COLLECTION
        # next input (i.e., where measurement is collected)
        s_list.append(s) # save current collection place
        sample_pos = s if s==1.0E-3 else next_sample # update collection place 

        # measurement
        # cooperarive selective
        actual = np.asscalar( 
                np.clip(
                    pP_true[np.searchsorted(x,sample_pos)-1]\
                    # prior[np.searchsorted(x,sample_pos)-1]*lambda_true(mu[np.searchsorted(segments,sample_pos)-1])\
                    + np.random.uniform(-meas_noise,meas_noise),0,1 ))
        # non-cooperative
        actual_extensive = np.asscalar( 
            np.clip(
                    pP_true[np.searchsorted(x,s)-1]\
                    + np.random.uniform(-meas_noise, meas_noise),0,1 ))

        # *** DATASET UPDATE
        # update the dataset (X,y)
        if s==1.0E-3: # first update
            # cooperative selective
            X =  np.array([sample_pos]).reshape(1,1)
            y =  np.array([actual]).reshape(1,1)
            # non-cooperative
            X_extensive = np.array([s]).reshape(1,1)
            y_extensive =  np.array([actual_extensive]).reshape(1,1)
            # cooperative non-selective
            X_nonSel =  np.array([sample_pos]).reshape(1,1)
            y_nonSel =  np.array([actual]).reshape(1,1)
        else: # successive updates
            # cooperative selective
            try:
                X = np.vstack((X, [[sample_pos]]))
            except ValueError:
                X = np.vstack((X, [[np.asscalar(sample_pos)]]))
            y = np.vstack((y, [[actual]]))
            # non-cooperative
            try:
                X_extensive = np.vstack((X_extensive, [[s]]))
            except ValueError:
                X_extensive = np.vstack((X_extensive, [[np.asscalar(s)]]))
            y_extensive = np.vstack((y_extensive, [[actual_extensive]]))
            # cooperative non-selective
            if len(Xsamples)>0:
                np.random.shuffle(Xsamples)  
            else:
                Xsamples = np.array([[next_sample]])
            for count, sample in enumerate(Xsamples):
                if count >= max_samples:
                    break
                sample = sample[0]
                try:
                    X_nonSel = np.vstack((X_nonSel, [[sample]]))
                except ValueError:
                    X_nonSel = np.vstack((X_nonSel, [[np.asscalar(sample)]]))
                actual_nonSel = np.asscalar( 
                                np.clip(
                                    pP_true[np.searchsorted(x,sample)-1]\
                                    + np.random.uniform(-meas_noise, meas_noise),0,1 ))
                y_nonSel = np.vstack((y_nonSel, [[actual_nonSel]]))
        # *** MODEL TRAINING
        # cooperative selective
        start_time = time.time()    
        # fit the model
        gp_model.fit(X, y)
        # prediction
        pred,std =  gp_model.predict(x.reshape(len(x),1), return_std=True)
        end_time = time.time()        
        # predition time
        pred_time.append(end_time-start_time)
        # rmse
        rmse.append( np.sqrt(np.mean((pred[:,0]-pP_true)**2)))

        # non-cooperative
        # fit the model
        gp_model_extensive.fit(X_extensive, y_extensive)
        # prediction
        pred_extensive,std_extensive = gp_model_extensive.predict(x.reshape(len(x),1),return_std=True) 
        # rmse
        rmse_extensive.append( np.sqrt(np.mean((pred_extensive[:,0]-pP_true)**2)) )

        # cooperative non-selective
        start_time = time.time()
        # fit the model
        gp_model_nonSel.fit(X_nonSel, y_nonSel)
        # prediction over x
        pred_nonSel,std_nonSel = gp_model_nonSel.predict(x.reshape(len(x),1),return_std=True)
        end_time = time.time()
        # prediction time
        pred_time_nonSel.append(end_time-start_time)
        # rmse
        rmse_nonSel.append( np.sqrt(np.mean((pred_nonSel[:,0]-pP_true)**2)))
        
        # update vehicle position (according to traffic density)
        v = v_const # car_speed(mu[np.searchsorted(segments,s)-1])
        s += v*DeltaT

        # *** VEHICLES COMMUNICATION
        nb_vehicles = max_nb_vehicles # np.random.randint(0,max_nb_vehicles) # (time-varying) number of connected vehicles
        Xsamples = np.sort(np.random.uniform(0,L,nb_vehicles)) # vehicles positions
        if nb_vehicles > 0:
            Xsamples = Xsamples.reshape(len(Xsamples), 1)
            # calculate mean and std dev of current model (surrogate function)
            mu_samples, std_samples = gp_model.predict(Xsamples, return_std=True)
            # score vehicles locations according to the uncertainty level of the model
            scores = std_samples	
            # locate the index of the largest scores
            next_sample = Xsamples[ np.argmax(scores),0]
        else: 
            next_sample = s

        # *** ONLINE VISUALIZATION
        if nb_MC_tests==1:
            # traffic density
            # sub1=plt.subplot(4,1,1)
            # plt.scatter(s,mu[min(np.searchsorted(segments,s)-1,len(mu)-1)])
            # plt.scatter(next_sample,mu[min(np.searchsorted(segments,next_sample)-1,len(mu)-1)])
            # plt.plot(x,traffic_density)
            # plt.xlabel(r'$x$')
            # plt.ylabel(r'$\mu(x,t)$')
            # plt.title(r'$v= $' +'%4.2f' %(v))
            # Learning curve
            # sub2=plt.subplot(3,1,1)
            # plt.plot(s_list,100*np.asarray(rmse)/rmse[0],color='k',label='C')
            # plt.plot(s_list,100*np.asarray(rmse_nonSel)/rmse_nonSel[0],color='g',label='IH')
            # plt.plot(s_list,100*np.asarray(rmse_extensive)/rmse_extensive[0],color='b',label='NC')
            # plt.xlabel(r'$x$')
            # plt.ylabel(r'$rmse_x/rmse_0 \; \%$')
            # plt.title('Learning curve')
            # plt.xlim([0,x[-1]])
            # plt.ylim([0,None])
            # plt.legend()
            # Pparking
            sub3 = plt.subplot(1,1,1)
            plt.plot(x,pP_true,color='r', linewidth=2,label=r'$f(x)$')
            plt.plot(x,prior,color='r', linestyle='--', linewidth=0.5,label=r'$\pi(x)$')
            plt.plot(x,pred,color='k',label='proposed')
            plt.plot(x,pred_nonSel,color='g',label='Rnd')
            plt.plot(x,pred_extensive,color='b',label='noCom')
            plt.ylim([-0.1,2])
            plt.fill_between(np.squeeze(x),\
                np.squeeze(pred) - std,\
                np.squeeze(pred) + std,\
                alpha=0.1, facecolor='k') 
            plt.xlabel(r'$x\;[km]$',fontsize=35)
            ax = plt.gca()
            ax.patch.set_edgecolor('black')  
            ax.patch.set_linewidth('2')
            ax.grid(ls = ':', lw = 0.5)
            plt.yticks(fontsize=35)
            plt.xticks(fontsize=35)
            plt.tight_layout()
            plt.legend(fontsize=30,framealpha=0.5,ncol=3)
            plt.show()
            # time prediction
            # sub4 = plt.subplot(3,1,3)
            # plt.plot(np.ones(1000), linestyle='--')
            # plt.plot(s_list, (np.asarray(pred_time)/pred_time[0])/(np.asarray(pred_time_nonSel)/pred_time_nonSel[0]),label=r'$T_{sel}/T_{NonSel}$',c='m')
            # plt.plot(s_list, (np.asarray(rmse)/rmse[0])/(np.asarray(rmse_nonSel)/rmse_nonSel[0]),label=r'$rmse_{sel}/rmse_{NonSel}$',c='g')
            # plt.plot(s_list, (np.asarray(rmse)/rmse[0])/(np.asarray(rmse_extensive)/rmse_extensive[0]),label=r'$rmse_{sel}/rmse_{extensive}$',c='b')
            # plt.legend()
            # plt.xlim([0,x[-1]])
            # plt.tight_layout()

            if plt.waitforbuttonpress(0.1):
                break 
            plt.pause(0.001)
            # sub1.cla()
            # sub2.cla()
            sub3.cla()
            # sub4.cla()

    rmse_MC[0,MC_test,:] = 100*np.asarray(rmse)/rmse[0]
    rmse_MC[1,MC_test,:] = 100*np.asarray(rmse_nonSel)/rmse_nonSel[0]
    rmse_MC[2,MC_test,:] = 100*np.asarray(rmse_extensive)/rmse_extensive[0]
    performance[0,MC_test,:] = (np.asarray(pred_time)/pred_time[0])/(np.asarray(pred_time_nonSel)/pred_time_nonSel[0])
    performance[1,MC_test,:] = (np.asarray(rmse)/rmse[0])/(np.asarray(rmse_nonSel)/rmse_nonSel[0])
    performance[2,MC_test,:] = (np.asarray(rmse)/rmse[0])/(np.asarray(rmse_extensive)/rmse_extensive[0])

plt.ioff()
plt.close()

# *** OFFLINE VISUALIZATION
# plot final prediction
# plt.figure()
# plt.plot(x, pP_true,label=r'$p_P(x)$',color='b')
# plt.plot(x, prior*lambda_est(traffic_density.reshape(len(traffic_density), 1),gp_model)[:,0],label=r'$\hat{p}_P(x)$',color='k')
# plt.ylim([0,1])
# plt.legend()
# plt.show()

fig =plt.figure(figsize=(9,6))
plt.plot(np.mean(performance[0,:,:],axis=0),color='m', linewidth=2)
plt.fill_between(range(int(L/(v_const*DeltaT))),\
    np.mean(performance[0,:,:],axis=0) - np.std(performance[0,:,:],axis=0),\
    np.mean(performance[0,:,:],axis=0) + np.std(performance[0,:,:],axis=0),\
    alpha=0.1,facecolor='k')
# ecdf = ECDF(performance[0,:,:].reshape(-1))
# plt.plot(ecdf.x, ecdf.y, \
#         linewidth=2,color='m')
plt.xlabel(r"$...$",fontsize=35)
plt.ylabel(r"$...$",fontsize=35)
plt.legend(fontsize=30)
plt.xticks(fontsize=35)
plt.yticks(fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(np.mean(performance[1,:,:],axis=0),color='g', linewidth=2)
plt.fill_between(range(int(L/(v_const*DeltaT))),\
    np.mean(performance[1,:,:],axis=0) - np.std(performance[1,:,:],axis=0),\
    np.mean(performance[1,:,:],axis=0) + np.std(performance[1,:,:],axis=0),\
    alpha=0.1,facecolor='g')
plt.plot(np.mean(performance[2,:,:],axis=0),color='b', linewidth=2)
plt.fill_between(range(int(L/(v_const*DeltaT))),\
    np.mean(performance[2,:,:],axis=0) - np.std(performance[2,:,:],axis=0),\
    np.mean(performance[2,:,:],axis=0) + np.std(performance[2,:,:],axis=0),\
    alpha=0.1,facecolor='b')
plt.show()

plt.figure()
plt.plot(np.mean(rmse_MC[0,:,:],axis=0),color='k', linewidth=2)
plt.fill_between(range(int(L/(v_const*DeltaT))),\
    np.mean(rmse_MC[0,:,:],axis=0) - np.std(rmse_MC[0,:,:],axis=0),\
    np.mean(rmse_MC[0,:,:],axis=0) + np.std(rmse_MC[0,:,:],axis=0),\
    alpha=0.1,facecolor='k')
plt.plot(np.mean(rmse_MC[1,:,:],axis=0),color='g', linewidth=2)
plt.fill_between(range(int(L/(v_const*DeltaT))),\
    np.mean(rmse_MC[1,:,:],axis=0) - np.std(rmse_MC[1,:,:],axis=0),\
    np.mean(rmse_MC[1,:,:],axis=0) + np.std(rmse_MC[1,:,:],axis=0),\
    alpha=0.1,facecolor='g')
plt.plot(np.mean(rmse_MC[2,:,:],axis=0),color='b', linewidth=2)
plt.fill_between(range(int(L/(v_const*DeltaT))),\
    np.mean(rmse_MC[2,:,:],axis=0) - np.std(rmse_MC[2,:,:],axis=0),\
    np.mean(rmse_MC[2,:,:],axis=0) + np.std(rmse_MC[2,:,:],axis=0),\
    alpha=0.1,facecolor='b')    
plt.show()

# fig =plt.figure(figsize=(9,6))
# ecdf = ECDF(performance[1,:,:].reshape(-1))
# plt.plot(ecdf.x, ecdf.y, \
#         linewidth=2,color='g')
# ecdf = ECDF(performance[2,:,:].reshape(-1))
# plt.plot(ecdf.x, ecdf.y, \
#         linewidth=2,color='b')
# plt.xlabel(r"$...$",fontsize=35)
# plt.ylabel(r"$...$",fontsize=35)
# plt.legend(fontsize=30)
# plt.xticks(fontsize=35)
# plt.yticks(fontsize=35)
# ax = plt.gca()
# ax.patch.set_edgecolor('black')  
# ax.patch.set_linewidth('2')
# ax.grid(ls = ':', lw = 0.5)
# plt.tight_layout()
# plt.show()


# savings
with open('./data/performance.pkl', 'wb') as f:
    pickle.dump(performance, f, pickle.HIGHEST_PROTOCOL)
with open('./data/rmse_MC.pkl', 'wb') as f:
    pickle.dump(rmse_MC, f, pickle.HIGHEST_PROTOCOL)