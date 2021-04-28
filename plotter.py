import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
rcParams['text.usetex'] = True
from statsmodels.distributions.empirical_distribution import ECDF
import pickle

############################################################################################################################

# road length 
L = 10 # [m]
# sampling interval of the platform
DeltaT = 1/500 
# constant speed of the car
v_const = 50 # [km/h]

with open('./data/performance.pkl', 'rb') as f:
    performance = pickle.load(f)
 
with open('./data/rmse_MC_timeVariant.pkl', 'rb') as f:
    rmse_MC = pickle.load(f)
 
x = np.linspace(1,int(L/(v_const*DeltaT))*DeltaT*3600,int(L/(v_const*DeltaT)))
plt.figure(figsize=(9,6))
plt.plot(x,np.mean(performance[0,:,:],axis=0),color='m', linewidth=2)
plt.plot(x,np.ones(len(x)), linestyle=':')
plt.fill_between(x,\
    np.mean(performance[0,:,:],axis=0) - np.std(performance[0,:,:],axis=0),\
    np.mean(performance[0,:,:],axis=0) + np.std(performance[0,:,:],axis=0),\
    alpha=0.1,facecolor='k')
plt.ylabel(r'$\tau_t$',fontsize=35)  
plt.xlabel(r'$t\;[s]$',fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.tight_layout()
plt.show()


# rmse_MC = np.delete(rmse_MC,8,1)
x = np.linspace(1,int(L/(v_const*DeltaT))*DeltaT*3600,int(L/(v_const*DeltaT)))
plt.figure(figsize=(9,6))
plt.plot(x,np.mean(rmse_MC[0,:,:],axis=0),color='k', linewidth=2, label='proposed')
plt.fill_between(x,\
    np.mean(rmse_MC[0,:,:],axis=0) - np.std(rmse_MC[0,:,:],axis=0),\
    np.mean(rmse_MC[0,:,:],axis=0) + np.std(rmse_MC[0,:,:],axis=0),\
    alpha=0.1,facecolor='k')
plt.plot(x,np.mean(rmse_MC[1,:,:],axis=0),color='g', linewidth=2, label='Rnd')
plt.fill_between(x,\
    np.mean(rmse_MC[1,:,:],axis=0) - np.std(rmse_MC[1,:,:],axis=0),\
    np.mean(rmse_MC[1,:,:],axis=0) + np.std(rmse_MC[1,:,:],axis=0),\
    alpha=0.1,facecolor='g')
plt.plot(x,np.mean(rmse_MC[2,:,:],axis=0),color='b', linewidth=2, label='noCom')
plt.fill_between(x,\
    np.mean(rmse_MC[2,:,:],axis=0) - np.std(rmse_MC[2,:,:],axis=0),\
    np.mean(rmse_MC[2,:,:],axis=0) + np.std(rmse_MC[2,:,:],axis=0),\
    alpha=0.1,facecolor='b')    
plt.ylabel(r'$RMSE_t/RMSE_0$',fontsize=35)  
plt.xlabel(r'$t\;[s]$',fontsize=35)
ax = plt.gca()
ax.patch.set_edgecolor('black')  
ax.patch.set_linewidth('2')
ax.grid(ls = ':', lw = 0.5)
plt.legend(fontsize=30,framealpha=0.5,ncol=2)
plt.yticks(fontsize=35)
plt.xticks(fontsize=35)
plt.tight_layout()
plt.show()

# plt.figure()
# for i in range(np.shape(rmse_MC[0,:,:])[0]):
#     plt.plot(rmse_MC[0,i,:], label=str(i))
# plt.legend()
# plt.show()

