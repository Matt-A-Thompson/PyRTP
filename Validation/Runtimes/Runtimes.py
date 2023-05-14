#%%
import numpy as np

CN=np.load('CNrun.npy')
EM=np.load('EMrun.npy')
ETRS=np.load('ETRSrun.npy')
AETRS=np.load('AETRSrun.npy')
CAETRS=np.load('CAETRSrun.npy')
CFM4=np.load('CFM4run.npy')

CNmean=np.mean(CN)
EMmean=np.mean(EM)
ETRSmean=np.mean(ETRS)
AETRSmean=np.mean(AETRS)
CAETRSmean=np.mean(CAETRS)
CFM4mean=np.mean(CFM4)

print(CNmean)
print(EMmean)
print(ETRSmean)
print(AETRSmean)
print(CAETRSmean)
print(CFM4mean)
# %%
