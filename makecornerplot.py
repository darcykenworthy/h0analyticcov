#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np

import matplotlib.pyplot as plt
from scipy import optimize as op,linalg,stats

import re,timeit,pickle,argparse,multiprocessing,os,tqdm,corner
import matplotlib.ticker as mtick
from os import path


# In[2]:


parser = argparse.ArgumentParser(description='Calculate predicted peculiar velocity correction and covariance between supernovae from results of MCMC ')
parser.add_argument('extractedparspickle',type=str, 
                    help='File with MCMC chain')
parser.add_argument('destoutput',type=str, 
                    help='Where to write corner plot')
args = parser.parse_args()
picklefile=args.extractedparspickle
output=args.destoutput 

with open(picklefile ,'rb') as file: results=pickle.load(file)
pars=results['pars']
vars= [x for x in ['velscaling','intrins','correctionstd','veldispersion_additional','betarescale'] if x in results['pars']]#'offset','intrins',
labels={'offset':'$\Delta_\mu$','correctionstd':'$\sigma_{2M++}$','intrins':'$\sigma_{int}$','velscaling':'$A_s$','veldispersion_additional':'$\\sigma_{v+}$','betarescale':'$S_v$'}
samples=np.array([results['pars'][par] for par in vars]).T
samples[:,0]-=0.06380
print('Medians :'+', '.join([f"{x}: {np.median(samples[:,i]):.2f} +{np.percentile(samples[:,i],84)-np.median(samples[:,i]):.3f} -{-np.percentile(samples[:,i],16)+np.median(samples[:,i]):.3f} " for i,x in enumerate(vars)]))
corner.corner(samples,labels=[labels[x] for x in vars])
plt.savefig(output)
