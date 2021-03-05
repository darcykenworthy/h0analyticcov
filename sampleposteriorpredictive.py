#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.lib import recfunctions

import matplotlib.pyplot as plt
from scipy import optimize as op,linalg,stats
from itertools import combinations

from astropy.coordinates import SkyCoord
from astropy import units as u, constants
from astropy.cosmology import FlatLambdaCDM,Planck15
import re,timeit,pickle,argparse,multiprocessing,os,tqdm
import matplotlib.ticker as mtick
from os import path
cosmo=Planck15


# In[2]:


parser = argparse.ArgumentParser(description='Calculate posterior predictive distribution of mu ')
parser.add_argument('marginalpickle',type=str, 
                    help='File with MCMC chain')
parser.add_argument('nsamples',type=int, 
                    help='Number of times to sample from posterior predictive distribution ')
parser.add_argument('--destoutput',type=str,default=None, 
                    help='Path to write pickle with bias and covariance')
args = parser.parse_args()
picklefile=args.marginalpickle
output=args.destoutput if not  args.destoutput is None else path.join(path.dirname(picklefile), 'posteriorpredictive_'+path.basename(picklefile))
# In[3]:
with open(picklefile,'rb') as file:  model,fit,opfit=pickle.load(file)
with open(path.join(path.dirname(picklefile), 'data_'+path.basename(picklefile) ),'rb') as file: data=pickle.load(file)
extract=lambda x: fit.extract(x)[x]
offset=extract('offset')
intrins=extract('intrins')
velscaling=extract('velscaling')
veldispersion=extract('veldispersion_additional')
veldispersiontotal=extract('veldispersion')
if 'betarescale' in fit.extract().keys():
	betarescale=extract('betarescale')
else:
	betarescale=np.zeros(velscaling.size)


correction=data['mucorrections']
nonlinearmu=data['nonlinearmu']
pecvelcovmu=data['pecvelcovmu']
biasscale=data['biasscale']
muerr=data['muerr']
musample=np.empty((correction.size,args.nsamples))
parsample=[]
for i,postindex in enumerate(tqdm.tqdm(np.random.choice(np.arange(velscaling.size),musample.shape[1]))):
	musample[:,i]=stats.multivariate_normal.rvs((betarescale[postindex]*correction+offset[postindex]),np.diag(biasscale*intrins[postindex]**2 +muerr**2)+  velscaling[postindex]**2 * pecvelcovmu + veldispersion[postindex]**2 *nonlinearmu)
	parsample+=[{'offset':offset[postindex],'intrins':intrins[postindex],'velscaling':velscaling[postindex],'veldispersion_additional':veldispersion[postindex],'betarescale':betarescale[postindex],'veldispersion':veldispersiontotal[postindex]}]

with open(output,'wb') as file:  pickle.dump([musample,parsample],file)

print(f'Output written to {output}')
