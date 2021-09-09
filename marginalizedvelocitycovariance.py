#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from numpy.lib import recfunctions

import matplotlib.pyplot as plt
from scipy import optimize as op,linalg,stats
from itertools import combinations

from astropy.coordinates import SkyCoord
from astropy import units as u, constants,coordinates as coord
from astropy.cosmology import FlatLambdaCDM,Planck15

import re,timeit,pickle,argparse,multiprocessing,os
import matplotlib.ticker as mtick
from os import path
import time
cosmo=Planck15
import utilfunctions 
import tqdm
import csv

import matplotlib as mpl
plt.rcParams['font.size'] = 10


# In[12]:

parser = argparse.ArgumentParser(description='Calculate predicted peculiar velocity correction and covariance between supernovae from results of MCMC ')
parser.add_argument('extractpickle',type=str,
                    help='File with MCMC chain')
parser.add_argument('fitres',type=str, 
                    help='File with SNe')
parser.add_argument('--velocitycov',type=str,default=None,
                    help='.npy file containing peculiar velocity covariance matrix')
parser.add_argument('--destoutput',type=str,default=None, 
                    help='Path to write pickle with bias and covariance')
parser.add_argument('--nsamples',type=int,default=1000, 
                    help='Number of samples to draw from posterior')

args = parser.parse_args()
fitres=args.fitres	
output=args.destoutput if not  args.destoutput is None else path.join(path.dirname(args.extractpickle), path.basename(args.extractpickle).replace('extract_','posteriorbiascov_') )
nsamples=args.nsamples
sndatadups=utilfunctions.readFitres(fitres)
sndataFull=sndatadups.copy()
#sndataFull['MUERR_RAW']=np.sqrt(sndataFull['MUERR_RAW']**2-sndataFull['MUERR_VPEC']**2)
sndatadups=utilfunctions.renameDups(sndatadups)
sndataFull=utilfunctions.cutdups(sndatadups,reweight=True)
#


sndata=sndataFull
#sndata=sndataFull.copy()
z=sndata['zCMB']
if (sndata['DEC']<-90).any():
    sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['RA']=202.303420
    sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['DEC']=11.272390

    sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['HOST_RA']=202.306840
    sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['HOST_DEC']=11.275820

    if 'SNF20080909-030' in sndata['CID']:
        sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['RA']=330.454458
        sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['DEC']= 13.055306

        sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['HOST_RA']=330.456000
        sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['HOST_DEC']= 13.055194

    sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['RA']=3.749292
    sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['DEC']=16.335000
    

    sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['HOST_RA']=3.750292
    sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['HOST_DEC']=16.333242
assert(sndata['DEC']>-90).all()


sncoords,snpos,separation,angsep=utilfunctions.getpositions(sndata,hostlocs=False)
sndata=utilfunctions.separatevpeccontributions(sndata,sncoords)



zmin=0.06
hascorrection=z<zmin


# In[13]:


z=sndata['zCMB']
dmudvpec=5/np.log(10)*((1+z)**2/(cosmo.efunc(z)*cosmo.H0*cosmo.luminosity_distance(z))).to(u.s/u.km).value
pecvelcov=np.load(args.velocitycov if args.velocitycov else 'velocitycovariance-{}-darksky_class.npy'.format(path.splitext(path.split(fitres)[-1])[0]))
pecvelcov=utilfunctions.checkposdef(pecvelcov)




# In[14]:


    



# In[22]:


def posterior(prior,pred,constraint,hasconstraint):
        sumcovtransform=linalg.cholesky(prior[hasconstraint,:][:,hasconstraint]+constraint,lower=True)
        sumcovtransform_times_priorcov= linalg.solve_triangular(sumcovtransform,prior[hasconstraint,:][:,hasconstraint],lower=True);
        sumcovtransform_times_tmppcov=  linalg.solve_triangular(sumcovtransform, constraint,lower=True);
        sumcovtransform_times_transfermatrix=  linalg.solve_triangular(sumcovtransform, prior[hasconstraint,:][:,~hasconstraint] ,lower=True);
        pecvelmeanmarginal= np.empty(prior.shape[0])
        pecvelmeanmarginal[hasconstraint]=np.dot(sumcovtransform_times_priorcov.T ,linalg.solve_triangular(sumcovtransform,pred,lower=True));
        pecvelmeanmarginal[~hasconstraint]=np.dot(sumcovtransform_times_transfermatrix.T ,linalg.solve_triangular(sumcovtransform,pred,lower=True));
        
        pecvelcovmarginal=np.empty(prior.shape)
        pecvelcovmarginal[np.outer(hasconstraint,hasconstraint  )]= (np.dot(sumcovtransform_times_tmppcov.T ,sumcovtransform_times_priorcov )).flatten()
        pecvelcovmarginal[np.outer(hasconstraint,~hasconstraint)] = (np.dot(sumcovtransform_times_tmppcov.T ,sumcovtransform_times_transfermatrix) ).flatten()
        pecvelcovmarginal[np.outer(~hasconstraint,hasconstraint)] = (pecvelcovmarginal[np.outer(hasconstraint,~hasconstraint)].T).flatten()
        pecvelcovmarginal[np.outer(~hasconstraint,~hasconstraint)]= prior[np.outer(~hasconstraint,~hasconstraint)]- (np.dot(sumcovtransform_times_transfermatrix.T,sumcovtransform_times_transfermatrix)).flatten()
        return pecvelmeanmarginal,pecvelcovmarginal
with open(args.extractpickle,'rb') as file:
    samplingrun=pickle.load(file)

marginalizedvelmean,marginalizedvelcov=np.empty((sndata.size,nsamples)),np.empty((sndata.size,sndata.size,nsamples))

for index,i in tqdm.tqdm(enumerate(np.random.randint(0,samplingrun['offset'].size,nsamples)),total=nsamples):
    post,postcov=posterior(samplingrun['sigmarescale'][i]**2*pecvelcov,(samplingrun['betarescale'][i]*sndata['VPEC_LOCAL']+samplingrun['vextrescale'][i]*sndata['VPEC_BULK'])[hascorrection],np.diag(np.ones(hascorrection.sum())*samplingrun['correctionstd'][i]**2),hascorrection)
    marginalizedvelmean[:,index]=post
    marginalizedvelcov[:,:,index]=postcov+np.diag(np.ones(sndata.size)*samplingrun['veldispersion_additional'][i])
marginalizedvelmean,marginalizedvelcov=np.mean(marginalizedvelmean,axis=-1),np.mean(marginalizedvelcov,axis=-1)+np.cov(marginalizedvelmean)

cid=sndata['CID']
zcmb=sndata['zCMB']
marginalizedcorrection,marginalizedcov=dmudvpec*marginalizedvelmean,np.outer(dmudvpec,dmudvpec)*marginalizedvelcov


# In[23]:
with open(output,'wb') as file: pickle.dump((marginalizedcorrection,marginalizedcov),file)

csvoutput=output.replace('pickle','csv')
with open(csvoutput,'w') as file:
	writer=csv.writer(file,delimiter=',')
	writer.writerow(['CID_1','zCMB_1','MU_VEL_CORRECTION_1','CID_2','zCMB_2','MU_VEL_CORRECTION_1','MU_VEL_COVARIANCE'])
	for i in range(sndata.size):
		for j in range(sndata.size):
		   writer.writerow([cid[i],zcmb[i],marginalizedcorrection[i],cid[j],zcmb[j],marginalizedcorrection[j],marginalizedcov[i,j]])


# In[ ]:




