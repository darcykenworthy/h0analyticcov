#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from numpy.lib import recfunctions

import matplotlib.pyplot as plt
from scipy import optimize as op,linalg,stats
from itertools import combinations
import csv


from astropy.coordinates import SkyCoord
from astropy import units as u, constants
from astropy.cosmology import FlatLambdaCDM,Planck15
import pystan
import re,timeit,pickle,argparse,multiprocessing,os,tqdm
import matplotlib.ticker as mtick
from os import path
cosmo=Planck15


# In[2]:


parser = argparse.ArgumentParser(description='Calculate predicted peculiar velocity correction and covariance between supernovae from results of MCMC ')
parser.add_argument('pickle',type=str, nargs='+',
                    help='File with MCMC chain')
parser.add_argument('fitres',type=str, 
                    help='File with SNe')
parser.add_argument('nsamples',type=int,default=30,help='Number of samples from predictive posterior')

parser.add_argument('--destoutput',type=str,default=None, 
                    help='Path to write pickle with bias and covariance')
args = parser.parse_args()
picklefile=args.pickle[0]
fitres=args.fitres
nsamples=args.nsamples
output=args.destoutput if not  args.destoutput is None else path.join(path.dirname(picklefile), 'velocityposterior_'+path.basename(picklefile) )
# In[3]:

sdss=['16314','16392','16333','14318','17186','17784','7876']
trunames=['2006oa','2006ob','2006on','2006py','2007hx','2007jg','2005ir']
def readFitres(fileName):			
    with open(fileName,'r') as file : fileText=file.read()
    result=re.compile('VARNAMES:([\w\s]+)\n').search(fileText)
    names= ['VARNAMES:']+[x for x in result.groups()[0].split() if not x=='']
    namesToTypes={'VARNAMES':'U3','CID':'U20','FIELD':'U4','IDSURVEY':int}
    types=[namesToTypes[x] if x in namesToTypes else float for x in names]
    data=np.genfromtxt(fileName,skip_header=fileText[:result.start()].count('\n')+1,dtype=list(zip(names,types)))
    for sdssName,truName in zip(sdss,trunames):
            data['CID'][data['CID']==sdssName]=truName
    return data
def weightedMean(vals,covs,cut=None):
    if cut is None:cut=np.ones(vals.size,dtype=bool)
    if covs.ndim==1:
        vars=covs
        mean=((vals)/vars)[cut].sum()/(1/vars)[cut].sum()
        chiSquared=((vals-mean)**2/vars)[cut].sum()
        var=1/((1/vars)[cut].sum())
    else:
        vals=vals[cut]
        covs=covs[cut[:,np.newaxis]&cut[np.newaxis,:]].reshape((cut.sum(),cut.sum()))
        #Use cholesky transform instead of numerical inversion of symmetric matrix
        transform=np.linalg.cholesky(covs)
        design=linalg.solve_triangular(transform,np.ones(vals.size),lower=True)
        var=1/np.dot(design,design)
        mean=var*np.dot(design,linalg.solve_triangular(transform,vals,lower=True))
        pulls=linalg.solve_triangular(transform,vals-mean,lower=True)
        chiSquared=np.dot(pulls,pulls)
    return mean,np.sqrt(var),chiSquared

sndataFull=readFitres(fitres)



accum=[]
result=[]
finalInds=[]
sndataNoDups=sndataFull.copy()
for name in np.unique(sndataNoDups['CID']):
    dups=sndataNoDups[sndataNoDups['CID']==name]
    inds=np.where(sndataNoDups['CID']==name)[0]
# #    Prefer non-SDSS surveys
#     if (dups['IDSURVEY']==1).sum()>0 and (dups['IDSURVEY']!=1).sum()>0:
#         cut=dups['IDSURVEY']!=1
#         inds=inds[cut][::-1]
    if (dups['IDSURVEY']==4).sum()>0:
        cut=dups['IDSURVEY']==4
        inds=inds[cut]
    elif (dups['IDSURVEY']==1).sum()>0:
        cut=dups['IDSURVEY']==1
        inds=inds[cut]
    elif (dups['IDSURVEY']==15).sum()>0:
        cut=dups['IDSURVEY']==15
        inds=inds[cut]

    finalInds+=[inds[0]]

sndataNoDups=sndataNoDups[finalInds].copy()
sndata=sndataNoDups.copy()
pecvelcov=np.load('velocitycovariance-{}-darksky_class.npy'.format(path.splitext(path.split(fitres)[-1])[0]))

correction=(cosmo.distmod(sndata['zCMB']) - cosmo.distmod(sndata['zHD']) ).value
z=sndata['zCMB']
zcmb=sndata['zCMB']
snra=sndata['RA']*u.degree
deckey='DEC' if 'DEC' in sndata.dtype.names else 'DECL'
sndec=sndata[deckey]*u.degree
sncoords=SkyCoord(ra=sndata['RA'],dec=sndata[deckey],unit=u.deg)

chi=cosmo.comoving_distance(zcmb).to(u.Mpc).value
snpos=np.zeros((sndata.size,3))
snpos[:,0]=np.cos(sndec)*np.sin(snra)
snpos[:,1]=np.cos(sndec)*np.cos(snra)
snpos[:,2]=np.sin(sndec)
snpos*=chi[:,np.newaxis]

def checkposdef(matrix):
	vals,vecs=linalg.eigh(matrix)
	if (vals>=0).all(): return matrix
	covclipped=np.dot(vecs,np.dot(np.diag(np.clip(vals,vals.max()*1e-10,None)),vecs.T))
	return covclipped 
separation=np.sqrt(((snpos[:,np.newaxis,:]-snpos[np.newaxis,:,:])**2).sum(axis=2))
nonlinear=separation==0

pecvelcov=checkposdef(pecvelcov)

uncorrected=z>.1

bothcorrected=( (~uncorrected[:,np.newaxis]) & (~uncorrected[np.newaxis,:])  )
neithercorrected=( (uncorrected[:,np.newaxis]) & (uncorrected[np.newaxis,:])  )
onecorrected=~ (neithercorrected | bothcorrected)
print(f'{uncorrected.sum()} SNe with no correction and {uncorrected.size-uncorrected.sum()} SNe with correction')
pecvelcovbothcorrected=pecvelcov * bothcorrected
pecvelcovonecorrected=pecvelcov*onecorrected
pecvelcovuncorrected= pecvelcov *neithercorrected

with open(picklefile,'rb') as file:  model,fit,opfit=pickle.load(file)
with open(path.join(path.dirname(picklefile), 'data_'+path.basename(picklefile) ),'rb') as file: data=pickle.load(file)

velscaling=fit.extract('velscaling')['velscaling']
veldispersion=fit.extract('veldispersion_additional')['veldispersion_additional']
if 'betarescale' in fit.extract().keys():
	betarescale=fit.extract(['betarescale'])['betarescale']
else:
	betarescale=np.zeros(velscaling.size)
def calcpecvelcov(data,velscaling):
        return velscaling**2 * pecvelcovbothcorrected+ velscaling * pecvelcovonecorrected+  pecvelcovuncorrected

vpecsamples=np.empty((sndata.size,nsamples))
for i,postindex in enumerate(tqdm.tqdm(np.random.choice(np.arange(velscaling.size),nsamples))):
	vpecsamples[:,i]=stats.multivariate_normal.rvs( betarescale[postindex]*sndata['VPEC'], calcpecvelcov(data,velscaling[postindex])+ veldispersion[postindex]**2 *nonlinear)

print(f'{np.std(betarescale)}, {np.sqrt(np.mean(velscaling**2))}, {np.sqrt(np.mean(veldispersion**2))}')

cid=sndata['CID']
zcmb=sndata['zCMB']
csvoutput=output.replace('pickle','csv')
with open(csvoutput,'w') as file:
    writer=csv.writer(file,delimiter=',')
    writer.writerow(['CID','zCMB']+[f'VPEC_{i}' for i in range(nsamples)])
    for i in range(sndata.size):
           writer.writerow([cid[i],zcmb[i]]+list(vpecsamples[i]))
print(f'Output written to {csvoutput}')



#(betarescale-np.mean(betarescale)[np.newaxis,np.newaxis,:])**2*np.outer(correction,correction)[:,:,np.newaxis] + velscaling[np.newaxis,np.newaxis,:] **2 * pecvelcovmu[:,:,np.newaxis] + veldispersion[np.newaxis,np.newaxis,:]**2 * nonlinearmu[:,:,np.newaxis]
