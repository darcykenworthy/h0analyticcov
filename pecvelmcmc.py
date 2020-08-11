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
import pystan
import re,timeit,pickle,argparse,multiprocessing,os
import matplotlib.ticker as mtick
from os import path
cosmo=Planck15


# In[2]:


parser = argparse.ArgumentParser(description='Calculate velocity covariance between supernovae from fitres file')
parser.add_argument('fitres',type=str, 
                    help='File with supernova distances')
parser.add_argument('redshiftfile',type=str, 
                    help='File with supernova redshifts in labeled columns')
parser.add_argument('zkey',type=str,
                    help='Name (or column) of the redshift column to be used for this run')
parser.add_argument('--niter',type=int,default=3000,
                    help='Number of iterations per chain')
parser.add_argument('--nchains',type=int,default=4,
                    help='Number of chains to run')
parser.add_argument('--flatprior',action='store_true',
                    help='Use flat priors on all parameters (default is Jeffreys prior)')

args = parser.parse_args()
fitres=args.fitres
redshiftfile=args.redshiftfile
redshiftcolumn=args.zkey
niter,nchains=args.niter,args.nchains
redshiftcut= (.0,.08)
pickleoutput='{}_{}_{}.pickle'.format( path.splitext(path.split(fitres)[-1])[0],path.splitext(path.split(redshiftfile)[-1])[0],redshiftcolumn)
if args.flatprior:
	pickleoutput='flat_'+pickleoutput
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
sndataFull['MUERR']=sndataFull['MUERR_RAW']


# In[4]:


try:
    ncpus = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
except KeyError:
    ncpus = multiprocessing.cpu_count()

try:
    zvector=np.genfromtxt(redshiftfile,skip_header=1)[:,int(redshiftcolumn)]
except:
    zvector=np.genfromtxt(redshiftfile,names=True)[redshiftcolumn]
zvectornames=readFitres('lowz_comb_csp_hst.fitres')['CID']
zvectorindices=[]
i=0
j=0
while i<sndataFull.size and j<zvectornames.size:
    if sndataFull['CID'][i]==zvectornames[j]:
        zvectorindices+=[j]
        i+=1
        j+=1
    else:
        j+=1
zvectorindices=np.array(zvectorindices)
sndataFull=np.array(recfunctions.append_fields(sndataFull,'zeval',zvector[zvectorindices]))

# In[5]:

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
    if len(inds)>1:
        for x in ['MU','c','x1']:
            sndataNoDups[x][inds[0]],sndataNoDups[x+'ERR'][inds[0]],_=weightedMean(sndataNoDups[x],sndataNoDups[x+'ERR']**2,sndataNoDups['CID']==name)

sndataNoDups=sndataNoDups[finalInds].copy()
sndata=sndataNoDups.copy()


# In[6]:

zcmb=sndata['zCMB']
snra=sndata['RA']*u.degree
sndec=sndata['DECL']*u.degree
sncoords=SkyCoord(ra=sndata['RA'],dec=sndata['DECL'],unit=u.deg)

chi=cosmo.comoving_distance(zcmb).to(u.Mpc).value
snpos=np.zeros((sndata.size,3))
snpos[:,0]=np.cos(sndec)*np.sin(snra)
snpos[:,1]=np.cos(sndec)*np.cos(snra)
snpos[:,2]=np.sin(sndec)
snpos*=chi[:,np.newaxis]

separation=np.sqrt(((snpos[:,np.newaxis,:]-snpos[np.newaxis,:,:])**2).sum(axis=2))
angsep=np.empty((sndata.size,sndata.size))
for i in range(sndata.size):
    angsep[i,:]=sncoords.separation(sncoords[i])



# In[8]:


muresiduals=cosmo.distmod(sndata['zeval']).value - sndata['MU']

# In[10]:


z=sndata['zCMB']

dmudvpec=5/np.log(10)*((1+z)**2/(cosmo.efunc(z)*cosmo.H0*cosmo.luminosity_distance(z))).to(u.s/u.km).value
velocityprefactor=np.outer(dmudvpec,dmudvpec)

pecvelcov=np.load('velocitycovariance-{}-darksky_class.npy'.format(path.splitext(path.split(fitres)[-1])[0]))

nonlinear=separation<10
nonlinearmu=velocityprefactor*nonlinear+np.diag(2.5e-6*np.ones(z.size))
pecvelcovmu=velocityprefactor*pecvelcov+np.diag(2e-20*np.ones(z.size))
assert(linalg.eigvalsh(pecvelcovmu).min()>=0)
assert(linalg.eigvalsh(nonlinearmu).min()>=0)


# In[13]:


dataparamblocks="""
data {
    int<lower=0> N; // number of SNe
    vector[N] muresiduals;
    vector<lower=0>[N] muerr; // s.e. of effect estimates
    vector[N] biasscale;
    cov_matrix[N] nonlinearmu;
    cov_matrix[N] pecvelcovmu;
    matrix[N,N] systematics;
}
transformed data{
    vector[N] zeros= rep_vector(0,N);
    cov_matrix[N] identity=diag_matrix(rep_vector(1.0,N));
}

parameters {
    real<multiplier=0.01> offset;
    real<lower=.01,upper=10> velscaling;
    real<lower=10,upper=500> veldispersion_additional;
    real<lower=.01,upper=.3> intrins;
}
transformed parameters {
    real veldispersion;
    veldispersion=sqrt(square(veldispersion_additional)+square(243*velscaling));
}
"""

jeffreyspriors = """
model {
    real sigparams[3] = {intrins,veldispersion_additional, velscaling};
    matrix[N,N] sigma = diag_matrix(biasscale*square(intrins)+square(muerr))+nonlinearmu*square(veldispersion_additional)+systematics+square(velscaling)*pecvelcovmu;
    matrix[N,N] invsigdesigns[3] = {mdivide_left_spd(sigma,diag_matrix(biasscale)),mdivide_left_spd(sigma,nonlinearmu),mdivide_left_spd(sigma,pecvelcovmu)};
    matrix[3,3] fishermatrix;
    
    muresiduals ~ multi_normal(zeros+offset, sigma);
    
    for (i in 1:3){
        for (j in 1:3){
            fishermatrix[i,j]= sigparams[i]*sigparams[j] *  sum(invsigdesigns[i].*invsigdesigns[j]);
        }
    }
    target+=(.5*log_determinant(fishermatrix));
}
"""

flatpriors="""
model {
    matrix[N,N] sigma = diag_matrix(biasscale*square(intrins)+square(muerr))+nonlinearmu*square(veldispersion_additional)+systematics+square(velscaling)*pecvelcovmu;
    muresiduals ~ multi_normal(zeros+offset, sigma);
}
"""


# In[14]:

if args.flatprior:
	model = pystan.StanModel(model_code=dataparamblocks+flatpriors,model_name='flatmodel')
else:
	model=pystan.StanModel(model_code=dataparamblocks+jeffreyspriors,model_name='jeffreysmodel')


# In[15]:


cut= (z>redshiftcut[0])&(z<redshiftcut[1])
standat = {'N': cut.sum(),
               'zeros':np.zeros(cut.sum()),
               'muresiduals': muresiduals[cut],
               'muerr': sndata['MUERR_RAW'][cut],
               'biasscale':sndata['biasScale_muCOV'][cut],
               'nonlinearmu':nonlinearmu[cut,:][:,cut],
               'systematics': np.zeros((cut.sum(),cut.sum())),
               'pecvelcovmu':pecvelcovmu[cut,:][:,cut],
              }




fit = model.sampling(data=standat, iter=niter, chains=nchains,n_jobs=ncpus,warmup = min(niter//2,1000),init=lambda :{
    'offset': np.random.normal(0,1e-2),
    'velscaling': np.exp(np.random.normal(0,.1)),
    'veldispersion_additional': np.exp(np.random.normal(np.log(200),.1)),
    'intrins': np.exp(np.random.normal(np.log(.12),.1))
})

# In[17]:


with open(pickleoutput,'wb') as picklefile: pickle.dump([model,fit],picklefile,-1)

