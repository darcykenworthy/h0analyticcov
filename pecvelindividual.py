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
import hashlib
cosmo=Planck15
from utilfunctions import *

# In[2]:


parser = argparse.ArgumentParser(description='Calculate velocity covariance between supernovae from fitres file')
parser.add_argument('fitres',type=str, 
                    help='File with supernova distances')

parser.add_argument('--redshiftcut',type=float,nargs=2,default=[0.01,10],
                    help='Minimum and maximum redshift values')
parser.add_argument('--niter',type=int,default=3000,
                    help='Number of iterations per chain')
parser.add_argument('--outputdir',type=str,default='.',
                    help='Directory to write output pickle file')
parser.add_argument('--nchains',type=int,default=4,
                    help='Number of chains to run')
parser.add_argument('--clobber', action='store_true', 
                    help='Overwrite existing final output file')
parser.add_argument('--njobs', type=int,default=0, 
                    help='Number of jobs to launch (defaults to the number of cpu\'s available)')

parser.add_argument('--optimizing',action='store_true',
                    help='Useful for debugging: don\'t run mcmc, instead just return mode of posterior')
parser.add_argument('--flatprior',action='store_true',
                    help='Use flat priors on all parameters (default is Jeffreys prior)')
parser.add_argument('--sampleprior', action='store_true', 
                    help='Sample directly from prior distribution without constraints from data')

parser.add_argument('--nocorrections',action='store_true',
					help='No 2M++ corrections')
parser.add_argument('--fixcorrectionparams',action='store_true',
					help='Fix 2M++ nuisance parameters to fiducial values')
parser.add_argument('--fixveldispersion',action='store_true',
					help='Fix nonlinear velocity dispersion to 250 km/s')
parser.add_argument('--intrins',type=float,default=None,
                    help='Fix intrinsic scatter to a given value')

args = parser.parse_args()

fitres=args.fitres
niter,nchains=args.niter,args.nchains
outputdir=args.outputdir
redshiftcut= args.redshiftcut
pickleoutput='{}.pickle'.format( path.splitext(path.split(fitres)[-1])[0])
fixintrins=not (args.intrins is None)
intrins=args.intrins

if args.flatprior:
	pickleoutput='flat_'+pickleoutput
if args.sampleprior:
	pickleoutput='prior_'+pickleoutput
	
os.makedirs(outputdir,exist_ok=True)
pickleoutput=path.join(outputdir,pickleoutput)
if path.exists(pickleoutput) and not args.clobber and not args.optimizing:
	raise ValueError('Clobber is false and output file already exists')
# In[3]:

    
sndataFull=readFitres(fitres)
sndata=cutdups(sndataFull,reweight=True)

z=sndata['zCMB']

cut= (z>redshiftcut[0])&(z<redshiftcut[1])
print(f'{cut.sum()} SNe in redshift range {redshiftcut}')
sndata=sndata[cut].copy()
z=sndata['zCMB']

sncoords,separation,angsep=getseparation(sndata)
sndata=separatevpeccontributions(sndata,sncoords)

try:
    ncpus = sum([int(x) for x in os.environ["SLURM_JOB_CPUS_PER_NODE"].split(',')])
except KeyError:
    ncpus = multiprocessing.cpu_count()
    
print(ncpus)
if args.njobs==0:
	njobs=ncpus
else:
	njobs=args.njobs
	


# In[8]:
muresiduals=cosmo.distmod(sndata['zCMB']).value - sndata['MU']
# In[10]:



dmudvpec=5/np.log(10)*((1+z)**2/(cosmo.efunc(z)*cosmo.H0*cosmo.luminosity_distance(z))).to(u.s/u.km).value
velocityprefactor=np.outer(dmudvpec,dmudvpec)

pecvelcov=np.load('velocitycovariance-{}-darksky_class.npy'.format(path.splitext(path.split(fitres)[-1])[0]))[cut,:][:,cut]
nonlinear=(separation==0).astype(float)



pecvelcov=checkposdef(pecvelcov)
nonlinear=checkposdef(nonlinear)



define_additional_data=''
define_additional_constants=''
define_additional_params=''
define_additional_model_quantities=''
model_name=''
# In[13]:


if fixintrins:
	model_name+='fixintrins_'
	define_additional_constants+=f"""
    real<lower=.01,upper=1> intrins={intrins};
    vector[N] sigmamuresids = sqrt(biasscale*square(intrins)+square(muerr));
"""
else:
	define_additional_params+="""
    real<lower=.01,upper=1> intrins;
"""
	define_additional_model_quantities+="""
	vector[N] sigmamuresids = sqrt(biasscale*square(intrins)+square(muerr));
"""

if args.fixveldispersion:
	model_name+='fixveldispersion_'
	define_additional_constants+="""
    real<lower=10,upper=500> veldispersion_additional=200;
    matrix[N,N] pecvelcovtot=pecvelcov+diag_matrix(rep_vector(square(veldispersion_additional),N));
"""
else:
	define_additional_params+="""
    real<lower=10,upper=500> veldispersion_additional;
"""
	define_additional_model_quantities+="""
	matrix[N,N] pecvelcovtot=pecvelcov+diag_matrix(rep_vector(square(veldispersion_additional),N));
"""
if args.fixcorrectionparams:
	model_name+='fixedcorrections_'
	define_additional_constants+="""
	real<lower=10,upper=1000> correctionstd=150;
"""


else:

	define_additional_params+="""
	real<lower=10,upper=1000> correctionstd;
"""

if args.flatprior:
	prior_block="""
	//lambdacdm prior
	peculiarvelocities ~ multi_normal(zeros,pecvelcovtot);	
"""
else:
	prior_block="""
	//SN priors
	offset ~ normal(0,.5);
	intrins ~ lognormal(log(0.1),.5);
	
	//2m++ prior (based on N body simulations)
	correctionstd ~ lognormal(log(150),.5);

	veldispersion_additional ~ lognormal(log(250),.5);
	
	// inverse improper priors on scale parameters
	target+=-log(veldispersion_additional);
	target+=-log(intrins);
	target+=-log(correctionstd);
	
"""
prior_block+="""
	//lambdacdm prior
	peculiarvelocities ~ multi_normal(zeros,pecvelcovtot);
"""
if args.sampleprior:
	model_name='prior_'+model_name
	calculate_residuals="" 
else:
	if args.nocorrections:
		calculate_residuals="""
	//No 2M++ constraints
	"""
	else:
		calculate_residuals="""
    //2M++ constraints
    vextrescale ~ normal(1,vextfracerr);
    peculiarvelocities[correctedinds] ~ normal(velcorrections+bulkcorrections[correctedinds],correctionstd);
""" 

	calculate_residuals+="""
	//SN constraints
	muresiduals ~ normal(offset+dmudvpec .* peculiarvelocities, sigmamuresids);
"""

model_name+='individual_model'
model_code=f"""

data {{
    int<lower=0> N; // number of SNe
    vector[N] muresiduals;
    vector<lower=0>[N] muerr; 
    vector[N] biasscale;
    vector[N] zCMB;
    
    vector[N] dmudvpec;

    int<lower=0> M; // number of 2M++ local corrections
    int  correctedinds[M]; //indices of SNe with local 2m++ corrections
    vector[M] velcorrections;
    vector[N] bulkcorrections;
    matrix[N,N] pecvelcov;
    {define_additional_data}
    
}}

transformed data{{
    vector[N] zeros= rep_vector(0,N);
    cov_matrix[N] identity=diag_matrix(rep_vector(1.0,N));

    real vextfracerr= 23./159;

    real bulkfiducial=dot_self(bulkcorrections);
    {define_additional_constants}

}}


parameters {{
    real<offset=0,multiplier=.1> offset;
    vector<offset=0,multiplier=400>[N] peculiarvelocities; 
    
    {define_additional_params}
}}

transformed parameters {{
    real vextrescale = dot_product(bulkcorrections,peculiarvelocities)/bulkfiducial ;
}}


model {{
    
    
    {define_additional_model_quantities}

    {prior_block}
    {calculate_residuals}
}}
"""
codefile=path.join(outputdir,f'{model_name}.stan')
if path.exists(codefile):
	with open(codefile,'r') as file: oldcode=file.read()
	existingmodelisgood= oldcode==model_code	
else:
	existingmodelisgood=False
with open(codefile,'w') as file: file.write(model_code)

hascorrection=sndata['VPEC_LOCAL']!=0
correctedindices=np.where(hascorrection)[0]

standat = {'N': sndata.size,
               'muresiduals': muresiduals,
               'muerr': sndata['MUERR_RAW'],
               'biasscale':sndata['biasScale_muCOV'],
          		'zCMB':sndata['zCMB'],
          		
               'dmudvpec':dmudvpec,
               'M':  len(correctedindices),
               
               'correctedinds': correctedindices+1, #Stan indexes from 1 &*#$*@#^$
               'velcorrections': sndata['VPEC_LOCAL'][correctedindices],
               'bulkcorrections': sndata['VPEC_BULK'],
               
               'pecvelcov':pecvelcov,
              }

if fixintrins:
	standat['intrins']=intrins

modelpickleoutput=path.join(outputdir,f'{model_name}.pickle')
with open(path.join(outputdir,'data_'+path.basename(pickleoutput)),'wb') as file: pickle.dump(standat,file,-1) 
if existingmodelisgood and path.exists(modelpickleoutput):
	print('reloading model from existing file')
	with open(modelpickleoutput,'rb') as file: model=pickle.load(file) 
else:
	print('recompiling model')
	model = pystan.StanModel(model_code=model_code,model_name=model_name)
	with open(modelpickleoutput,'wb') as file: pickle.dump(model,file,-1) 
initfun=lambda :{
    'offset': np.random.normal(0,1e-2),
    'velscaling': np.exp(np.random.normal(0,.1)),
    'veldispersion_additional': np.exp(np.random.normal(np.log(200),.1)),
    'intrins': np.exp(np.random.normal(np.log(.12),.1)),
    'correctionstd': np.exp(np.random.normal(np.log(150),.1)),
    'peculiarvelocities': np.random.multivariate_normal(sndata['VPEC_LOCAL']+sndata['VPEC_BULK'],pecvelcov+np.diag(200**2*np.ones(sndata.size)))
}
opfit = model.optimizing(data=standat,init=initfun,iter=niter)
print(opfit)

if not args.optimizing:
	fit = model.sampling(data=standat, iter=niter, chains=nchains,n_jobs=njobs,warmup = min(niter//2,1000),init=initfun)
	print(fit.stansummary())
	with open(pickleoutput,'wb') as picklefile: pickle.dump([model,fit,opfit],picklefile,-1)

# In[17]:



