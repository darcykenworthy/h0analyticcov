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
import re,timeit,pickle,argparse,multiprocessing,os,sys
import matplotlib.ticker as mtick
from os import path
import hashlib
cosmo=Planck15

from utilfunctions import *
# In[2]:


parser = argparse.ArgumentParser(description='Calculate velocity covariance between supernovae from fitres file')
parser.add_argument('fitres',type=str, 
                    help='File with supernova distances')
parser.add_argument('redshiftfile',type=str,
                    help='File with supernova redshifts in labeled columns')
parser.add_argument('zkey',type=str,
                    help='Name (or column) of the redshift column to be used for this run')
parser.add_argument('--intrins',type=float,default=None,
                    help='Fix intrinsic scatter to a given value')
parser.add_argument('--redshiftcut',type=float,nargs=2,default=[0.01,.08],
                    help='Minimum and maximum redshift values')
parser.add_argument('--niter',type=int,default=3000,
                    help='Number of iterations per chain')
parser.add_argument('--outputdir',type=str,default='.',
                    help='Directory to write output pickle file')
parser.add_argument('--nchains',type=int,default=4,
                    help='Number of chains to run')
parser.add_argument('--extra_dispersion',action='store_true',
                    help='Add an additional parameter for diagonal velocity uncertainty')
parser.add_argument('--optimizing',action='store_true',
                    help='Useful for debugging: don\'t run mcmc, instead just return mode of posterior')
parser.add_argument('--flatprior',action='store_true',
                    help='Use flat priors on all parameters (default is Jeffreys prior)')
parser.add_argument('--fixuncorrectedcovariance',default=False, nargs='?',const=.1,type=float, 
                    help='Fix the scale of the peculiar velocity covariance to 1 for all SNe without a peculiar velocity correction')
parser.add_argument('--fixcovariance', action='store_true',
                    help='Fix the scale of the peculiar velocity covariance matrix to 1')
parser.add_argument('--varyscaling', action='store_true', 
                    help='Vary the overall scale of 2m++ corrections')
parser.add_argument('--groupsonly', action='store_true', 
                    help='If true, only use supernovae whose redshifts are different from those in the 2m++ column')
parser.add_argument('--posteriorpredictive', default=None,type=str,
                    help='Path to pickle file with simulated samples and values. If provided, these override muresiduals from fitres and redshift file')
parser.add_argument('--sampleprior', action='store_true', 
                    help='Sample directly from prior distribution without constraints from data')
parser.add_argument('--njobs', type=int,default=0, 
                    help='Number of jobs to launch (defaults to the number of cpu\'s available)')
parser.add_argument('--clobber', action='store_true', 
                    help='Overwrite existing final output file')
args = parser.parse_args()
print(args)
fitres=args.fitres
redshiftfile=args.redshiftfile
redshiftcolumn=args.zkey
niter,nchains=args.niter,args.nchains
outputdir=args.outputdir
redshiftcut= args.redshiftcut
rescalebeta=( args.varyscaling )
isposteriorpredictivecheck=( not args.posteriorpredictive is None)
pickleoutput='{}_{}_{}.pickle'.format( path.splitext(path.split(fitres)[-1])[0],path.splitext(path.split(redshiftfile)[-1])[0],redshiftcolumn)
fixintrins=not (args.intrins is None)
intrins=args.intrins

if rescalebeta:
	pickleoutput="rescalebeta_"+pickleoutput
if args.extra_dispersion:
	pickleoutput="extra_dispersion_"+pickleoutput
if args.flatprior:
	pickleoutput='flat_'+pickleoutput
if args.sampleprior:
	pickleoutput='prior_'+pickleoutput
elif args.posteriorpredictive:
	pickleoutput='posteriorpredictive_'+pickleoutput	
os.makedirs(outputdir,exist_ok=True)
pickleoutput=path.join(outputdir,pickleoutput)
if path.exists(pickleoutput) and not args.clobber and not args.optimizing:
	raise ValueError('Clobber is false and output file already exists')
# In[3]:

sndataFull=readFitres(fitres)

if not isposteriorpredictivecheck: sndataFull=addredshiftcolumnfromfile(redshiftfile,sndataFull,redshiftcolumn)
sndata=cutdups(sndataFull,reweight=True)

try:
    ncpus = sum([int(x) for x in os.environ["SLURM_JOB_CPUS_PER_NODE"].split(',')])
except KeyError:
    ncpus = multiprocessing.cpu_count()
print(ncpus)
if args.njobs==0:
	njobs=ncpus
else:
	njobs=args.njobs
	

# In[5]:



# In[6]:

sncoords,separation,angsep=getseparation(sndata)


# In[8]:
if isposteriorpredictivecheck:
        with open(args.posteriorpredictive,'rb') as file: musamples,parsamples=pickle.load(file)
        simmedmuresiduals=musamples[:,int(args.zkey)]
        simmedparams=parsamples[int(args.zkey)]
else:
        muresiduals=cosmo.distmod(sndata['zeval']).value - sndata['MU']

# In[10]:

mucorrections=(cosmo.distmod(sndata['zCMB']) - cosmo.distmod(sndata['zHD']) ).value

z=sndata['zCMB']
dmudvpec=5/np.log(10)*((1+z)**2/(cosmo.efunc(z)*cosmo.H0*cosmo.luminosity_distance(z))).to(u.s/u.km).value
velocityprefactor=np.outer(dmudvpec,dmudvpec)

pecvelcov=np.load('velocitycovariance-{}-darksky_class.npy'.format(path.splitext(path.split(fitres)[-1])[0]))

nonlinear=separation==0


nonlinearmu=velocityprefactor*(nonlinear)
pecvelcovmu=velocityprefactor*(pecvelcov)

pecvelcovmu=checkposdef(pecvelcovmu)
nonlinearmu=checkposdef(nonlinearmu)


define_additional_data=''
define_additional_constants=''
model_name=''
# In[13]:


if args.fixuncorrectedcovariance:
	model_name+='uncorrectedfixed_'
	define_additional_data="""
	matrix[N,N] pecvelcovmuonecorrected;
	matrix[N,N] pecvelcovmuuncorrected;
"""
	define_sigma=""" matrix[N,N] sigma = diag_matrix(biasscale*square(intrins)+square(muerr))+systematics+square(velscaling)*pecvelcovmu + velscaling *pecvelcovmuonecorrected + pecvelcovmuuncorrected"""

else:
	define_sigma=""" matrix[N,N] sigma = diag_matrix(biasscale*square(intrins)+square(muerr))+systematics+square(velscaling)*pecvelcovmu"""



if args.extra_dispersion:
	model_name+='extra_veldispersion_'
	define_additional_params="""
	    real<lower=10,upper=500> veldispersion_additional;
"""
	define_veldispersion="""
	    real veldispersion=sqrt(square(veldispersion_additional)+square(243*velscaling));
"""
	define_sigma=define_sigma+'+nonlinearmu*square(veldispersion_additional)'
else:
	define_additional_params="""
"""
	define_veldispersion="""
	    real veldispersion=243*velscaling;
"""
if fixintrins:
	model_name+='fixintrins_'
	define_additional_constants+="""
    		real<lower=.01,upper=.3> intrins;
"""
else:
	define_additional_params+="""
    		real<lower=.01,upper=.3> intrins;
"""
if args.fixcovariance:
	model_name+='fixedpeccov_'
	define_additional_constants+="""
	    real<lower=.01,upper=10> velscaling=1;
"""
else:
	define_additional_params+="""
	    real<lower=.01,upper=10> velscaling;
"""
if not args.flatprior:
	model_name+='jeffreys_'
	sigparams,design= zip(*([('intrins','intrins*diag_matrix(biasscale)')] if not fixintrins else []) + ([('veldispersion_additional','veldispersion_additional*nonlinearmu')] if args.extra_dispersion else []) + ([] if args.fixcovariance else [
	('velscaling', '2 *(velscaling)*pecvelcovmu + pecvelcovmuonecorrected' if args.fixuncorrectedcovariance else 'velscaling*pecvelcovmu')]))
	nsig= len(sigparams) 
	
	prior_block=f"""
	    real sigparams[{nsig}] = {{ { ','.join(sigparams) }  }};
	    matrix[N,N] invsigdesigns[{nsig}] = {{ {','.join(['mdivide_left_spd(sigma,'+x+')' for x in design])} }};
	    matrix[{nsig},{nsig}] fishermatrix;
	    for (i in 1:{nsig}){{
	        for (j in 1:{nsig}){{
		    fishermatrix[i,j]=  sum(invsigdesigns[i].*invsigdesigns[j]);
	        }}
	    }}
	    target+=(.5*log_determinant(fishermatrix));    
"""
else:
	model_name+='flat_'
	prior_block=''
if rescalebeta:
	model_name+='rescale_beta_'
	define_additional_params+="""
	    real<lower=-1,upper=2> betarescale;
"""
	calculate_residuals="""
	    muresiduals ~ multi_normal(betarescale*mucorrections+offset, sigma);
"""
else:
	calculate_residuals="""
	    muresiduals ~ multi_normal(zeros+offset, sigma);
""" 

if args.sampleprior:
	model_name='prior_'+model_name
	calculate_residuals="" 

model_name+='model'
model_code=f"""
data {{
    int<lower=0> N; // number of SNe
    vector[N] muresiduals;
    vector[N] mucorrections;
    vector<lower=0>[N] muerr; // s.e. of effect estimates
    vector[N] biasscale;
    matrix[N,N] pecvelcovmu;
    matrix[N,N] nonlinearmu;
    matrix[N,N] systematics;
    {define_additional_data}
}}
transformed data{{
    vector[N] zeros= rep_vector(0,N);
    cov_matrix[N] identity=diag_matrix(rep_vector(1.0,N));
    {define_additional_constants}
}}

parameters {{
    real<lower=-.3,upper=.3> offset;
    {define_additional_params}
}}
transformed parameters {{
    {define_veldispersion}
}}
model {{
    {define_sigma};
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

cut= (z>redshiftcut[0])&(z<redshiftcut[1])
print(f'{cut.sum()} SNe in redshift range {redshiftcut}')
if args.groupsonly:
	cut=cut&sndata['isgroup']
standat = {'N': cut.sum(),
               'zeros':np.zeros(cut.sum()),
               'muresiduals': simmedmuresiduals if args.posteriorpredictive else muresiduals[cut],
               'mucorrections':mucorrections[cut],
	       'muerr': sndata['MUERR_RAW'][cut],
               'biasscale':sndata['biasScale_muCOV'][cut],
               'nonlinearmu':nonlinearmu[cut,:][:,cut],
               'systematics': np.zeros((cut.sum(),cut.sum())),
               'pecvelcovmu':pecvelcovmu[cut,:][:,cut],
              }
if args.fixuncorrectedcovariance:
	cutpecvelcovmu=pecvelcovmu[cut,:][:,cut]
	uncorrected=z[cut]>args.fixuncorrectedcovariance
	
	bothcorrected=( (~uncorrected[:,np.newaxis]) & (~uncorrected[np.newaxis,:])  )
	neithercorrected=( (uncorrected[:,np.newaxis]) & (uncorrected[np.newaxis,:])  )
	onecorrected=~ (neithercorrected | bothcorrected)
	print(f'{uncorrected.sum()} SNe with no correction and {uncorrected.size-uncorrected.sum()} SNe with correction')
	standat['pecvelcovmu']=cutpecvelcovmu * bothcorrected
	standat['pecvelcovmuonecorrected']=cutpecvelcovmu*onecorrected
	standat['pecvelcovmuuncorrected']= cutpecvelcovmu *neithercorrected
if fixintrins:
	standat['intrins']=intrins

modelpickleoutput=path.join(outputdir,f'{model_name}.pickle')
print(f'model file: {codefile}')
print(f'output file: {pickleoutput}')
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
    'betarescale': np.exp(np.random.normal(0,.1)),
    'velscaling': np.exp(np.random.normal(0,.1)),
    'veldispersion_additional': np.exp(np.random.normal(np.log(200),.1)),
    'intrins': np.exp(np.random.normal(np.log(.12),.1))
}
opfit = model.optimizing(data=standat,init=initfun)
print(opfit)
if args.posteriorpredictive:
        print('Posterior produced with parameters :',{x:simmedparams[x] for x in opfit})
if not args.optimizing:
	fit = model.sampling(data=standat, iter=niter, chains=nchains,n_jobs=njobs,warmup = min(niter//2,1000),init=initfun)
	print(fit.stansummary())
	with open(pickleoutput,'wb') as picklefile: pickle.dump([model,fit,opfit],picklefile,-1)

# In[17]:



