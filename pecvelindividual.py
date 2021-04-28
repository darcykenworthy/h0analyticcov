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
print(' '.join(sys.argv))
fitres=args.fitres
niter,nchains=args.niter,args.nchains
outputdir=args.outputdir
redshiftcut= args.redshiftcut
fixintrins=not (args.intrins is None)
intrins=args.intrins
dataname=path.splitext(path.split(fitres)[-1])[0]
dataname+=f'{redshiftcut[0]:.2f}{redshiftcut[1]:.2f}'.replace('.','')



model_name=''
if fixintrins:
	model_name+='fixintrins_'
if args.fixveldispersion:
	model_name+='fixveldispersion_'
if args.fixcorrectionparams:
	model_name+='fixedcorrections_'
if args.sampleprior:
	model_name='prior_'+model_name
else:
	if args.nocorrections:
		model_name='no2mm_'+model_name
model_name+='individual'


pickleoutput=f'{dataname}.pickle'
os.makedirs(outputdir,exist_ok=True)
pickleoutput=path.join(outputdir,pickleoutput)


if path.exists(pickleoutput) and not args.clobber and not args.optimizing:
	raise ValueError('Clobber is false and output file already exists')
# In[3]:

    
sndataFull=readFitres(fitres)
sndataFull=renameDups(sndataFull)
sndata=cutdups(sndataFull,reweight=True)
if (sndata['DEC']<-90).any():
	sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['RA']=202.303420
	sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['DEC']=11.272390

	sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['HOST_RA']=202.306840
	sndata[np.where(sndata['CID']=='SNF20080514-002')[0][0]]['HOST_DEC']=11.275820


	sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['RA']=330.454458
	sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['DEC']= 13.055306

	sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['HOST_RA']=330.456000
	sndata[np.where(sndata['CID']=='SNF20080909-030')[0][0]]['HOST_DEC']= 13.055194

	sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['RA']=3.749292
	sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['DEC']=16.335000


	sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['HOST_RA']=3.750292
	sndata[np.where(sndata['CID']=='SNF20071021-000')[0][0]]['HOST_DEC']=16.333242
	print('Warning: Special Case fixes for KAIT SNe')

z=sndata['zCMB']

cut= (z>redshiftcut[0])&(z<redshiftcut[1])
print(f'{cut.sum()} SNe in redshift range {redshiftcut}')
sndata=sndata[cut].copy()
z=sndata['zCMB']

sncoords,separation,angsep=getseparation(sndata,hostlocs=False)
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
# In[13]:
define_vpec_uncertainties="""
	matrix[N,N] pecvelcovtot=pecvelcov;
"""
define_mu_uncertainties="""
	vector[N] sigmamuresids = sqrt(square(veldispersion_additional*dmudvpec) + biasscale .* (square(intrins)+square(muerr)));
"""
if fixintrins:
	define_additional_constants+=f"""
    real<lower=.01,upper=1> intrins={intrins};"""+define_mu_uncertainties
else:
	define_additional_params+="""
    real<lower=.01,upper=1> intrins;
"""
	define_additional_model_quantities+=define_mu_uncertainties

if args.fixveldispersion:
	define_additional_constants+="""
    real<lower=10,upper=500> veldispersion_additional=200;
"""+define_vpec_uncertainties
else:
	define_additional_params+="""
    real<lower=10,upper=500> veldispersion_additional;
"""
	define_additional_model_quantities+=define_vpec_uncertainties
if args.fixcorrectionparams:
	define_additional_constants+="""
	real<lower=10,upper=1000> correctionstd=150;
"""
else:

	define_additional_params+="""
	real<lower=10,upper=1000> correctionstd;
"""

if args.flatprior:
	prior_block="""
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
	calculate_residuals="" 
else:
	if args.nocorrections:
		calculate_residuals="""
	//No 2M++ constraints
	"""
	else:
		calculate_residuals="""
    //2M++ constraints
    peculiarvelocities[tmppinds]~ normal(velcorrections+bulkcorrections,correctionstd);
""" 

	calculate_residuals+="""
	//SN constraints
	muresiduals ~ normal(offset+(dmudvpec .* peculiarvelocities)[sninds], sigmamuresids[sninds]);

"""

model_code=f"""

data {{
    int<lower=0> N; // number of SNe
    vector<lower=0>[N] muerr; 
    vector[N] biasscale;
    vector[N] zCMB;    
    vector[N] dmudvpec;
    matrix[N,N] pecvelcov;
	
	int<lower=0> nsnobs;
	int sninds[nsnobs];
    vector[nsnobs] muresiduals;
	
	int<lower=0> nsnobs_pred;
	int sninds_pred[nsnobs_pred];
	vector[nsnobs_pred] mu_excluded;

    int<lower=0> ntmppobs; // number of 2M++ local corrections
    int  tmppinds[ntmppobs]; //indices of SNe with local 2m++ corrections
    vector[ntmppobs] velcorrections;
    vector[ntmppobs] bulkcorrections;
 	
 	int<lower=0> ntmppobs_pred;
	int tmppinds_pred[ntmppobs_pred];
	vector[ntmppobs_pred] tmpp_excluded;
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
    real vextrescale = dot_product(bulkcorrections,peculiarvelocities[tmppinds])/bulkfiducial ;
}}


model {{
    
    {define_additional_model_quantities}

    {prior_block}
    {calculate_residuals}
}}

generated quantities {{
    // elementwise log likelihood
    real log_likelihood_mu[nsnobs];
    real log_likelihood_tmpp[ntmppobs];
    
    // posterior predictive
    real mu_hat[nsnobs];
    real tmpp_hat[ntmppobs];
    
    // out of sample prediction
    real mu_pred[nsnobs_pred];
    real tmpp_pred[ntmppobs_pred];
    real log_likelihood_mu_pred[nsnobs_pred];
    real log_likelihood_tmpp_pred[ntmppobs_pred];
    
    {define_additional_model_quantities}
    // posterior predictive
    for (k in 1:nsnobs){{
		log_likelihood_mu[k] = normal_lpdf(muresiduals[k] | offset+(dmudvpec .* peculiarvelocities)[sninds[k]], sigmamuresids[sninds[k]]);
    }}
	mu_hat = normal_rng(offset+(dmudvpec .* peculiarvelocities)[sninds], sigmamuresids[sninds]);
	
    for (k in 1:ntmppobs){{
		log_likelihood_tmpp[k] = normal_lpdf(velcorrections[k]+bulkcorrections[k] | peculiarvelocities[tmppinds[k]],correctionstd);
	}}
	tmpp_hat = normal_rng(peculiarvelocities[tmppinds],correctionstd);
 
    // out of sample prediction
	mu_pred = normal_rng(offset+(dmudvpec .* peculiarvelocities)[sninds_pred], sigmamuresids[sninds_pred]);
    for (k in 1:nsnobs_pred){{
		log_likelihood_mu_pred[k] = normal_lpdf(mu_excluded[k]|offset+(dmudvpec .* peculiarvelocities)[sninds_pred[k]], sigmamuresids[sninds_pred[k]]);
	}}
	tmpp_pred = normal_rng(peculiarvelocities[tmppinds_pred],correctionstd);
    for (k in 1:ntmppobs_pred){{
		log_likelihood_tmpp_pred[k] = normal_lpdf(tmpp_excluded[k]|peculiarvelocities[tmppinds_pred[k]],correctionstd);
	}}
}}
"""
codefile=path.join(outputdir,f'{model_name}.stan')
if path.exists(codefile):
	with open(codefile,'r') as file: oldcode=file.read()
	existingmodelisgood= oldcode==model_code	
else:
	existingmodelisgood=False
with open(codefile,'w') as file: file.write(model_code)

hascorrection= sndata['zCMB']<0.06
correctedindices=np.where(hascorrection)[0]
usesn=np.arange(sndata.size)
standat = {'N': sndata.size,
               'muerr': sndata['MUERR_RAW'],
               'biasscale':sndata['biasScale_muCOV'],
                'systematics': np.zeros((sndata.size,sndata.size)),
                'zCMB':sndata['zCMB'],
               'dmudvpec':dmudvpec,
               'pecvelcov':pecvelcov,
           
               'ntmppobs':  len(correctedindices),
               'tmppinds': 1+correctedindices,
               'velcorrections': sndata['VPEC_LOCAL'][correctedindices],
               'bulkcorrections': sndata['VPEC_BULK'][correctedindices],
               
               'nsnobs':len(usesn),
               'sninds':usesn+1,
                'muresiduals': muresiduals[usesn],
                
               'nsnobs_pred':1,
               'sninds_pred':np.array([1],dtype=int),
               'mu_excluded':np.array([0],dtype=float),
               'ntmppobs_pred':1,
               'tmppinds_pred':np.array([1],dtype=int),
               'tmpp_excluded':np.array([0],dtype=float),
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
	with open(pickleoutput,'wb') as picklefile: pickle.dump([model,fit,opfit],picklefile,-1)

# In[17]:



