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
parser.add_argument('--nocorrections',action='store_true',
					help='No 2M++ corrections')

					
					
parser.add_argument('--fixcorrectionparams',action='store_true',
					help='Fix 2M++ nuisance parameters to fiducial values')
parser.add_argument('--fixveldispersion',action='store_true',
					help='Fix nonlinear velocity dispersion to 250 km/s')
parser.add_argument('--intrins',type=float,default=None,
                    help='Fix intrinsic scatter to a given value')
parser.add_argument('--flatprior',action='store_true',
                    help='Use flat priors on all parameters (default is Jeffreys prior)')
parser.add_argument('--sampleprior', action='store_true', 
                    help='Sample directly from prior distribution without constraints from data')
parser.add_argument('--generatedquantities',action='store_true',
					help='Generate posterior predictive data and allow excluded data')
args = parser.parse_args()
print(' '.join(sys.argv))
fitres=args.fitres
niter,nchains=args.niter,args.nchains
outputdir=args.outputdir
redshiftcut= args.redshiftcut
fixintrins=not (args.intrins is None)
intrins=args.intrins
dataname=path.splitext(path.split(fitres)[-1])[0]
dataname+=f'_{redshiftcut[0]:.2f}_{redshiftcut[1]:.2f}'.replace('.','')



model_name=''
if fixintrins:
	model_name+='fixintrins_'
if args.fixveldispersion:
	model_name+='fixveldispersion_'
if args.fixcorrectionparams:
	model_name+='fixedcorrections_'
if args.sampleprior:
	model_name='prior_'+model_name
model_name+='individual'


pickleoutput=f'{model_name}_{dataname}.pickle'
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



define_additional_constants=''
define_additional_params=''

# In[13]:
declare_pecvel_quantities="""
	matrix[N,N] pecvelcovmarginal;
	vector[N] pecvelmeanmarginal;
"""
define_pecvel_quantities="""
	if (ntmppobs==0){
		pecvelcovmarginal=pecvelcov;
		pecvelmeanmarginal=rep_vector(0,N);
	}
	else if (ntmppobs==N){
		matrix[N,N] sumcovtransform;
		matrix[N,N] sumcovtransform_times_priorcov;
		matrix[N,N] sumcovtransform_times_tmppcov;

		sumcovtransform=cholesky_decompose(pecvelcov+diag_matrix(rep_vector(square(correctionstd),N)));
		sumcovtransform_times_priorcov=  mdivide_left_tri_low(sumcovtransform,pecvelcov);
		sumcovtransform_times_tmppcov=  mdivide_left_tri_low(sumcovtransform, diag_matrix(rep_vector(square(correctionstd),N)));
		
		pecvelmeanmarginal=sumcovtransform_times_priorcov'* mdivide_left_tri_low(sumcovtransform,velcorrections+bulkcorrections);
		pecvelcovmarginal= sumcovtransform_times_tmppcov' *sumcovtransform_times_priorcov ;
	}
	else{
		//quantities purely for calculation, no relevance
		matrix[ntmppobs,ntmppobs] sumcovtransform;
		matrix[ntmppobs,ntmppobs] sumcovtransform_times_priorcov;
		matrix[ntmppobs,ntmppobs] sumcovtransform_times_tmppcov;
		matrix[ntmppobs,N-ntmppobs] sumcovtransform_times_transfermatrix;

		sumcovtransform=cholesky_decompose(pecvelcov[tmppinds,tmppinds]+diag_matrix(rep_vector(square(correctionstd),ntmppobs)));
		sumcovtransform_times_priorcov=  mdivide_left_tri_low(sumcovtransform,pecvelcov[tmppinds,tmppinds]);
		sumcovtransform_times_tmppcov=  mdivide_left_tri_low(sumcovtransform, diag_matrix(rep_vector(square(correctionstd),ntmppobs)));
		sumcovtransform_times_transfermatrix=  mdivide_left_tri_low(sumcovtransform, pecvelcov[tmppinds,notmppinds] );

		pecvelmeanmarginal[tmppinds]=sumcovtransform_times_priorcov'* mdivide_left_tri_low(sumcovtransform,velcorrections+bulkcorrections);
		pecvelmeanmarginal[notmppinds]=rep_vector(0,N-ntmppobs);

		pecvelcovmarginal[tmppinds,tmppinds]= sumcovtransform_times_tmppcov' *sumcovtransform_times_priorcov ;
		pecvelcovmarginal[tmppinds,notmppinds]=  sumcovtransform_times_tmppcov' *sumcovtransform_times_transfermatrix ;
		pecvelcovmarginal[notmppinds,tmppinds]= pecvelcovmarginal[tmppinds,notmppinds]';
		pecvelcovmarginal[notmppinds,notmppinds]=pecvelcov[notmppinds,notmppinds]- crossprod(sumcovtransform_times_transfermatrix);
	}
"""

declare_mu_quantities="""
	vector[N] meanmuresids;
	vector[N] muresidsvar ;
	matrix[N,N] covsigmamuresids;
"""
define_mu_quantities="""
	meanmuresids=offset+(dmudvpec .* pecvelmeanmarginal);
	muresidsvar= (square(veldispersion_additional * dmudvpec) + biasscale .* (square(intrins)+square(muerr)));
	covsigmamuresids=quad_form_diag(pecvelcovmarginal,dmudvpec)+diag_matrix(muresidsvar);
"""
if fixintrins:
	define_additional_constants+=f"""
    real<lower=.01,upper=1> intrins={intrins};"""
else:
	define_additional_params+="""
    real<lower=.01,upper=1> intrins;
"""

if args.fixveldispersion:
	define_additional_constants+="""
    real<lower=10,upper=500> veldispersion_additional=200;
"""
else:
	define_additional_params+="""
    real<lower=10,upper=500> veldispersion_additional;
"""
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


if args.sampleprior:
	calculate_residuals=""
else:

	calculate_residuals="""
	//SN constraints
	muresiduals ~ multi_normal(meanmuresids[sninds],covsigmamuresids[sninds,sninds] );
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
}}

transformed data{{
    vector[N] zeros= rep_vector(0,N);
    cov_matrix[N] identity=diag_matrix(rep_vector(1.0,N));

    int notmppinds[N-ntmppobs];
    int j=1;
    int k=1;
    for (i in 1:ntmppobs){{
    	while (j<tmppinds[i]){{
    		notmppinds[k]=j;
    		k=k+1;
    		j=j+1;
    	}}
    	j=j+1;
    }}
    while (j< N+1){{
    	notmppinds[k]=j;
    	k=k+1;
    	j=j+1;
    }}
{define_additional_constants}
}}

parameters {{
    real<offset=0,multiplier=.1> offset;    
    {define_additional_params}
}}

model {{
{declare_pecvel_quantities}
{declare_mu_quantities}

{define_pecvel_quantities}
{define_mu_quantities}
   
{prior_block}
{calculate_residuals}
}}

"""
if args.generatedquantities:
	model_code+=f"""
generated quantities{{
	vector[nsnobs] mu_hat;
	vector[nsnobs_pred] mu_pred;
	real log_lik;
	real log_lik_ex;
	{{
{declare_pecvel_quantities}
{declare_mu_quantities}

{define_pecvel_quantities}
{define_mu_quantities}

		log_lik = multi_normal_lpdf( muresiduals|meanmuresids[sninds], covsigmamuresids[sninds,sninds] );
		mu_hat = multi_normal_rng(meanmuresids[sninds], covsigmamuresids[sninds,sninds] );
		if (nsnobs_pred !=0){{
			matrix[nsnobs,nsnobs] chol_sigma ;
			vector[nsnobs] invchol_muobs;
			matrix[nsnobs,nsnobs_pred] invchol_transfermatrix ;
			vector[nsnobs_pred] meanpred;
			matrix[nsnobs_pred,nsnobs_pred] sigmamupred;
			chol_sigma = cholesky_decompose( covsigmamuresids[sninds,sninds]);
			invchol_muobs= mdivide_left_tri_low(chol_sigma, muresiduals - (  meanmuresids[sninds]));
			invchol_transfermatrix = mdivide_left_tri_low(chol_sigma,  covsigmamuresids[sninds,sninds_pred]);
			
			meanpred= meanmuresids[sninds_pred] + invchol_transfermatrix' * invchol_muobs;
			sigmamupred=  covsigmamuresids[sninds_pred,sninds_pred]- crossprod(invchol_transfermatrix);
		
			log_lik_ex=multi_normal_lpdf(mu_excluded| meanpred,   sigmamupred);
			mu_pred= multi_normal_rng(meanpred, sigmamupred);
		}}
	}}
}}

"""
# generated quantities {{
#     // elementwise log likelihood
#     real log_likelihood_mu[nsnobs];
#     real log_likelihood_tmpp[ntmppobs];
#     
#     // posterior predictive
#     real mu_hat[nsnobs];
#     real tmpp_hat[ntmppobs];
#     
#     // out of sample prediction
#     real mu_pred[nsnobs_pred];
#     real tmpp_pred[ntmppobs_pred];
#     real log_likelihood_mu_pred[nsnobs_pred];
#     real log_likelihood_tmpp_pred[ntmppobs_pred];
#     {{
# 		{define_additional_model_quantities}
# 		// posterior predictive
# 		for (k in 1:nsnobs){{
# 			log_likelihood_mu[k] = normal_lpdf(muresiduals[k] | offset+(dmudvpec .* peculiarvelocities)[sninds[k]], sigmamuresids[sninds[k]]);
# 		}}
# 		mu_hat = normal_rng(offset+(dmudvpec .* peculiarvelocities)[sninds], sigmamuresids[sninds]);
# 	
# 		for (k in 1:ntmppobs){{
# 			log_likelihood_tmpp[k] = normal_lpdf(velcorrections[k]+bulkcorrections[k] | peculiarvelocities[tmppinds[k]],correctionstd);
# 		}}
# 		tmpp_hat = normal_rng(peculiarvelocities[tmppinds],correctionstd);
#  
# 		// out of sample prediction
# 		mu_pred = normal_rng(offset+(dmudvpec .* peculiarvelocities)[sninds_pred], sigmamuresids[sninds_pred]);
# 		for (k in 1:nsnobs_pred){{
# 			log_likelihood_mu_pred[k] = normal_lpdf(mu_excluded[k]|offset+(dmudvpec .* peculiarvelocities)[sninds_pred[k]], sigmamuresids[sninds_pred[k]]);
# 		}}
# 		tmpp_pred = normal_rng(peculiarvelocities[tmppinds_pred],correctionstd);
# 		for (k in 1:ntmppobs_pred){{
# 			log_likelihood_tmpp_pred[k] = normal_lpdf(tmpp_excluded[k]|peculiarvelocities[tmppinds_pred[k]],correctionstd);
# 		}}
# 	}}
# }}

codefile=path.join(outputdir,f'{model_name}.stan')
if path.exists(codefile):
	with open(codefile,'r') as file: oldcode=file.read()
	existingmodelisgood= oldcode==model_code	
else:
	existingmodelisgood=False
with open(codefile,'w') as file: file.write(model_code)

if args.nocorrections:

	hascorrection= sndata['zCMB']<0.06
	correctedindices=np.where(hascorrection)[0]
else:
	correctedindices=np.array([],dtype=int)
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
                
               'nsnobs_pred':0,
               'sninds_pred':np.array([],dtype=int),
               'mu_excluded':np.array([],dtype=float),
               'ntmppobs_pred':0,
               'tmppinds_pred':np.array([],dtype=int),
               'tmpp_excluded':np.array([],dtype=float),
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
}
opfit = model.optimizing(data=standat,init=initfun,iter=niter)
print(opfit)

if not args.optimizing:
	fit = model.sampling(data=standat, iter=niter, chains=nchains,n_jobs=njobs,warmup = min(niter//2,1000),init=initfun)
	with open(pickleoutput,'wb') as picklefile: pickle.dump([model,fit,opfit],picklefile,-1)

# In[17]:



