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
parser.add_argument('--velocitycov',type=str,default=None,
                    help='.npy file containing peculiar velocity covariance matrix')
parser.add_argument('--systematics',type=str,default=None,nargs=2,
					help="File identifying CID's and file with systematic covariance matrix")

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
parser.add_argument('--simulateddata',nargs=2,type=str,default=None,
					help='file with simulated data vectors and the index to use for this run')
parser.add_argument('--nocorrections',action='store_true',
					help='No 2M++ corrections')

parser.add_argument('--tdist',action='store_true',
					help='Use multidimensional t-distributions for SN Ia residuals rather than normal')
parser.add_argument('--separatenonlinear',action='store_true',
					help='Use description of nonlinear scaling from provided peculiar velocity covariance')
parser.add_argument('--correlatetmpp',action='store_true',
					help='Do not assume 2M++ errors are uncorrelated')
parser.add_argument('--fixcorrectionparams',action='store_true',
					help='Fix 2M++ nuisance parameters to fiducial values')
parser.add_argument('--fixveldispersion',type=float,default=None,
					help='Fix nonlinear velocity dispersion to given value (km/s)')
parser.add_argument('--intrins',type=float,default=None,
                    help='Fix intrinsic scatter to a given value (mag)')
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
data_name=path.splitext(path.split(fitres)[-1])[0]
usesystematics=not (args.systematics is None)

if args.velocitycov:
	data_name+=f'_{path.splitext(path.split(args.velocitycov)[-1])[0]}'
	pecvelcovfile=args.velocitycov
else:
	pecvelcovfile='velocitycovariance-{}-darksky_class.npy'.format(path.splitext(path.split(fitres)[-1])[0])
if args.nocorrections:
	data_name+='_no2mm'
if usesystematics:
	data_name+='_systematics'
data_name+=f'_{redshiftcut[0]:.2f}_{redshiftcut[1]:.2f}'.replace('.','')


if args.simulateddata:
	simfile=args.simulateddata[0]
	simindex=int(args.simulateddata[1])
	data_name+=f"_simulated_{simindex}"
	with open(simfile,'rb') as file: 
		simdata= pickle.load(file)[1]
model_name=''
if args.tdist:
	model_name+='tdist_'
if args.separatenonlinear:
	model_name+='separatenonlinear_'
if args.correlatetmpp:
	model_name+='correlatetmpp_'
if fixintrins:
	model_name+='fixintrins_'
if not (args.fixveldispersion is  None):
	model_name+='fixveldispersion_'
if args.fixcorrectionparams:
	model_name+='fixedcorrections_'
if args.sampleprior:
	model_name+='prior_'+model_name
if args.generatedquantities:
	model_name+='generated_quantities_'
model_name+='individual'


os.makedirs(outputdir,exist_ok=True)
pickleoutput=path.join(outputdir,f'output_{model_name}_{data_name}.pickle')
extractoutput=path.join(outputdir,f'extract_{model_name}_{data_name}.pickle')
codefile=path.join(outputdir,f'code_{model_name}.stan')
modelpickleoutput=path.join(outputdir,f'compiled_{model_name}.pickle')
datapickleoutput=path.join(outputdir,f'data_{data_name}.pickle')

if path.exists(pickleoutput) and not args.clobber and not args.optimizing:
	raise ValueError('Clobber is false and output file already exists')
# In[3]:

    
sndata=readFitres(fitres)


if usesystematics:
	passedsystematics=np.genfromtxt(args.systematics[0],dtype=[int,'U40',float,float,float],names=True)
	covmatfile=np.genfromtxt(args.systematics[1])
	covmatsize,covmat=int(covmatfile[0]),covmatfile[1:]
	covmat=covmat.reshape((covmatsize,covmatsize))
	
	listsbysurvcid=[[f"{x['IDSURVEY']:.0f}_{x['CID']}" for x in y] for y in [sndata,passedsystematics]]
	cut = np.array([x in listsbysurvcid[1] for x in listsbysurvcid[0]])
	print(f"SNe missing from systematics: {', '.join(sndata['CID'][~cut])}")
	sndata=sndata#[cut]
	indexmapping=[listsbysurvcid[1].index(x) for x in (listsbysurvcid[0]) if x in listsbysurvcid[1]]
	systematics=covmat[indexmapping,:][:,indexmapping]
else:
	cut=np.ones(sndata.size,dtype=bool)
	systematics=np.zeros((sndata.size,sndata.size))
	
pecvelcov=np.load(pecvelcovfile)
if pecvelcov.ndim==3:
	if args.separatenonlinear:
		pecvelcov,nonlinear=pecvelcov[cut,:,:][:,cut,:].T
		nonlinear=checkposdef(nonlinear)
	else:
		pecvelcov=pecvelcov.sum(axis=-1)[cut,:][:,cut]
else:
	if args.separatenonlinear:
		raise ValueError("No separate nonlinear covariance available")
	pecvelcov=pecvelcov[cut,:][:,cut]
pecvelcov=checkposdef(pecvelcov)

sndata=sndata[cut].copy()
sndata=renameDups(sndata)
#,dupscut=cutdups(sndataFull,reweight=False,returninds=True)
#systematics=systematics[dupscut,:][:,dupscut]
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
sndata=sndata[cut].copy()
systematics=systematics[cut,:][:,cut]
pecvelcov=pecvelcov[cut,:][:,cut]
if args.separatenonlinear:
	nonlinear=nonlinear[cut,:][:,cut]
#cut=cut&redshiftcut
print(f'{cut.sum()} SNe in redshift range {redshiftcut}')
z=sndata['zCMB']
sncoords,snpos,sep,angsep=getpositions(sndata,hostlocs=False)
sndata=separatevpeccontributions(sndata,sncoords)

dupsmatrix=(sndata['CID'][np.newaxis,:]==sndata['CID'][:,np.newaxis])*1.

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
if args.simulateddata:
	muresiduals=simdata['mu_hat'][simindex]
	print('WARNING: SIMULATED DATA')
	print('True values:',', '.join([f'{key} {simdata[key][simindex]:.3f}' for key in simdata.constrained_param_names() if simdata[key].ndim==1]))
else:
	muresiduals=cosmo.distmod(sndata['zCMB']).value - sndata['MU']
# In[10]:


dmudvpec=5/np.log(10)*((1+z)**2/(cosmo.efunc(z)*cosmo.H0*cosmo.luminosity_distance(z))).to(u.s/u.km).value
velocityprefactor=np.outer(dmudvpec,dmudvpec)

	
	
define_additional_priors=''
define_additional_constants=''
define_additional_params=''
define_additional_data=''
# In[13]:
declare_pecvel_quantities="""
	matrix[N,N] pecvelcovmarginal;
	vector[N] pecvelmeanmarginal;
"""
define_pecvel_quantities="""
	matrix[N,N] cosmoprior = square(sigmarescale)*pecvelcov;
"""

if args.separatenonlinear:
	define_additional_data+="""
	matrix[N,N] nonlinear;
"""
	define_pecvel_quantities+="""
	matrix[N,N] nonlinearfinal = square(nlrescale)*nonlinear;
"""
	define_additional_params+="""
	real<lower=.01,upper=10> nlrescale;
"""
	define_additional_priors+="""
	nlrescale~lognormal(0,.5);
"""
	if not args.fixveldispersion is None:
		raise ValueError("Veldispersion specified with separate nonlinear matrix")
else:
	if args.fixveldispersion is None:
		define_additional_params+="""real<lower=10,upper=500> veldispersion_additional;
"""
		define_additional_priors+="""
		veldispersion_additional ~ lognormal(log(250),.5);
"""
	else:
		define_additional_constants+=f"""
		real veldispersion_additional={args.fixveldispersion};
	"""
	define_pecvel_quantities+="""
	matrix[N,N] nonlinearfinal =square(veldispersion_additional)*dupsmatrix;// diag_matrix(rep_vector(square(veldispersion_additional),N));
"""

if args.correlatetmpp:
	define_additional_params+="""
	real<lower=0,upper=200> tmppcorscale;
"""
	define_pecvel_quantities+="""
	matrix[ntmppobs,ntmppobs] tmppcov= cov_exp_quad( positions[tmppinds],correctionstd, tmppcorscale);

"""
else: 
	define_pecvel_quantities+="""
	matrix[ntmppobs,ntmppobs] tmppcov= diag_matrix(rep_vector(square(correctionstd),ntmppobs));
"""

define_pecvel_quantities+="""
	vector[ntmppobs] tmppmeasuredvel=betarescale*velcorrections+vextrescale*bulkcorrections;
	if (ntmppobs==0){
		pecvelcovmarginal=cosmoprior;
		pecvelmeanmarginal=rep_vector(0,N);
	}
	else if (ntmppobs==N){
		matrix[N,N] sumcovtransform;
		matrix[N,N] sumcovtransform_times_priorcov;
		matrix[N,N] sumcovtransform_times_tmppcov;

		sumcovtransform=cholesky_decompose(cosmoprior+tmppcov);
		sumcovtransform_times_priorcov=  mdivide_left_tri_low(sumcovtransform,cosmoprior);
		sumcovtransform_times_tmppcov=  mdivide_left_tri_low(sumcovtransform, tmppcov);
		
		pecvelmeanmarginal=sumcovtransform_times_priorcov'* mdivide_left_tri_low(sumcovtransform,tmppmeasuredvel);
		pecvelcovmarginal= sumcovtransform_times_tmppcov' *sumcovtransform_times_priorcov ;
		//Symmetrise, try to make sure pos def
		pecvelcovmarginal=(pecvelcovmarginal'+pecvelcovmarginal)/2+diag_matrix(rep_vector(1,N));
	}
	else{
		//quantities purely for calculation, no relevance
		matrix[ntmppobs,ntmppobs] sumcovtransform;
		matrix[ntmppobs,ntmppobs] sumcovtransform_times_priorcov;
		matrix[ntmppobs,ntmppobs] sumcovtransform_times_tmppcov;
		matrix[ntmppobs,N-ntmppobs] sumcovtransform_times_transfermatrix;
		vector[ntmppobs] sumcovtransform_times_tmppmeasuredvel;
		
		sumcovtransform=cholesky_decompose(cosmoprior[tmppinds,tmppinds]+tmppcov);
		sumcovtransform_times_priorcov=  mdivide_left_tri_low(sumcovtransform,cosmoprior[tmppinds,tmppinds]);
		sumcovtransform_times_tmppcov=  mdivide_left_tri_low(sumcovtransform, tmppcov);
		sumcovtransform_times_transfermatrix=  mdivide_left_tri_low(sumcovtransform, cosmoprior[tmppinds,notmppinds] );
		sumcovtransform_times_tmppmeasuredvel=mdivide_left_tri_low(sumcovtransform,tmppmeasuredvel);
		
		pecvelmeanmarginal[tmppinds]=sumcovtransform_times_priorcov'* sumcovtransform_times_tmppmeasuredvel;
		pecvelmeanmarginal[notmppinds]= sumcovtransform_times_transfermatrix'*sumcovtransform_times_tmppmeasuredvel ;

		pecvelcovmarginal[tmppinds,tmppinds]= sumcovtransform_times_tmppcov' *sumcovtransform_times_priorcov ;
		pecvelcovmarginal[tmppinds,notmppinds]=  sumcovtransform_times_tmppcov' *sumcovtransform_times_transfermatrix ;
		pecvelcovmarginal[notmppinds,tmppinds]= pecvelcovmarginal[tmppinds,notmppinds]';
		pecvelcovmarginal[notmppinds,notmppinds]=cosmoprior[notmppinds,notmppinds]- crossprod(sumcovtransform_times_transfermatrix);
		//Symmetrise, try to make sure pos def
		pecvelcovmarginal=(pecvelcovmarginal'+pecvelcovmarginal)/2+diag_matrix(rep_vector(1,N));
	}
	pecvelcovmarginal+=nonlinearfinal;
"""

declare_mu_quantities="""
	vector[N] meanmuresids;
	matrix[N,N] covsigmamuresids;
"""
define_mu_quantities="""
	meanmuresids=offset+(dmudvpec .* pecvelmeanmarginal);
	covsigmamuresids=quad_form_diag(pecvelcovmarginal,dmudvpec)+square(intrins)*dupsmatrixscaled+muvarscaled +systematics;
"""

if args.tdist:
	extradistparam='nu,'
	distname='multi_student_t'
	varprefactor='nu/(nu-2)*'
	define_additional_params+="""
	real <lower=2,upper=20> nu;
"""
	define_additional_priors+="""
	nu ~ gamma(4,.5);
"""
	define_log_lik_pointwise="""
		inversesigma= inverse_spd(covsigmamuresids[sninds,sninds]);
		pointwisedegree= nu + nsnobs-1;
		residualsfinal=muresiduals-meanmuresids[sninds];
		invsig_residuals=mdivide_left_spd(covsigmamuresids[sninds,sninds],residualsfinal );
THIS DON'T WORK YET
		for (i in 1:nsnobs){{
			pointwisemumean =  muresiduals -  1/inversesigma[i,i] *invsig_residuals[i] ;
			collessthisindex=append_col( head( col(inversesigma,i) , i-1), tail( col(inversesigma,i) , nsnobs- (i)))
			inversesigma - collessthisindex*collessthisindex' / inversesigma[i,i]
			log_lik_pointwise[i]=normal_lpdf( muresiduals[i] | pointwisemumean[i],sqrt(pointwisemuvar[i]));
		}}
"""
else:
	extradistparam=''
	distname='multi_normal'
	varprefactor=''
	define_log_lik_pointwise="""
		pointwisemuvar=  1 ./ diagonal(inverse_spd(covsigmamuresids[sninds,sninds]));
		pointwisemumean =  muresiduals -  pointwisemuvar .* mdivide_left_spd(covsigmamuresids[sninds,sninds], muresiduals-meanmuresids[sninds]);
		for (i in 1:nsnobs){{
			log_lik_pointwise[i]=normal_lpdf( muresiduals[i] | pointwisemumean[i],sqrt(pointwisemuvar[i]));
		}}
"""

if fixintrins:
	define_additional_constants+=f"""real<lower=.01,upper=1> intrins={intrins};
"""
else:
	define_additional_params+="""real<lower=.01,upper=1> intrins;
"""

if args.fixcorrectionparams:
	define_additional_constants+="""real<lower=10,upper=1000> correctionstd=150;
"""
else:

	define_additional_params+="""real<lower=10,upper=1000> correctionstd;
"""
	define_additional_priors+="""
	//2m++ prior (based on N body simulations)
	//correctionstd ~ gamma(75,2);
	correctionstd ~ lognormal(log(150),.3);
"""

if args.flatprior:
	prior_block="""
"""
else:
	
	prior_block=f"""
	//SN priors
	offset ~ normal(0,.5);
	intrins ~ lognormal(log(0.1),.5);
	
	//Carrick2015 uncertainties
	vextrescale ~ normal(1,vextfracerr);
	betarescale ~ normal(1,betafracerr);

	sigmarescale~lognormal(0,.5);
{define_additional_priors}
"""
	if args.correlatetmpp:
		prior_block+="""tmppcorscale~lognormal(log(10),1);
"""
if args.sampleprior:
	calculate_residuals=""
else:

	calculate_residuals=f"""
	//SN constraints
	muresiduals ~ {distname}({extradistparam}meanmuresids[sninds],{varprefactor}covsigmamuresids[sninds,sninds] );
"""

model_code=f"""
data {{
    int<lower=0> N; // number of SNe
    vector<lower=0>[N] muerr; 
    vector[N] biasscale;
    vector[N] zCMB;    
    vector[N] dmudvpec;
    matrix[N,N] pecvelcov;
	vector[3] positions[N];
	matrix[N,N] systematics;
	matrix[N,N] dupsmatrix;
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
{define_additional_data}
}}

transformed data{{
    vector[N] zeros= rep_vector(0,N);
    cov_matrix[N] identity=diag_matrix(rep_vector(1.0,N));
	real vextfracerr= 23./159;
	real betafracerr= 0.021/0.431; 
	matrix[N,N] dupsmatrixscaled=quad_form_diag(dupsmatrix,sqrt(biasscale));
	matrix[N,N] muvarscaled=  diag_matrix(biasscale .* (square(muerr)));

    int notmppinds[N-ntmppobs];
{define_additional_constants}

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
}}

parameters {{
    real<offset=0,multiplier=.1> offset;    
    {define_additional_params}
	real<offset=0,multiplier=1> betarescale;
	real<offset=0,multiplier=1> vextrescale;
	real<lower=.01,upper=10> sigmarescale;
}}

model {{
{declare_mu_quantities}
{declare_pecvel_quantities}
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
	vector[nsnobs] log_lik_pointwise;
	{{
		vector[nsnobs] pointwisemuvar;
		vector[nsnobs] pointwisemumean;
{declare_mu_quantities}
{declare_pecvel_quantities}
{define_pecvel_quantities}
{define_mu_quantities}
		log_lik = {distname}_lpdf( muresiduals|{extradistparam}meanmuresids[sninds],{varprefactor} covsigmamuresids[sninds,sninds] );
		mu_hat = {distname}_rng({extradistparam}meanmuresids[sninds],{varprefactor} covsigmamuresids[sninds,sninds] );
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
{define_log_lik_pointwise}	}}
}}

"""


if path.exists(codefile):
	with open(codefile,'r') as file: oldcode=file.read()
	existingmodelisgood= oldcode==model_code	
else:
	existingmodelisgood=False
with open(codefile,'w') as file: file.write(model_code)

if args.nocorrections:
	correctedindices=np.array([],dtype=int)
else:
	hascorrection= sndata['zCMB']<0.06
	correctedindices=np.where(hascorrection)[0]

usesn=np.arange(sndata.size)
standat = {'N': sndata.size,
               'muerr': sndata['MUERR_RAW'],
               'biasscale':sndata['biasScale_muCOV'],
                'systematics': systematics,
                'zCMB':sndata['zCMB'],
                'dmudvpec':dmudvpec,
                'pecvelcov':pecvelcov,
           		'positions':snpos,
           		'dupsmatrix':dupsmatrix,
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
if args.separatenonlinear:
	standat['nonlinear']=nonlinear
if fixintrins:
	standat['intrins']=intrins

with open(datapickleoutput,'wb') as file: pickle.dump(standat,file,-1) 
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
	with open(extractoutput,'wb') as picklefile: pickle.dump(fit.extract(),picklefile,-1)
# In[17]:



