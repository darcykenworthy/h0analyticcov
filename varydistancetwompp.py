#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import constants,units as u
from scipy import stats
from utilfunctions import *
from tqdm import tqdm
import stan
import argparse
import csv
from os import path
import logging
import io
import pickle

def wrapinterp(interp):
	def __wrapinterp__(interp,x):
		try:
			val= interp(x)
			if val.size==1:
				return val[0]
			return val
		except:
			return np.nan
	return (lambda x: __wrapinterp__(interp,x))


distinterpdelt=4
npointsinterp=20
rho=1
def readandshapenumpyfile(densityfile,velocityfile):
	density=np.load(densityfile)
	vel=np.load(velocityfile)
	return density,vel
def readandshapeasciifile(densityfile,velocityfile):
	size=201
	density=np.genfromtxt(densityfile).reshape((size,size,size))
	vel=np.genfromtxt(velocityfile).T.reshape((3,size,size,size))
	return density,vel
	
	
reconstructionsbysource={'Carrick15':(
	('twompp_density.npy','twompp_velocity.npy'), 400/256.,257, SkyCoord(l=304,b=6,frame='galactic',unit=u.deg), 
	{'betafiducial':.431,'betaerr':.021, 'vextfiducial':159,'vexterr':23,
	'vpecnonlindisp':150}),
	'Lilow21':(
	('cartesian_grid_density_zCMB.dat','cartesian_grid_velocity_zCMB.dat'), 2,201,SkyCoord(l=298,b=12,frame='galactic',unit=u.deg), 
	{'betafiducial':0.362,'betaerr':0.050, 'vextfiducial':209,'vexterr':59,
	'vpecnonlindisp':150})}


class PeculiarVelocityReconstruction:
	def __init__(self,datapaths,resolution,numpointsperaxis,dipolecoord,fixedparams):
		if path.splitext(datapaths[0])[-1]=='.npy':
			self.density,self.vel=readandshapenumpyfile(*datapaths)
		else:
			self.density,self.vel=readandshapeasciifile(*datapaths)		
		self.dipolecoord=dipolecoord
		self.fixedparams=fixedparams
		
		x=(np.arange(0,numpointsperaxis)-numpointsperaxis//2)*resolution
		y,z=x.copy(),x.copy()

		dist=np.sqrt(x[:,np.newaxis,np.newaxis]**2+ y[np.newaxis,:,np.newaxis]**2+z[np.newaxis,np.newaxis,:]**2)
		dist[x.size//2,y.size//2,z.size//2]=1e-4
		self.losvel=((self.vel[0]*x[:,np.newaxis,np.newaxis]) + (self.vel[1]*y[np.newaxis,:,np.newaxis]) + (self.vel[2]*z[np.newaxis,np.newaxis,:]))/dist

		losinterp=interpolate.RegularGridInterpolator((x,y,z),self.losvel)
		densityinterp=interpolate.RegularGridInterpolator((x,y,z),self.density)
		self.losinterp=wrapinterp(losinterp)
		self.densityinterp=wrapinterp(densityinterp)

# l,lerr = 304 , 11
# b,berr = 6,13
# In[2]:
parser = argparse.ArgumentParser(description='Calculate parameter covariance from input calibration parameter priors and FITRES calibration variants ')
parser.add_argument('fitres',type=str, 
                    help='Fitres file with RA, DEC, zCMB')
parser.add_argument('outputfile',type=str, 
                    help='Output csv')
parser.add_argument('--reconstructionsource',type=str, default='Carrick15',
                    help='Options are Carrick15 or Lilow21')
parser.add_argument('--maxsn',type=int, default=None,
                    help="Don't fit more than this many SNe")

args = parser.parse_args()
print(f'Loading {args.reconstructionsource}')
reconstruction=PeculiarVelocityReconstruction(*reconstructionsbysource[args.reconstructionsource])

#outputfile=path.splitext(args.fitres)[0]+'_VPEC_DISTMARG.csv'
print('Loading SN data')
fr=readFitres(args.fitres)
fr=renameDups(fr)
fr,dupinds=cutdups(fr,returninds=True)
sncoords,snpos,separation,angsep= getpositions(fr,hostlocs=False)


velonlycode="""
data {
	int numsn;
    int nintdist;
    
    //observed quantities
    vector[numsn] zobs; //Observed CMB frame redshift
    vector[numsn] zerr; //Errors in CMB frame redshift
    vector[numsn] dotproddipole; //dot product with 2M++ dipole vector
    
    //Quantities for interpolation of 2M++ grid
    real rho; //scale for GP interpolation
    real losdistancedelta; //difference in distance between points on the interpolation line
    vector[numsn] losdistancezero; //zeropoint distance for each SN
    matrix[numsn,nintdist] losvelocities; //LOS Velocities along line of sight for interpolation
    matrix[numsn,nintdist] losdensities ;//Densities along line of sight for interpolation
    
    //Cosmological params
    real vpecnonlindisp; //Uncertainty in nonlinear peculiar velocity
    real c;
    real q;
    real j;
    
    real vextfiducial;
    real betafiducial;
    real vexterr;
    real betaerr;
}
transformed data{
	vector[nintdist] zpecdesignmatrix[numsn];
	vector[nintdist] densitydesignmatrix[numsn];
	
	real numrange[nintdist];
	for (i in 1:nintdist){
		numrange[i]=i;
	}
    {      
		matrix[numsn,nintdist] temp;
		matrix[nintdist, nintdist] K=gp_exp_quad_cov(numrange, 1., rho);
		temp = mdivide_left_spd(K, (losvelocities/c)')';
		for (i in 1:numsn){zpecdesignmatrix[i]=temp[i]'; }
		temp = mdivide_left_spd(K, losdensities')';
		for (i in 1:numsn){densitydesignmatrix[i]=temp[i]'; }
    }  	
}
parameters{
    vector<lower=1,upper=1+nintdist>[numsn]realinterpindex;
    real<offset=0,multiplier=1> betarescale;
	real<offset=0,multiplier=100> vext;

}
transformed parameters{
    vector[numsn] latdistance=(realinterpindex-1)*losdistancedelta + losdistancezero;
    vector[numsn] distdimensionless= latdistance*100/c;
	vector[numsn] zcosm =   distdimensionless .*(1+ ( q + 1) *distdimensionless / 2 + ( j + 2*q + 1) *square(distdimensionless )/6   ) ;
    vector[numsn] zpecvel ;
    vector[numsn] densitycontrast ;
	{
		matrix[nintdist,numsn] k_x1_x2 =  gp_exp_quad_cov(to_array_1d(realinterpindex),numrange ,1,rho);
		for (i in 1:numsn){
			zpecvel[i]  = (vext-vextfiducial)/c*dotproddipole[i] + betarescale*(k_x1_x2[i] * zpecdesignmatrix[i]);
			densitycontrast[i]=(k_x1_x2[i] * densitydesignmatrix[i]);
		}
	}	

}
model{
	betarescale ~ normal(1,betaerr/betafiducial);
	vext ~ normal(vextfiducial,vexterr);
	zpecvel ~ normal( (1+zobs)./(1+zcosm) -1 , sqrt( square(vpecnonlindisp/c)+square(zerr ./(1+zcosm) ) ));
    target+=log(1+densitycontrast);
}
generated quantities{
	vector[numsn] zobs_hat;
	{	
		vector[numsn] zpecvel_hat =to_vector( normal_rng( zpecvel, sqrt( square(vpecnonlindisp/c)+square(zerr ./(1+zcosm) ) )));
		zobs_hat=(zpecvel_hat +1).*(1+zcosm) -1;
	}	
}
"""

c=constants.c.to(u.km/u.s).value 
locs,samples,scales,warnings=[],[],[],[]

losinterppoints=np.empty((fr.size, npointsinterp))
densityinterppoints=np.empty((fr.size,npointsinterp))
distzero= np.empty(fr.size)

for i,sn in tqdm(enumerate(fr),total=fr.size):
	unitvector= sncoords[i].galactic.cartesian
	unitvector=np.array([unitvector.x,unitvector.y,unitvector.z])
	#unitvector= np.array([np.cos(unitvector.b)*np.cos(unitvector.l),np.cos(unitvector.b)*np.sin(unitvector.l),np.sin(unitvector.b)])
	distfid=sn['zCMB']*c/100
	distzero[ i]=max(0,distfid-distinterpdelt*npointsinterp//2)
	losinterppoints[i]=np.array([reconstruction.losinterp((distzero[i]+distinterpdelt*j )*unitvector) for j in range(npointsinterp)])
	densityinterppoints[i]=np.array([reconstruction.densityinterp((distzero[i]+distinterpdelt*j )*unitvector) for j in range(npointsinterp)])

intmmvolumecut= ~np.isnan(losinterppoints).any(axis=1)
print(intmmvolumecut.sum())
if not (args.maxsn is  None) and intmmvolumecut.sum()>args.maxsn:
	print(f'Cutting to {args.maxsn} SNe' )
	cutindices=np.random.choice(np.where(intmmvolumecut)[0],size=args.maxsn,replace=False)
	intmmvolumecut=np.zeros(fr.size,dtype=bool)
	intmmvolumecut[cutindices]=True

datadictionary={
	'numsn': int(intmmvolumecut.sum()),
	 'zobs':  fr['zCMB'][intmmvolumecut],
	 'zerr': fr['zCMBERR'][intmmvolumecut],
	 'dotproddipole': np.cos(sncoords[intmmvolumecut].separation(reconstruction.dipolecoord)).value,

	 'rho': rho,
	 'nintdist': (npointsinterp),
	 'losdistancezero': distzero[intmmvolumecut],
	 'losdistancedelta':distinterpdelt ,
	 'losvelocities': losinterppoints[intmmvolumecut],
	 'losdensities':densityinterppoints[intmmvolumecut],
	 
	 'c':c ,
	 'q': -0.55,
	 'j':1,
}
datadictionary.update(reconstruction.fixedparams)

print('Compiling Stan model')

velonlymodel=stan.build(velonlycode,datadictionary)

print('Sampling from Stan model')
velsample=velonlymodel.sample(init= [{'realinterpindex':np.zeros(intmmvolumecut.sum())+npointsinterp//2}]*4)
datadictionary['fitres']=fr
datadictionary['volumecut']=intmmvolumecut
with open(args.outputfile.replace('.csv','.pickle'),'wb') as file:
	pickle.dump([datadictionary,velsample],file)
	
# locs,scales= np.mean( velsample['zcosm'],axis=1), np.std(velsample['zcosm'], axis=1)
# with open(args.outputfile,'w') as file:
# 	writer=csv.writer(file,delimiter=',')
# 	writer.writerow(['CID','IDSURVEY','zCMB','zHD', 'zHDERR'])
# 	for i in (range(fr[intmmvolumecut].size),total=fr[intmmvolumecut].size):
# 			if np.isnan(locs[i]): continue
# 			writer.writerow([cid[i],idsurvey[i],fr[intmmvolumecut]['zCMB'][i], locs[i],scales[i]])

pecvelcov=np.load('velocitycovariance-FITOPT000_MUOPT000-darksky_class.npy')
pecvelcov=pecvelcov[dupinds,:][:,dupinds]+reconstruction.fixedparams['vpecnonlindisp']**2*np.diag(np.ones(fr.size))
pvinf = datadictionary['c']*((1+fr[intmmvolumecut]['zCMB'][:,np.newaxis] )/(velsample['zcosm']+1)-1)
pvinfcov= np.cov(pvinf)

cholpecvelorig=linalg.cholesky(pecvelcov[intmmvolumecut,:][:,intmmvolumecut],lower=True)
regressioncoeffs=linalg.solve_triangular(cholpecvelorig,pecvelcov[intmmvolumecut,:][:,~intmmvolumecut],lower=True)
regressioncoeffs=linalg.solve_triangular(cholpecvelorig.T,regressioncoeffs,lower=False)

finalvcov=np.empty(pecvelcov.shape)
finalvcov[np.outer(intmmvolumecut,intmmvolumecut)] =pvinfcov.flatten()
finalvcov[np.outer(intmmvolumecut,~intmmvolumecut)] =np.dot(regressioncoeffs.T,pvinfcov).flatten()
finalvcov[np.outer(~intmmvolumecut,intmmvolumecut)] =np.dot(regressioncoeffs.T,pvinfcov).T.flatten()
finalvcov[np.outer(~intmmvolumecut,~intmmvolumecut)]= (pecvelcov[~intmmvolumecut,:][:,~intmmvolumecut]- np.linalg.multi_dot((regressioncoeffs.T ,(pvinfcov- pecvelcov[intmmvolumecut,:][:,intmmvolumecut]),regressioncoeffs))).flatten()

finalvmean=np.empty(fr.size)
finalvmean[intmmvolumecut]=np.mean(pvinf,axis=1)
finalvmean[~intmmvolumecut]=np.dot(regressioncoeffs.T,np.mean(pvinf,axis=1))
#'VPECS_ILOS_COVARIANCE_v0.1.csv'
cid,idsurvey,zcmb=fr['CID'],fr['IDSURVEY'],fr['zCMB']
with tqdm(total=fr.size**2) as pbar:
	with open(args.outputfile,'w') as file:
		writer=csv.writer(file,delimiter=',')
		writer.writerow(['CID_1','zCMB_1','VPEC_1','CID_2','zCMB_2','VPEC_2','VPEC_COVARIANCE'])
		for i in range(fr.size):
			for j in range(fr.size):
				pbar.update()
				writer.writerow([cid[i],zcmb[i],finalvmean[i],cid[j],zcmb[j],finalvmean[j],finalvcov[i,j]])


