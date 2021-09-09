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

# In[2]:
parser = argparse.ArgumentParser(description='Calculate parameter covariance from input calibration parameter priors and FITRES calibration variants ')
parser.add_argument('fitres',type=str, 
                    help='Fitres file with RA, DEC, zCMB')
parser.add_argument('outputfile',type=str, 
                    help='Output csv')

args = parser.parse_args()
#outputfile=path.splitext(args.fitres)[0]+'_VPEC_DISTMARG.csv'
print('Loading 2M++ and SN data')
fr=readFitres(args.fitres)
fr=renameDups(fr)
fr=fr[:100]
dupsmatrix=(fr['CID'][np.newaxis,:]==fr['CID'][:,np.newaxis])*1.

vel=np.load('twompp_velocity.npy')
density=np.load('twompp_density.npy')

# In[3]:

x=(np.arange(0,257)-128.)*400/256.
y,z=x.copy(),x.copy()

densityinterp=interpolate.RegularGridInterpolator((x,y,z),density)

dist=np.sqrt(x[:,np.newaxis,np.newaxis]**2+ y[np.newaxis,:,np.newaxis]**2+z[np.newaxis,np.newaxis,:]**2)
dist[128,128,128]=1e-4

losvel=((vel[0]*x[:,np.newaxis,np.newaxis]) + (vel[1]*y[np.newaxis,:,np.newaxis]) + (vel[2]*z[np.newaxis,np.newaxis,:]))/dist
losinterp=interpolate.RegularGridInterpolator((x,y,z),losvel)

losinterp=wrapinterp(losinterp)
densityinterp=wrapinterp(densityinterp)

# functions {
#     int intFloor(int leftStart, int rightStart, real iReal)
#     {
#       // This is absurd. Use bisection algorithm to find int floor.
#       int left;
#       int right;
# 
#       left = leftStart;
#       right = rightStart;
# 
#       while((left + 1) < right) {
#         int mid;
#         // print("left, right, mid, i, ", left, ", ", right, ", ", mid, ", ", iReal);
#         mid = left + (right - left) / 2;
#         if(iReal < mid) {
#           right = mid;
#         }
#         else {
#           left = mid;
#         }
#       }
#       return left;
#     }
#     // Interpolate arr using a non-integral index i
#     // Note: 1 <= i <= length(arr)
#     vector interpolateLinear(matrix arr, vector i)
#     {
#       int numinterppoints=dims(arr)[2];
#       int isize=dims(i)[1];
#       int iLeft;
#       int iRight;
#       vector[isize] valLeft;
#       vector[isize] valRight;
# 
#       // Get i, value at left. If exact time match, then return value.
#       for (idx in 1:isize){
#       		print(i[idx]);
# 		  iLeft = intFloor(1, numinterppoints, i[idx]);
# 		  // Get i, value at right.
# 		  iRight  = iLeft + 1;
# 		  valLeft[idx] = arr[idx, iLeft];
# 		  valRight[idx] = arr[idx,iRight];
#       }
# 
#       // Linearly interpolate between values at left and right.
#       return valLeft + (valRight - valLeft) .* (i - iLeft);
#     }
# }
velonlycode="""
data {
	int numsn;
    int nintdist;
    
    //observed quantities
    vector[numsn] zobs; //Observed CMB frame redshift
    vector[numsn] zerr; //Errors in CMB frame redshift
    vector[numsn] muobs;
	vector<lower=0>[numsn] muerr; 
	vector[numsn] biasscale;
	matrix[numsn,numsn] dupsmatrix;


    //Quantities for interpolation of 2M++ grid
    real rho; //scale for GP interpolation
    real losdistancedelta; //difference in distance between points on the interpolation line
    vector[numsn] losdistancezero; //zeropoint distance for each SN
    matrix[numsn,nintdist] losvelocities; //LOS Velocities along line of sight for interpolation
    matrix[numsn,nintdist] losdensities ;//Densities along line of sight for interpolation
    
    //Cosmological params
    real c;
    real j;
}
transformed data{
	vector[nintdist] zpecdesignmatrix[numsn];
	vector[nintdist] densitydesignmatrix[numsn];
	real numrange[nintdist];
	matrix[numsn,numsn] dupsmatrixscaled=quad_form_diag(dupsmatrix,sqrt(biasscale));
	matrix[numsn,numsn] muvarscaled=  diag_matrix(biasscale .* (square(muerr)));
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
    real<lower=0> sigint;
    real<lower=0> vpecnonlindisp; //Uncertainty in nonlinear peculiar velocity
    real<multiplier=.1> offset;
    real<lower=0, upper=1> OmegaM;

}
transformed parameters{
	real q = 3*OmegaM/2 -1;
    vector[numsn] latdistance=(realinterpindex-1)*losdistancedelta + losdistancezero;
	vector[numsn] zcosm  ;
    vector[numsn] zpecvel ;
    vector[numsn] densitycontrast ;
	vector[numsn] mu;
	{
		matrix[nintdist,numsn] k_x1_x2 =  gp_exp_quad_cov(to_array_1d(realinterpindex),numrange ,1,rho);
    	vector[numsn] distdimensionless= latdistance*100/c;
		zcosm =   distdimensionless .*(1+ ( q + 1) *distdimensionless / 2 + ( j + 2*q + 1) *square(distdimensionless )/6   ) ;
		
		mu = 5* log10 ( c*zcosm / 100 .* (1+ (1-q) * zcosm/ 2 - (1-q-3*square(q) +j)*square(zcosm)/6) )+25;
		for (i in 1:numsn){
			zpecvel[i]        =(k_x1_x2[i] * zpecdesignmatrix[i]);
			densitycontrast[i]=(k_x1_x2[i] * densitydesignmatrix[i]);
		}
	}
	

}
model{
	//priors
	OmegaM ~ normal(.3166,.0084);
	offset ~ normal(0,1);
	sigint ~ lognormal(log(.1),.5);
	vpecnonlindisp ~ lognormal(log(150),1);

	matrix[numsn,numsn] SIGMAmu= square(sigint)*dupsmatrixscaled+muvarscaled;
	muobs ~ multi_normal(mu +offset, SIGMAmu);
	zpecvel ~ normal( (1+zobs)./(1+zcosm) -1 , sqrt( square(vpecnonlindisp/c)+square(zerr ./(1+zcosm) ) ));
    target+=log(1+densitycontrast);
}
generated quantities{
	vector[numsn] muobs_hat;
	{
		matrix[numsn,numsn] SIGMAmu= square(sigint)*dupsmatrixscaled+muvarscaled;
		muobs_hat= multi_normal_rng(mu+offset,SIGMAmu);
	}
}
"""

c=constants.c.to(u.km/u.s).value 
locs,samples,scales,warnings=[],[],[],[]

distinterpdelt=4
npointsinterp=20
losinterppoints=np.empty((fr.size, npointsinterp))
densityinterppoints=np.empty((fr.size,npointsinterp))
distzero= np.empty(fr.size)
for i,sn in enumerate(fr):
	unitvector= SkyCoord(ra= sn['RA']*u.deg, dec=sn['DEC']*u.deg).galactic
	unitvector= np.array([np.cos(unitvector.b)*np.cos(unitvector.l),np.cos(unitvector.b)*np.sin(unitvector.l),np.sin(unitvector.b)])
	distfid=sn['zCMB']*c/100
	distzero[ i]=max(0,distfid-distinterpdelt*npointsinterp//2)
	losinterppoints[i]=np.array([losinterp((distzero[i]+distinterpdelt*j )*unitvector) for j in range(npointsinterp)])
	densityinterppoints[i]=np.array([densityinterp((distzero[i]+distinterpdelt*j )*unitvector) for j in range(npointsinterp)])

cut= ~np.isnan(losinterppoints).any(axis=1)
print(cut.sum())
datadictionary={
	'numsn': int(cut.sum()),
	 'nintdist': (npointsinterp),
	 'zobs':  fr['zCMB'][cut],
	 'zerr': fr['zCMBERR'][cut],
	 'muobs': fr['MU'][cut],
	 'muerr': fr['MUERR_RAW'][cut],
	 'biasscale':fr['biasScale_muCOV'][cut],
	 'dupsmatrix': dupsmatrix[cut,:][:,cut],
	 
	 'rho': 1,
	 'losdistancezero': distzero[cut],
	 'losdistancedelta':distinterpdelt ,
	 'losvelocities': losinterppoints[cut],
	 'losdensities':densityinterppoints[cut],
	 
	 'c':c ,
	 'q': -0.55,
	 'j':1
}
print('Compiling Stan model')

velonlymodel=stan.build(velonlycode,datadictionary)

num_chains=4
print('Sampling from Stan model')
velsample=velonlymodel.sample(num_chains=num_chains,init= [{
		'realinterpindex':np.zeros(cut.sum())+npointsinterp//2,
		'offset':0,
		'sigint':.1,
		'vpecnonlindisp':250,
		'OmegaM':.3
		
		}]*num_chains )#

with open(args.outputfile.replace('.csv','.pickle'),'wb') as file:
	pickle.dump([datadictionary,velsample],file)
	
locs,scales= np.mean( velsample['zcosm'],axis=1), np.std(velsample['zcosm'], axis=1)
fr=fr[cut]
cid,idsurvey=fr['CID'],fr['IDSURVEY']
with open(args.outputfile,'w') as file:
	writer=csv.writer(file,delimiter=',')
	writer.writerow(['CID','IDSURVEY','zCMB','zHD', 'zHDERR'])
	for i in tqdm(range(fr.size),total=fr.size):
			if np.isnan(locs[i]): continue
			writer.writerow([cid[i],idsurvey[i],fr['zCMB'][i], locs[i],scales[i]])
