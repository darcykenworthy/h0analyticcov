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
sncoords,snpos,separation,angsep= getpositions(fr,hostlocs=False)
dotproddipole= np.cos(sncoords.separation(dipolecoord)).value

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
			zpecvel[i]  = (vext-159)/c*dotproddipole[i] + betarescale*(k_x1_x2[i] * zpecdesignmatrix[i]);
			densitycontrast[i]=(k_x1_x2[i] * densitydesignmatrix[i]);
		}
	}	

}
model{
	betarescale ~ normal(1,.021/.431);
	vext ~ normal(159,23);
	zpecvel ~ normal( (1+zobs)./(1+zcosm) -1 , sqrt( square(vpecnonlindisp/c)+square(zerr ./(1+zcosm) ) ));
    target+=log(1+densitycontrast);
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
	unitvector= sncoords[i].galactic
	unitvector= np.array([np.cos(unitvector.b)*np.cos(unitvector.l),np.cos(unitvector.b)*np.sin(unitvector.l),np.sin(unitvector.b)])
	distfid=sn['zCMB']*c/100
	distzero[ i]=max(0,distfid-distinterpdelt*npointsinterp//2)
	losinterppoints[i]=np.array([losinterp((distzero[i]+distinterpdelt*j )*unitvector) for j in range(npointsinterp)])
	densityinterppoints[i]=np.array([densityinterp((distzero[i]+distinterpdelt*j )*unitvector) for j in range(npointsinterp)])

intmmvolumecut= ~np.isnan(losinterppoints).any(axis=1)
print(intmmvolumecut.sum())
datadictionary={
	'rho': 1,
	'numsn': int(intmmvolumecut.sum()),
	 'nintdist': (npointsinterp),
	 'zobs':  fr['zCMB'][intmmvolumecut],
	 'zerr': fr['zCMBERR'][intmmvolumecut],
	 'dotproddipole': dotproddipole[intmmvolumecut],
	 
	 'vpecnonlindisp': 150,
	 'losdistancezero': distzero[intmmvolumecut],
	 'losdistancedelta':distinterpdelt ,
	 'losvelocities': losinterppoints[intmmvolumecut],
	 'losdensities':densityinterppoints[intmmvolumecut],
	 'c':c ,
	 'q': -0.55,
	 'j':1
}
print('Compiling Stan model')

velonlymodel=stan.build(velonlycode,datadictionary)

print('Sampling from Stan model')
velsample=velonlymodel.sample(init= [{'realinterpindex':np.zeros(intmmvolumecut.sum())+npointsinterp//2}]*4)
datadictionary['fitres']=fr
datadictionary['volumecut']=intmmvolumecut
with open(args.outputfile.replace('.csv','.pickle'),'wb') as file:
	pickle.dump([datadictionary,velsample],file)
	
locs,scales= np.mean( velsample['zcosm'],axis=1), np.std(velsample['zcosm'], axis=1)
cid,idsurvey=fr[intmmvolumecut]['CID'],fr[intmmvolumecut]['IDSURVEY']
with open(args.outputfile,'w') as file:
	writer=csv.writer(file,delimiter=',')
	writer.writerow(['CID','IDSURVEY','zCMB','zHD', 'zHDERR'])
	for i in tqdm(range(fr[intmmvolumecut].size),total=fr[intmmvolumecut].size):
			if np.isnan(locs[i]): continue
			writer.writerow([cid[i],idsurvey[i],fr[intmmvolumecut]['zCMB'][i], locs[i],scales[i]])

pecvelcov= np.load(args.pecvelcov)+np.diag(datadictionary['vpecnonlindisp']**2)
zpvtotal = (1+fr[intmmvolumecut]['zCMB'][:,np.newaxis] )/(velsample['zcosm']+1)-1
zpvinfcov= np.cov(zpvtotal)
regressioncoeffs=pecvelcov[intmmvolumecut,~intmmvolumecut]


