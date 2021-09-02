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
import pystan
import argparse
import csv
from os import path
import logging
import io
import pickle
# silence logger, there are better ways to do this
# see PyStan docs
logging.getLogger("pystan").propagate=False
stream=io.StringIO('')
handler=logging.StreamHandler(stream)
handler.setLevel('WARNING')
logging.getLogger("pystan").handlers=[]
logging.getLogger("pystan").addHandler(handler )

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

vel=np.load('/Users/darcykenworthy/Downloads/twompp_velocity.npy')
density=np.load('/Users/darcykenworthy/Downloads/twompp_density.npy')

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

print('Compiling Stan model')

velonlycode="""
functions {
    int intFloor(int leftStart, int rightStart, real iReal)
    {
      // This is absurd. Use bisection algorithm to find int floor.
      int left;
      int right;

      left = leftStart;
      right = rightStart;

      while((left + 1) < right) {
        int mid;
        // print("left, right, mid, i, ", left, ", ", right, ", ", mid, ", ", iReal);
        mid = left + (right - left) / 2;
        if(iReal < mid) {
          right = mid;
        }
        else {
          left = mid;
        }
      }
      return left;
    }
    // Interpolate arr using a non-integral index i
    // Note: 1 <= i <= length(arr)
    real interpolateLinear(real[] arr, real i)
    {
      int iLeft;
      real valLeft;
      int iRight;
      real valRight;

      // print("interpolating ", i);

      // Get i, value at left. If exact time match, then return value.
      iLeft = intFloor(1, size(arr), i);
      valLeft = arr[iLeft];
      if(iLeft == i) {
        return valLeft;
      }
    
      // Get i, value at right.
      iRight = iLeft + 1;
      valRight = arr[iRight];

      // Linearly interpolate between values at left and right.
      return valLeft + (valRight - valLeft) * (i - iLeft);
    }
}

data {
    int nintdist;
    real zobs;
    real zerr;
    real vpecnonlindisp;
    real losdistancezero;
    real losdistancedelta;
    real losvelocities[nintdist];
    real losdensities[nintdist];
    real c;
    real q;
    real j;
}

parameters{
    real<lower=losdistancezero,upper=losdistancezero+losdistancedelta*nintdist> latdistance;
}
transformed parameters{
    real distdimensionless= latdistance*100/c;
    real zpecvel = interpolateLinear(losvelocities,1+ (latdistance-losdistancezero)/losdistancedelta)/c;
	real zcosm =   distdimensionless*(1+ ( q + 1) *distdimensionless / 2 + ( j + 2*q + 1) *square(distdimensionless )/6   ) ;
    real densitycontrast = interpolateLinear(losdensities,1+ (latdistance-losdistancezero)/losdistancedelta);
}
model{
    zpecvel ~ normal( (1+z)/(1+zcosm) -1 , sqrt( square(vpecnonlindisp/c)+square(zerr/(1+zcosm)));
    target+=log(1+densitycontrast);
}
"""
velonlymodel=pystan.StanModel(model_code=velonlycode)

print('Running model on each supernova')
c=constants.c.to(u.km/u.s).value 
locs,samples,scales,warnings=[],[],[],[]

def samplezcosmforsn(sn):
	unitvector= SkyCoord(ra= sn['RA']*u.deg, dec=sn['DEC']*u.deg).galactic
	unitvector= np.array([np.cos(unitvector.b)*np.cos(unitvector.l),np.cos(unitvector.b)*np.sin(unitvector.l),np.sin(unitvector.b)])
	distfid=sn['zCMB']*c/100
	distinterpdelt=2
	npointsinterp=40
	distzero=max(0,distfid-distinterpdelt*npointsinterp//2)
	losinterppoints=np.array([losinterp((distzero+distinterpdelt*i )*unitvector) for i in range(npointsinterp)])
	densityinterppoints=np.array([densityinterp((distzero+distinterpdelt*i )*unitvector) for i in range(npointsinterp)])
	if np.isnan(losinterppoints).any():
		return [],np.nan,np.nan,{},''
	else:
		velsample=velonlymodel.sampling({
		 'nintdist': npointsinterp,
		 'zobs':  sn['zCMB'],
		 'zerr': sn['zCMBERR'],
		 'vpecnonlindisp': 250,
		 'losdistancezero': distzero,
		 'losdistancedelta':distinterpdelt ,
		 'losvelocities': losinterppoints,
		 'losdensities':densityinterppoints,
		 'c':c ,
		 'q': -0.55,
		 'j':1
	},control={'max_treedepth':20},init=lambda :{
		'z': np.random.normal(sn['zCMB'],1e-4),
		'latdistance': np.random.normal(distfid,10)
	})
		loc,scale=stats.norm.fit(velsample['zcosm'])
		warnings=stream.getvalue()
		stream.truncate(0)
		return velsample['zcosm'],loc,scale,{key:velsample[key] for key in ['densitycontrast','zpecvel','latdistance']},warnings
results= [samplezcosmforsn(sn) for sn in fr]
with open(args.outputfile.replace('.csv','.pickle'),'wb') as file:
	pickle.dump(results,file)
	
zcosm,locs,scales,zpecvel,warnings=list(zip(*results))
cid,idsurvey=fr['CID'],fr['IDSURVEY']
with open(args.outputfile,'w') as file:
	writer=csv.writer(file,delimiter=',')
	writer.writerow(['CID','IDSURVEY','zCMB','zHD', 'zHDERR'])
	for i in tqdm(range(fr.size),total=fr.size):
			if np.isnan(locs[i]): continue
			writer.writerow([cid[i],idsurvey[i],fr['zCMB'][i], locs[i],scales[i]])
stream.close()
