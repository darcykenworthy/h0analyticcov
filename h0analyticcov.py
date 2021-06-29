import classylss.binding as CLASS
import classylss

from scipy.special import spherical_jn,legendre
from scipy import constants,stats
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
from astropy import units as units

from tqdm import tqdm,trange
from os import path
from math import ceil
import re,argparse,os
 

from utilfunctions import *

def calc_pk_nl_dlog10k(kmin,kspacing,kmax,sig_rsd,background=0):
	"""Calculate the nonlinear power spectrum in log k with CLASS/halofit for a given minimum k, # of k points, and maximum k, along with a rsd term"""
	k = np.logspace(np.log10(kmin),np.log10(kmax),kspacing)
	dlog10k=(np.log10(kmax)-np.log10(kmin))/(kspacing-1)
	pk_nl = sp.get_pk(k=k, z=0)
	if sig_rsd==0:
		redshiftdistortion=1
	else:
		redshiftdistortion=np.sinc(k*sig_rsd/np.pi)**2
		redshiftdistortion[k>np.pi/sig_rsd]=0
	pk_nl*=redshiftdistortion
	pk_nl+=background
	pk_nl_dlog10k=(pk_nl)*k*np.log(10)
	return k,dlog10k,pk_nl_dlog10k


def w(u,v,cosa):
	"""Calculates the series of sums over the derivative of bessell functions times a legendre polynomials """
	w=np.sqrt(u**2+v**2-2*u*v*cosa)
	wdwdu=u-v*cosa
	wdwdv=v-u*cosa
	sinw=np.sin(w)
	cosw=np.cos(w)
	wdwdutimeswdwdv=wdwdu*wdwdv
	wcoswlesssinw=w*cosw-sinw
	wsq=w**2
	return (-wdwdutimeswdwdv * (wsq*sinw+3*wcoswlesssinw) - wcoswlesssinw*(wsq*cosa ))/w**5



def calculateintegralterm(k,dlog10k,pk_nl_dlog10k,i,j,subtractmono=False):
	""" Calculate the integral of the bessell summation times the power spectrum"""
	sum=0
	if i==j: 
		sum=1/3
		if subtractmono: sum-=spherical_jn(0,k*chi[i],derivative=True)**2 
	else: 
		pointing=np.cos(sncoords[i].separation(sncoords[j])).value
		phases=k*chi[i],k*chi[j]
		sum=bessellsummation(*phases,pointing)
		if subtractmono: sum-= spherical_jn(0,phases[0],derivative=True)*spherical_jn(0,phases[1],derivative=True)
	return (pk_nl_dlog10k*sum).sum()*dlog10k/(2*np.pi**2)
    #    return np.trapz(sum*pk_nl,k)/(2*np.pi**2)



##########################################################################################    
import argparse

parser = argparse.ArgumentParser(description='Calculate velocity covariance between supernovae from fitres file')
parser.add_argument('fitresfile',type=str, 
                    help='File with supernova redshifts')
parser.add_argument('classfile',type=str, 
                    help='CLASS ini file')
parser.add_argument('--zkey',default='zCMB',
                    help='Name of the redshift column in fitres')
parser.add_argument('--outputstem',default=False,
                    help='Add this to the output file name')
parser.add_argument('--sigrsd',type=float,default=14,
                    help='Redshift space distortion scale in Mpc/h(default is 14, 0 deactivates redshift space distortions)')
parser.add_argument('--kprecision',type=float,nargs=3,default=[2e-4,3000,.2],
                    help='Minimum value of k for integration, number of points, and maximum value of k')
parser.add_argument('--nonlinear_scale',type=float,default=-1,
                    help='Splits the integral over k in two at this value in h/Mpc and saves them to output separately. Set to -1 to cancel')
parser.add_argument('--chunksize',type=int,default=10000,
                    help='Number of covariance matrix elements to calculate in each vectorized operation. Setting this larger may improve performance, but increases memory footprint')
parser.add_argument('--redshiftrange',type=float,default=[0,20],nargs=2,
                    help='Minimum and maximum redshift')
parser.add_argument('--cutdups',action='store_const',
                    const=True, default=False,
                    help='If true, cut any duplicates from the covariance matrix')


args = parser.parse_args()
separatenonlinear=args.nonlinear_scale!=-1
fitresfile=args.fitresfile
inifile=args.classfile
redshiftkey=args.zkey
#Smallest k, # of log-spaced points, max k
sig_rsd=args.sigrsd
kprecision=args.kprecision[0],int(args.kprecision[1]),args.kprecision[2]
# sig_rsd=0
# kprecision=2e-4,1000,30
minred,maxred=args.redshiftrange

dummyval=np.nan
chunksize=args.chunksize
filenameonly=lambda x: path.splitext(path.split(x)[-1])[0]
outputstem='-'.join([filenameonly(fitresfile),filenameonly(inifile)])+('-separate' if separatenonlinear else '')
if args.outputstem: outputstem+='-'+args.outputstem
##########################################################################################    

sndata=readFitres(fitresfile)#sndata=readFitres('../../voidtest/SALT2mu_wcsphst2.fitres')
sndata=renameDups(sndata)
if args.cutdups:
	sndata=cutdups(sndata)

# r16names=np.genfromtxt('../voidtest/R16_hubflo.txt',skip_header=3,usecols=0,dtype='U20')
# r16err=np.genfromtxt('../voidtest/R16_hubflo.txt',skip_header=3,usecols=2)
#cut=np.array([x in r16names for x in sndata['CID']])
cut=( sndata[redshiftkey]>minred) & ( sndata[redshiftkey]<maxred) 
#cut= sndata['zCMB']<.5
sndata=sndata[cut].copy()
print(f'Calculating covariance matrix for {sndata.size} SNe')
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

#Initialize CLASS
classparams=classylss.load_ini(inifile)
classparams['output']='mPk,mTk,vTk'
cosmo = CLASS.ClassEngine(classparams)
sp = CLASS.Spectra(cosmo)
bg = CLASS.Background(cosmo)
print('Evaluating background quantities')
z=sndata[redshiftkey]
#Calculate background cosmology quantites
D= bg.scale_independent_growth_factor(z)
f=bg.scale_independent_growth_rate(z)
#The only dimensionfull quantity that isn't in Mpc/h units already. Seems weird but ...
H=bg.hubble_function(z)/bg.h

#These are in Mpc/h *!#**!(#@*&$^(!&@$))
D_L=bg.luminosity_distance(z)
tau=bg.tau(z)
chi=bg.comoving_distance(z)
#plt.plot(z,D , label=r"$f(z)$")

# f(z)= d ln D / d ln a
# d D / d tau= f * D * H / (1+z)
dDdtau=H*D*f/(1+z)

#Define SN coordinates on the sky
sncoords=SkyCoord(ra=sndata['RA'],dec=sndata['DEC'],unit=units.deg)
snra=sndata['RA']*units.degree
sndec=sndata['DEC']*units.degree

#Calculate comoving positions
snpos=np.zeros((z.size,3))
snpos[:,0]=np.cos(sndec)*np.sin(snra)
snpos[:,1]=np.cos(sndec)*np.cos(snra)
snpos[:,2]=np.sin(sndec)
snpos*=chi[:,np.newaxis]

print('Calculating physical and angular separation between supernovae')

#Separation in comoving space (Mpc/h)
separation=np.sqrt(((snpos[:,np.newaxis,:]-snpos[np.newaxis,:,:])**2).sum(axis=2))
#Angular separations (degrees)
angsep=np.empty((z.size,z.size))
for i in range(z.size):
    angsep[i,:]=sncoords.separation(sncoords[i])

print('Evaluating power spectrum')
kmin,kspacing,kmax=kprecision
k,dlog10k,pk_nl_dlog10k=calc_pk_nl_dlog10k(*kprecision,sig_rsd,0)

#Calculate only the elements that aren't nonlinear, and only calculate the upper triangular part, then copy to lower triangular
calculatelements=np.ones(separation.shape,dtype=bool)
calculatelements[np.tril_indices(z.size,-1)]=False
calculatelements[separation==0]=False
print('Calculating velocity covariance matrix')
if separatenonlinear:
	integralterm=np.ones((z.size,z.size,2))*dummyval
else:
	integralterm=np.ones((z.size,z.size))*dummyval
calculateindices=np.where(calculatelements)
numchunks=ceil(calculateindices[0].size/chunksize)
print('{} elements to calculate in {} chunks'.format(calculateindices[0].size,numchunks))
#Calculating this in chunks so that we don't blow past memory limits calculating billions of elements, while also being computationally fast
for i in trange(numchunks):
	chunk=slice(i*chunksize,(i+1)*chunksize)
	chunkindices=tuple([x[chunk] for x in calculateindices])

	#Calculate the quantities that the Bessell series is sensitive to, k*chi for each index, k*separation, and cos(angsep)
	#First axis of these is k, second is chunk index
	u=k[:,np.newaxis]*chi[chunkindices[0]][np.newaxis,:]
	v=k[:,np.newaxis]*chi[chunkindices[1]][np.newaxis,:]
	w=k[:,np.newaxis]*separation[chunkindices][np.newaxis,:]
	#Convert to rad from deg
	cosa=np.cos(angsep[chunkindices]*np.pi/180)[np.newaxis,:]*np.ones((k.size,1))
	#Exclude all the ones that are the same position as those reduce to 1/3
# 	nosep=w==0
# 	u,v,w,cosa=u[~nosep],v[~nosep],w[~nosep],cosa[~nosep]
	#Make sure I'm not wasting function evaluations
	wdwdu=u-v*cosa
	wdwdv=v-u*cosa
	sinw=np.sin(w)
	cosw=np.cos(w)
	wdwdutimeswdwdv=wdwdu*wdwdv
	wcoswlesssinw=w*cosw-sinw
	wsq=w**2
	#Calculate the series over Bessell functions (this relation corresponds to d^2 sinc(w) / du dv) (as the separation, w^2=u^2+v^2-2 uv cos(alpha))
	series= (-wdwdutimeswdwdv * (wsq*sinw+3*wcoswlesssinw) - wcoswlesssinw*(wsq*cosa ))/w**5
	#integrate over the power spectrum dlogk
	if separatenonlinear:
		integrand=series*pk_nl_dlog10k[:,np.newaxis]
		elements=([(integrand[k<args.nonlinear_scale,:] ).sum(axis=0)*dlog10k/(2*np.pi**2),(integrand[k>=args.nonlinear_scale,:] ).sum(axis=0)*dlog10k/(2*np.pi**2)])
		integralterm[chunkindices[0],chunkindices[1],np.tile(0,chunkindices[0].size)]=elements[0]
		integralterm[chunkindices[1],chunkindices[0],np.tile(0,chunkindices[0].size)]=elements[0]
		integralterm[chunkindices[0],chunkindices[1],np.tile(1,chunkindices[0].size)]=elements[1]
		integralterm[chunkindices[1],chunkindices[0],np.tile(1,chunkindices[0].size)]=elements[1]
	else:
		elements=(series*pk_nl_dlog10k[:,np.newaxis]).sum(axis=0)*dlog10k/(2*np.pi**2)
		integralterm[chunkindices]=elements
		integralterm.T[chunkindices]=elements
integralterm[separation==0]=1./3*(pk_nl_dlog10k).sum(axis=0)*dlog10k/(2*np.pi**2)	

velcovoutput='velocitycovariance-{}.npy'.format(outputstem)
namelistoutput='snnames-{}.npy'.format(outputstem)
distanceprefactor=D*f * (1+z)/ D_L * 5 /np.log(10)
#Convert dDdtau from dimensionless to km/s
velocityprefactor=dDdtau*constants.c.to(units.km/units.s).value
if separatenonlinear:
	velocitycovariance=np.outer(velocityprefactor,velocityprefactor)[:,:,np.newaxis]*integralterm
else:
	velocitycovariance=np.outer(velocityprefactor,velocityprefactor)*integralterm
print(f'Saving velocity covariance to {velcovoutput} and names of SNe in same order to {namelistoutput}')
np.save(velcovoutput,velocitycovariance)
np.save(namelistoutput,sndata['CID'])
