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
import csv
cosmo=Planck15


sdss=['16314','16392','16333','14318','17186','17784','7876']
trunames=['2006oa','2006ob','2006on','2006py','2007hx','2007jg','2005ir']
def readFitres(fileName):			
    with open(fileName,'r') as file : fileText=file.read()
    result=re.compile('VARNAMES:([\w\s]+)\n').search(fileText)
    names= ['VARNAMES:']+[x for x in result.groups()[0].split() if not x=='']
    namesToTypes={'VARNAMES':'U3','CID':'U20','FIELD':'U4','IDSURVEY':int}
    types=[namesToTypes[x] if x in namesToTypes else float for x in names]
    data=np.genfromtxt(fileName,skip_header=fileText[:result.start()].count('\n')+1,dtype=list(zip(names,types)))
    for sdssName,truName in zip(sdss,trunames):
            data['CID'][data['CID']==sdssName]=truName
    return data

def weightedMean(vals,covs,cut=None):
    if cut is None:cut=np.ones(vals.size,dtype=bool)
    if covs.ndim==1:
        vars=covs
        mean=((vals)/vars)[cut].sum()/(1/vars)[cut].sum()
        chiSquared=((vals-mean)**2/vars)[cut].sum()
        var=1/((1/vars)[cut].sum())
    else:
        vals=vals[cut]
        covs=covs[cut[:,np.newaxis]&cut[np.newaxis,:]].reshape((cut.sum(),cut.sum()))
        #Use cholesky transform instead of numerical inversion of symmetric matrix
        transform=np.linalg.cholesky(covs)
        design=linalg.solve_triangular(transform,np.ones(vals.size),lower=True)
        var=1/np.dot(design,design)
        mean=var*np.dot(design,linalg.solve_triangular(transform,vals,lower=True))
        pulls=linalg.solve_triangular(transform,vals-mean,lower=True)
        chiSquared=np.dot(pulls,pulls)
    return mean,np.sqrt(var),chiSquared

def addredshiftcolumnfromfile(redshiftfile,sndataFull,redshiftcolumn):
	if redshiftfile.lower().endswith('.fitres'):
		zvector,evalcolumn=readFitres(redshiftfile)[redshiftcolumn],redshiftcolumn
		sndataFull=np.array(recfunctions.append_fields(sndataFull,'zeval',zvector))
		sndataFull=np.array(recfunctions.append_fields(sndataFull,'isgroup',np.tile(False,zvector.size)))
	else:
		redshifttable=np.genfromtxt(redshiftfile,names=True)
		def loadzvec(index):
			try: 
				retvals=redshifttable[index],index
			except:
				name=redshifttable.dtype.names[int(index)]
				retvals=redshifttable[name],name
			return retvals
		if args.posteriorpredictive:
			zvector,evalcolumn=np.zeros(redshifttable.size),'NULL'
		zvector,evalcolumn=loadzvec(redshiftcolumn)

		zvectornames=readFitres('lowz_comb_csp_hst.fitres')['CID']
		zvectorindices=[]
		i=0
		j=0
		while i<sndataFull.size and j<zvectornames.size:
			if sndataFull['CID'][i]==zvectornames[j]:
				zvectorindices+=[j]
				i+=1
				j+=1
			else:
				j+=1
		print(f'Loaded redshift vector \"{evalcolumn}\"')
		zvectorindices=np.array(zvectorindices)
		if evalcolumn=='lowz_comb_csp_hst':
			isgroup=~(zvector==redshifttable['tully_avg'])
		else:
			isgroup=~(zvector == redshifttable['lowz_comb_csp_hst'])
		sndataFull=np.array(recfunctions.append_fields(sndataFull,'zeval',zvector[zvectorindices]))
		sndataFull=np.array(recfunctions.append_fields(sndataFull,'isgroup',isgroup[zvectorindices]))
	return sndataFull
	
def catcolumnfromotherfitres(initial,additional,column):
	crossmatchedindices=[]
	failedcids=[]
	for cid in initial['CID']:
		try:
			crossmatchedindices+= [np.where(additional['CID']==cid)[0][0] ]
		except:
			failedcids+=[cid]
	if len(failedcids)>0:
		raise ValueError('No matching values for cids ',failedcids)
	return np.array(recfunctions.append_fields(initial,column,additional[column][crossmatchedindices]))
	
def writefitres(fitres,data):
	with open(fitres,'w') as file:
		writer=csv.writer(file,delimiter=' ')
		writer.writerow(list(data.dtype.names))
		for row in data:
			writer.writerow(['SN:']+list(row)[1:])

	
def cutdups(sndataFull,reweight=True):
	accum=[]
	result=[]
	finalInds=[]
	sndataFull=sndataFull.copy()
	for name in np.unique(sndataFull['CID']):
		dups=sndataFull[sndataFull['CID']==name]
		inds=np.where(sndataFull['CID']==name)[0]
		def sortingkey(ind):
			return {4:0, 
			1:1,
			15:2}.get(sndataFull[ind]['IDSURVEY'],3)
		inds=sorted(inds,key=sortingkey)
		finalInds+=[inds[0]]
		if len(inds)>1 and reweight:
			for x in ['MUERR_RAW','cERR','x1ERR']:
				sndataFull[x][inds[0]],sndataFull[x][inds[0]],_=weightedMean(sndataFull[x],sndataFull[x]**2,sndataFull['CID']==name)

	return sndataFull[finalInds].copy()
def checkposdef(matrix,condition=1e-10):
	vals,vecs=linalg.eigh(matrix)
	if (vals>=vals.max()*condition).all(): return matrix
	covclipped=np.dot(vecs,np.dot(np.diag(np.clip(vals,vals.max()*condition,None)),vecs.T))
	return covclipped 
def separatevpeccontributions(sndata,sncoords):

	vext,vexterr=159,23 
	l,lerr = 304 , 11
	b,berr = 6,13
	dipolecoord=SkyCoord(l=l,b=b,frame='galactic',unit=u.deg)
	vpecbulk=vext*np.cos(sncoords.separation(dipolecoord))
	z=sndata['zCMB']
	hascorrection=z<.06
	vpeclocal=np.zeros(sndata.size)
	vpeclocal[hascorrection]=(sndata['VPEC']-vpecbulk)[hascorrection]
	vpecbulk=vext*np.cos(sncoords.separation(dipolecoord))
	vpecbulk[~hascorrection]=sndata['VPEC'][~hascorrection]


	sndata=np.array(recfunctions.append_fields(sndata,'VPEC_LOCAL',vpeclocal))
	sndata=np.array(recfunctions.append_fields(sndata,'VPEC_BULK',vpecbulk))
	return sndata

def getseparation(sndata):
	zcmb=sndata['zCMB']
	snra=sndata['RA']*u.degree
	deckey='DEC' if 'DEC' in sndata.dtype.names else 'DECL'
	sndec=sndata[deckey]*u.degree
	sncoords=SkyCoord(ra=sndata['RA'],dec=sndata[deckey],unit=u.deg)

	chi=cosmo.comoving_distance(zcmb).to(u.Mpc).value
	snpos=np.zeros((sndata.size,3))
	snpos[:,0]=np.cos(sndec)*np.sin(snra)
	snpos[:,1]=np.cos(sndec)*np.cos(snra)
	snpos[:,2]=np.sin(sndec)
	snpos*=chi[:,np.newaxis]

	separation=np.sqrt(((snpos[:,np.newaxis,:]-snpos[np.newaxis,:,:])**2).sum(axis=2))
	angsep=np.empty((sndata.size,sndata.size))
	for i in range(sndata.size):
		angsep[i,:]=sncoords.separation(sncoords[i])
	return sncoords,separation,angsep