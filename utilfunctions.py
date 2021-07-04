import numpy as np
from numpy.lib import recfunctions

import matplotlib.pyplot as plt
from scipy import optimize as op,linalg,stats
from itertools import combinations

from astropy.coordinates import SkyCoord
from astropy import units as u, constants
from astropy.cosmology import FlatLambdaCDM,Planck15
import re,timeit,pickle,argparse,multiprocessing,os
import matplotlib.ticker as mtick
from os import path
import hashlib
import csv,json
# from v3_0_duplicate_map import dups as dillon_dups
from duplicates_v3_06_renamed import duplicate_dictionary as dillon_dups



cosmo=Planck15


def readFitres(fileName):			
    with open(fileName,'r') as file : fileText=file.read()
    result=re.compile('VARNAMES:([\w\s]+)\n').search(fileText)
    names= ['VARNAMES:']+[x for x in result.groups()[0].split() if not x=='']
    namesToTypes={'VARNAMES':'U3','CID':'U20','FIELD':'U4','IDSURVEY':int}
    types=[namesToTypes[x] if x in namesToTypes else float for x in names]
    data=np.genfromtxt(fileName,skip_header=fileText[:result.start()].count('\n')+1,dtype=list(zip(names,types)))
    return data

def weightedMean(vals,covs,cut=None,returnpulls=False):
    if cut is None:cut=np.ones(vals.size,dtype=bool)
    if covs.ndim==1:
        vars=covs
        mean=((vals)/vars)[cut].sum()/(1/vars)[cut].sum()
        chiSquared=((vals-mean)**2/vars)[cut].sum()
        pulls=((vals-mean)/np.sqrt(vars))[cut]
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
    if returnpulls:
    	return mean,np.sqrt(var),chiSquared,pulls
    else:
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

def renameDups(sndataFull):
	sndataFull=sndataFull.copy()
	uniqueids={}
	coveredcids=set()
	cids=np.unique(sndataFull['CID'])
# 	for cid in (cids):
# 		if cid in coveredcids: continue
# 		#if cid == '10805': import pdb;pdb.set_trace()
# 		coveredcids.add(cid)
# 		name,aliases=cid,{cid}
# 		while True:
# 			for edge in dillon_dups.items():
# 				if  (edge[0] in cids) and (edge[1] in cids) and (edge[0] in aliases)^(edge[1] in aliases):
# 					aliases.add(edge[0])
# 					aliases.add(edge[1])
# 					coveredcids.add(edge[0])
# 					coveredcids.add(edge[1])
# 					break
# 			else:
# 				break
# 		uniqueids[name]=aliases
# 	for unique in uniqueids:
# 		for alias in uniqueids[unique]:
# 			sndataFull['CID'][sndataFull['CID']==alias]=unique
	for alias in dillon_dups:
		sndataFull['CID'][sndataFull['CID']==alias]=dillon_dups[alias]
	return sndataFull
	
def cutdups(sndataFull,reweight=False,returninds=False):
	accum=[]
	result=[]
	finalInds=[]
	sndataFull=sndataFull.copy()
	
	results=[]
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
			sndataFull['MU'][inds[0]],sndataFull['MUERR_RAW'][inds[0]],chisq,resids=weightedMean(sndataFull['MU'],sndataFull['MUERR_RAW']**2,sndataFull['CID']==name,returnpulls=True)
			results+=[(name,chisq, (sndataFull['CID']==name).sum(),sndataFull[sndataFull['CID']==name]['IDSURVEY'],resids)]
	if returninds:
		return sndataFull[finalInds].copy(),finalInds
	else:
		return sndataFull[finalInds].copy()#,results
	
	
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

def getpositions(sndata,hostlocs=True):
	zcmb=sndata['zCMB']
	rakey='HOST_RA' if hostlocs else 'RA'
	snra=sndata['RA']*u.degree
	deckey=('HOST_' if hostlocs else '')+'DEC' if 'DEC' in sndata.dtype.names else 'DECL'
	sndec=sndata[deckey]*u.degree
	sncoords=SkyCoord(ra=snra,dec=sndec,unit=u.deg)

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
	return sncoords,snpos,separation,angsep