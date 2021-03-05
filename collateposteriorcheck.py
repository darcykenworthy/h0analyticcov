import numpy as np
import pickle
import matplotlib.pyplot as plt
import os
from os import path

with open('output_rescale/posteriorpredictive_rescalebeta_extra_dispersion_lowz_SALT2mu_no_vpecerr_bulk_redshifts_lowz_zero.pickle','rb') as file: simmedmu,simmedparams=pickle.load(file)
extract=lambda x: fit.extract(x)[x]
priorextract=lambda x: priorfit.extract(x)[x]
zscores=[]
dirname='posteriorcheck_rescale'
with open('output_rescale/prior_extra_dispersion_rescalebeta_lowz_SALT2mu_no_vpecerr_bulk_redshifts_lowz_zero.pickle','rb') as file: priormodel,priorfit,_=pickle.load(file)
with open('output_rescale/rescalebeta_extra_dispersion_lowz_SALT2mu_no_vpecerr_bulk_redshifts_lowz_zero.pickle','rb') as file: model,trufit=pickle.load(file)

trucontraction={param: 1-np.var(trufit.extract(param)[param])/np.var(priorextract(param))  for param in simmedparams[0]}
contraction=[]
indices=[]
for picklefile in os.listdir(dirname):
	if picklefile.startswith('extra_dispersion') and picklefile.endswith('pickle'):
		with open(path.join(dirname,picklefile),'rb') as file: model,fit,opfit=pickle.load(file)
		filenamesplit=(path.splitext(picklefile)[0]).split('_')
		index=int(filenamesplit[-1])
		indices+=[index]
		zscores+=[{param: (np.mean(extract(param)) - simmedparams[index][param] )/np.std(extract(param))  for param in simmedparams[index]}]
		contraction+=[{param: 1-np.var(extract(param))/np.var(priorextract(param))  for param in simmedparams[index]}]
print('Missing : ',[i for i in range(100) if not i in indices])
print(f'Number of posterior samples is {len(zscores)}') 
for i,param in enumerate(simmedparams[0].keys()):
    plt.subplot(231+i)
    plt.title(param)
    zscoresforparam=np.array([zscores[j][param] for j in range(len(zscores))])
    contractionforparam=np.array([contraction[j][param] for j in range(len(zscores))])
    pvalue=(contractionforparam>trucontraction[param]).sum()/(contractionforparam).size
    pvalue=min(pvalue,1-pvalue)
    plt.hist(zscoresforparam,bins=np.linspace(-3,3,12,True))
    plt.xlim(-3,3)
    print(f'{(np.abs(zscoresforparam)>3).sum()} outliers for {param}, mean is {np.mean(zscoresforparam)} std dev is {np.std(zscoresforparam)} ')
    print(f'median is {np.median(zscoresforparam)} 1.48 MAD is {1.48*np.median(np.abs(zscoresforparam-np.median(zscoresforparam)))} ')
    print(f'Median contraction {np.median(contractionforparam)} with contraction in true data {trucontraction[param]}, p-value is { pvalue}')
plt.tight_layout()
plt.savefig('zschorehists.png')
plt.clf()
ylim=-5,5
for i,param in enumerate(simmedparams[0].keys()):
    plt.subplot(231+i)
    plt.title(param)
    zscoresforparam=np.array([zscores[j][param] for j in range(len(zscores))])
    contractionforparam=np.array([contraction[j][param] for j in range(len(zscores))])
    plt.plot(contractionforparam,zscoresforparam,'k.')
    plt.plot([trucontraction[param]]*2,ylim,'b-')
    plt.ylim(*ylim)
    plt.xlim(.5,1.1)
plt.tight_layout()
plt.savefig('posteriorscatterplot.png')
plt.clf()
xlim=0.002, 0.5
xticks=[1e-2,1e-1]
minor=((np.arange(10)+1)[:,np.newaxis]*(10.**(np.arange(3) -3))[np.newaxis,:]).flatten()
minor=minor[(minor>xlim[0])&(minor<xlim[1])]
for i,param in enumerate(simmedparams[0].keys()):
    plt.subplot(231+i)
    plt.title(param)
    zscoresforparam=np.array([zscores[j][param] for j in range(len(zscores))])
    contractionforparam=np.array([contraction[j][param] for j in range(len(zscores))])
    plt.plot(1-contractionforparam,zscoresforparam,'k.')
    plt.plot([1-trucontraction[param]]*2,ylim,'b-')
    plt.ylim(*ylim)
    plt.xlim(*xlim)
    plt.ylabel('z-score')
    plt.xlabel('$\sigma^2_{Post}/\sigma^2_{Prior}$ ')
    plt.xscale('log')
    plt.gca().set_xticks(xticks,minor=False)
    plt.gca().set_xticks(minor,minor=True)

plt.tight_layout()
plt.savefig('semilogposteriorscatterplot.png')
