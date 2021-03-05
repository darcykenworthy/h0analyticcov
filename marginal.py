#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle,os
import numpy as np
import tqdm
from scipy import linalg,stats
import sys,os
from os import path
# In[2]:


# This is based on the code found at https://gist.github.com/junpenglao/4d2669d69ddfe1d788318264cdcf0583
def Marginal_llk(fit,logp,vars,bounds):
    r0, tol1, tol2 = 0.5, 1e-10, 1e-4
    mtrace= np.array([fit.extract(var)[var] for var in vars]).T
    # Split the samples into two parts  
    # Use the first 50% for fiting the proposal distribution and the second 50% 
    # in the iterative scheme.
    len_trace = len(mtrace)

    N1 = len_trace // 2
    N2 = len_trace - N1

    neff_list = dict() # effective sample size

    arraysz = len(vars)
    samples_4_fit = np.zeros((arraysz, N1))
    samples_4_iter = np.zeros((arraysz, N2))
    # matrix with already transformed samples
    for i,var in enumerate(vars):
        # for fitting the proposal
        x = mtrace[:N1,i]
        samples_4_fit[i, :] = x.T
        # for the iterative scheme
        x2 = mtrace[N1:,i]
        samples_4_iter[i, :] =x2.T

    neffdict=dict(zip(fit.summary()['summary_rownames'],fit.summary()['summary'][:,fit.summary()['summary_colnames'].index('n_eff')]))

    # median effective sample size (scalar)
    neff = np.median([neffdict[var] for var in vars])/2

    # get mean & covariance matrix and generate samples from proposal
    m = np.mean(samples_4_fit, axis=1)
    V = np.cov(samples_4_fit)
    L = linalg.cholesky(V, lower=True)

    # Draw N2 samples from the proposal distribution
    gen_samples = m[:, None] + np.dot(L, stats.norm.rvs(0, 1, 
                                         size=(samples_4_iter.shape[0],100000)))
    def checkbounds(i):
        return ((gen_samples[i]<bounds[i][0])|(gen_samples[i]>bounds[i][1]))|checkbounds(i-1) if i>=0 else False
    oob=checkbounds(len(vars)-1)
    gen_samples =  gen_samples[:,np.random.choice(np.where(~oob)[0],samples_4_iter.shape[1] ,replace=True)]
    goodfrac=1-(oob.sum())/oob.size
    # Evaluate proposal distribution for posterior & generated samples
    q12 = stats.multivariate_normal.logpdf(samples_4_iter.T, m, V)
    q22 = stats.multivariate_normal.logpdf(gen_samples.T, m, V)-np.log(goodfrac)

    # Evaluate unnormalized posterior for posterior & generated samples
    q11 = np.asarray([logp(point) for point in tqdm.tqdm(samples_4_iter.T)])
    q21 = np.asarray([logp(point) for point in tqdm.tqdm(gen_samples.T)])
    def iterative_scheme(q11, q12, q21, q22, r0, neff, tol, maxiter, criterion):
        l1 = q11 - q12
        l2 = q21 - q22
        lstar = np.median(l1) # To increase numerical stability, 
                              # subtracting the median of l1 from l1 & l2 later
        s1 = neff/(neff + N2)
        s2 = N2/(neff + N2)
        r = r0
        r_vals = [r]
        logml = np.log(r) + lstar
        criterion_val = 1 + tol

        i = 0
        while (i <= maxiter) & (criterion_val > tol):
            rold = r
            logmlold = logml
            numi = np.exp(l2 - lstar)/(s1 * np.exp(l2 - lstar) + s2 * r)
            deni = 1/(s1 * np.exp(l1 - lstar) + s2 * r)
            if np.sum(~np.isfinite(numi))+np.sum(~np.isfinite(deni)) > 0:
                warnings.warn("""Infinite value in iterative scheme, returning NaN. 
                Try rerunning with more samples.""")
            r = (N1/N2) * np.sum(numi)/np.sum(deni)
            r_vals.append(r)
            logml = np.log(r) + lstar
            i += 1
            if criterion=='r':
                criterion_val = np.abs((r - rold)/r)
            elif criterion=='logml':
                criterion_val = np.abs((logml - logmlold)/logml)

        if i >= maxiter:
            return dict(logml = np.NaN, niter = i, r_vals = np.asarray(r_vals))
        else:
            return dict(logml = logml, niter = i)
    tmp = iterative_scheme(q11, q12, q21, q22, r0, neff, tol1, 10000, 'r')
    if ~np.isfinite(tmp['logml']):
        warnings.warn("""logml could not be estimated within maxiter, rerunning with 
                      adjusted starting value. Estimate might be more variable than usual.""")
        # use geometric mean as starting value
        r0_2 = np.sqrt(tmp['r_vals'][-2]*tmp['r_vals'][-1])
        tmp = iterative_scheme(q11, q12, q21, q22, r0_2, neff, tol2, maxiter, 'logml')

    return dict(logml = tmp['logml'], niter = tmp['niter'], method = "normal", 
                q11 = q11, q12 = q12, q21 = q21, q22 = q22)

with open(sys.argv[1],'rb') as file: model,fit,opfit=pickle.load(file)
directory,base=path.split(sys.argv[1])
try: 
	outdirectory=sys.argv[2]
except:
	if directory=='': outdirectory='.'
	else: outdirectory=directory
os.makedirs(outdirectory,exist_ok=True)
bounds={ 'offset':(-np.inf,np.inf),'betarescale':(-1,2),'velscaling':(.01,10),'veldispersion_additional':(10,500),'intrins':(.01,.3)}
vars=fit.constrained_param_names()
bounds=[bounds[x] for x in vars]
logp=lambda x: fit.log_prob(fit.unconstrain_pars({var:val for var,val in zip(vars,x)}),adjust_transform=False)

# In[4]:
#result=Marginal_llk(fit,logp,vars,bounds)
result={}
result['pars']=fit.extract()
mtrace= np.array([fit.extract(var)[var] for var in vars]).T
#result['pars']['lp__']=np.asarray([logp(point) for point in tqdm.tqdm(mtrace)])
result['maxposterior']=opfit
result['maxposterior']['lp__']=fit.log_prob(fit.unconstrain_pars(opfit),adjust_transform=False)
with open(path.join(outdirectory,'marginal_'+base),'wb') as file: pickle.dump(result,file)
print(f"marginal likelihood is {result['logml']}")

# In[ ]:





