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
with open(sys.argv[1],'rb') as file: model,fit,opfit=pickle.load(file)
directory,base=path.split(sys.argv[1])
try: 
	outdirectory=sys.argv[2]
except:
	if directory=='': outdirectory='.'
	else: outdirectory=directory
os.makedirs(outdirectory,exist_ok=True)
vars=fit.constrained_param_names()
logp=lambda x: fit.log_prob(fit.unconstrain_pars({var:val for var,val in zip(vars,x)}),adjust_transform=False)

# In[4]:
#result=Marginal_llk(fit,logp,vars,bounds)
result={}
result['pars']={key:fit[key] for key in fit.model_pars}
result['lp__']=fit['lp__']
#result['pars']['lp__']=np.asarray([logp(point) for point in tqdm.tqdm(mtrace)])
result['maxposterior']=opfit
result['maxposterior']['lp__']=fit.log_prob(fit.unconstrain_pars({var:opfit[var] for var in opfit}),adjust_transform=True)
with open(path.join(outdirectory,'params_'+base),'wb') as file: pickle.dump(result,file)

# In[ ]:





