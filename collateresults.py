import numpy as np
import matplotlib.pyplot as plt
import pickle
import os,sys
from os import path

def constructtable(redshiftsfile,directory='.'):
	results={}
	names=np.genfromtxt(redshiftsfile,names=True).dtype.names
	rows=[]
	for filename in os.listdir(directory):
		if filename.endswith('pickle') and filename.startswith('marginal') and path.splitext(redshiftsfile)[0] in filename:
			with open(directory+'/'+filename,'rb') as file: result=pickle.load(file)
			index=int((path.splitext(filename)[0]).split('_')[-1])
			logprob=result['pars']['lp__']
			argmax=logprob.argmax()
			rows+=[(names[index], result['logml'], *[np.median(result['pars'][x]) for x in result['pars'] if x!= 'lp__'],logprob[argmax],*[(result['pars'][x][argmax]) for x in result['pars'] if x!= 'lp__'])]

	newnames=['Name','Bayes Factor'] +['Median '+x for x in result['pars'] if x!= 'lp__']+['Mode of posterior']+['Maximum a posteriori '+x for x in result['pars'] if x!= 'lp__']
	rows=sorted(rows,key=lambda x: names.index(x[0]))
	table=np.array(rows, dtype=list(zip(newnames,['U20' if i==0 else float for i in range(len(newnames))])))
	return table
if __name__ =='__main__':
	print('Collating results from all pickles into {}'.format('fitresults_'+sys.argv[1]))
	table=constructtable(*sys.argv[1:])
	np.savetxt('fitresults_'+sys.argv[1],table,fmt=['%'+ ('21s' if i==0 else str(max(10,len(table.dtype.names[i])))+'.5f') for i in range(len(table.dtype))],delimiter='\t', header='\t'.join([('%21s' if i==0 else '%10s')%(x) for i,x in enumerate(table.dtype.names)]), comments='')
