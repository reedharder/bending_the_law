import numpy as np
import pandas as pd
import random as rd
import scipy as sp
from matplotlib.pyplot import *
import time
import itertools
import os
import pickle
self={}






def load_data(data_dir):
	judicial = pd.read_csv(data_dir + 'judicial.csv')
	allcites = pd.read_csv(data_dir + 'allcites.txt',sep = ' ',names=np.array(['citation','contains']))
	judicial['usid_nice'] = judicial.usid.str.replace('US','_US_') + '.txt'
	Docs_We_Have = os.listdir("/Users/gregoryleibon/Citation_Network/Data/original_citations_from_folder1t8")
	Docs_We_Have2 = judicial.usid_nice[judicial.usid_nice.isin(Docs_We_Have)]
	phi = pd.read_csv(data_dir + 'fit-US-flattened-yearprefix-split-munich-phi.csv',index_col = 0)
	theta = pd.read_csv(data_dir + 'fit-US-flattened-yearprefix-split-munich-theta-merged.csv',index_col = 0)
      
	Docs_In_Comp_Data = np.array(theta.index)
	Dict0 = {}
	Dict0['usid'] = Docs_In_Comp_Data
	Dict0['usid_nice'] = Docs_In_Comp_Data+'.txt'
	Docs_In_Comp_Data = pd.DataFrame(Dict0)
	Docs_We_Have3 = Docs_In_Comp_Data.usid_nice[Docs_In_Comp_Data.usid_nice.isin(Docs_We_Have2)]
	Our_Docs = judicial[judicial.usid_nice.isin(Docs_We_Have3)]
	Our_Docs = Our_Docs.sort_index(by='year',ascending=False)
	Our_Docs['usid_index'] = Our_Docs.usid_nice.str.replace('.txt','')
	theta = theta.join(Our_Docs[['usid_index','usid']].set_index('usid_index'))
	Our_theta = theta[theta.usid.isnull()==False]
	del Our_theta['usid']
	# Get it down to good topics 
	Our_Docs=Our_Docs.set_index('usid')
	Our_Cites=allcites[(allcites.citation.isin(Our_Docs.caseid)) & (allcites.contains.isin(Our_Docs.caseid))]	
	Our_Cites['temp']=1
	G=Our_Cites.groupby(['citation'])['temp'].sum() 
	G.name='count_contains'
	G=pd.DataFrame(G)
	Our_Docs=Our_Docs.join(G,on='caseid')
	Our_Cites['temp']=1
	G=Our_Cites.groupby(['contains'])['temp'].sum() 
	G.name='count_citations'
	G=pd.DataFrame(G)
	Our_Docs=Our_Docs.join(G,on='caseid')
	del Our_Cites['temp']
	Our_Cites_Sym= pd.concat([Our_Cites.rename(columns={'citation':'case1','contains':'case2'}),Our_Cites.rename(columns={'citation':'case2','contains':'case1'})])
	Our_Cites_Sym=Our_Cites_Sym.drop_duplicates(['case1','case2'])
	Our_Cites_Sym['temp']=1
	G=Our_Cites_Sym.groupby(['case1'])['temp'].sum() 
	G.name='count_connections'
	G=pd.DataFrame(G)
	Our_Docs=Our_Docs.join(G,on='caseid')
	return [Our_Docs, Our_theta, phi, Our_Cites,  Our_Cites_Sym]

if __name__ =='__main__':
	'''	
source ~/.bashrc 
workon InitialPyEnv
ipython --pylab
	'''
	data_dir='/Users/gregoryleibon/Citation_Network/Data/'

	# run Bending_The_Law_Data.py
	t0=time.time() 
	[Our_Docs, Our_theta, phi, Our_Cites,  Our_Cites_Sym] = load_data(data_dir)
	print (time.time()-t0)


