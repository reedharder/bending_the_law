# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 14:56:39 2015

@author: d29905p
"""



data_dir='C:/users/d29905p/documents/bendinglaw/'

import numpy as np
import pandas as pd
import time
import itertools
import os
import matplotlib.pyplot as plt

def normalize(A,N):
	"""
	This is a simple normalizer... that leaves zero uneffected
	"""
      # sum across all rows of matrix
	RowSum=np.dot(A,np.ones([N,1]))	
      
	V=RowSum.reshape([np.size(RowSum),])
	W=1/V
	W[V<=0]=0
	# now we dvide by sum and multiply by Psim when not zero 
	# PreTran0=dot(np.diag(W),PreTran)
	A=np.tile(W.reshape([np.size(W),1]),[1,np.size(W)])*A
	return A

def join_replace(A,B,indexA,indexB='none'):
	# joins B ontop A using these indexes, deleting columns if they exist 
	if indexB=='none':
		indexB=indexA
	for col in B.columns:
		if (np.sum(A.columns==col)>0) & (np.sum(col==indexA)==0) & (np.sum(col==indexB)==0):
			del A[col]
	A=A.join(B.set_index(indexB),on=indexA)
	return A 

def join_make_force_null(G_on,G_this,name,primary_index,null_as):
	# simple join to compression 
	G_this.name=name
	G_this=pd.DataFrame(G_this)
	G_this=G_this.reset_index()
	if np.sum(G_on.columns==name)>0:
		del G_on[name]
	G_on=G_on.join(G_this.set_index(primary_index),on=primary_index) 
	if (null_as!='NULL'):
		G_on[name][G_on[name].isnull()]=null_as
	return G_on

def find_words(phi,Topic,Top_N=5,View=False): 
	# just find the Dist tribution of a topic and np array 
	Dist=np.array(phi[phi.index==Topic])[0]
	if View==True:
		Words=np.array(phi.columns)
		Words[Dist>0.01]
		Ind=np.argsort(-Dist)
		print(Words[Ind[0:Top_N]].tolist())
		print((np.round(10000*Dist[Ind[0:Top_N]])/100).tolist())
	return Dist

def find_topics(theta,phi,Doc,Top_N=5,View=False): 
	# just find the Dist of topic
	Dist=np.array(theta[theta.index==Doc])[0]
	if View==True:
		Topics=np.array(theta.columns)
		Ind=np.argsort(-Dist)
		for k in np.arange(0,Top_N,1):
			Topic=Topics[Ind[k]]
			print( 'Topic:'+ Topic + ' (Prob:' + str((np.round(10000*Dist[Ind[k]])/100).tolist()) + ')')
			find_words(phi,Topic,5,View)
	return Dist


def kl_div(X):
	"""
	Considering the rows of X as discrete probability distributions, calculate
	the Kullback-Leibler divergence between each pair of vectors.

	The Kullback-Leibler (KL) divergence between two discrete probability
	distributions, ``p`` and ``q`` is given by

	    :math:`D_KL(p||q) = \sum_i ln(p[i]/q[i])*p[i]

	Parameters
	----------
	X : array, shape = [n, p]

	Returns
	-------
	distances : array, shape = [n, n]

	Examples
	--------
	>>> from horizont.metrics import kl_div
	>>> X = [[0.7, 0.3], [0.5, 0.5]]
	>>> # distance between rows of X
	>>> kl_div(X)
	array([[ 0.,  0.082282878505051782],
	       [ 0.087176693572388914,  0.]])
	"""

	# FIXME: look into scipy.spatial.distance.pdist
	X = np.asarray(X, dtype=float)
	n = len(X)
	distances = np.empty((n, n))
	for i, j in itertools.product(range(n), range(n)):
		distances[i, j] = 0 if i == j else np.sum(X[i]*np.log(X[i]/X[j]))
	return distances

def js_div(X):
	return 0.5*(kl_div(X)+kl_div(X).T)
 
def get_title_to_usid():
    #read in metadata for each case
    judicial = pd.read_csv(data_dir+ 'judicial.csv')
     #read in citations for cases
    allcites = pd.read_csv(data_dir+ 'allcites.txt',sep=' ',names=np.array(['citation','contains'])).dropna()
     # this is the id in the title 
    judicial['usid_nice']=judicial.usid.str.replace('US','_US_') + '.txt'
     #list of case files in our directory
    Docs_We_Have=os.listdir(data_dir + "original_citations_from_folder1t8")	
     #overlapping set of files in metadata and folder
    Docs_We_Have2=judicial.usid_nice[judicial.usid_nice.isin(Docs_We_Have)]
    titles = []
    for doc in Docs_We_Have2:
        with open(data_dir+ 'original_citations_from_folder1t8/' +doc,'r') as infile:
            for line in infile:
                if line.strip():
                    title = line
                    break
            titles.append(title.strip())
    df = pd.DataFrame(Docs_We_Have2)
    df['title']=titles
    df['usid'] = df.apply(lambda row: row['usid_nice'].split('.')[0],1)
    os.chdir(data_dir)
    df.to_csv('case_titles.csv')


'''
require_casetext: if True, only include case in return if case text is present in /original_citations_from_file1t4/
require_metadata: if True, only include case in return if we have metadata about the case in judicial.csv
##CHECK IF ALLCITES CONTAINS ONLY CASES IN METADATA, IS IT BIASED THUS?
'''
    
def load_data(require_casetext = False, require_metadata=False):
    #read in metadata for each case
    judicial = pd.read_csv(data_dir+ 'judicial.csv')
     #read in citations for cases
    allcites = pd.read_csv(data_dir+ 'allcites.txt',sep=' ',names=np.array(['citation','contains'])).dropna()
     # this is the id in the title 
    judicial['usid_nice']=judicial.usid.str.replace('US','_US_') + '.txt'
	
    '''# file0='/Users/gregoryleibon/Citation_Network/Data/original_citations_from_file1t4/153_US_39.txt'
    k=4100
    name_cite=judicial.usid_nice.tolist()[k] 
    print(name_cite)
    ffile='original_citations_from_file1t4'
    ffolder='original_citations_from_folder1t8'
    file0=data_dir + ffile + '/' +name_cite 
    with open (file0, "r") as myfile:
    example_file=myfile.read().replace('\n', '')
    import os
    L=os.listdir("/Users/gregoryleibon/Citation_Network/Data/original_citations_from_file1t4")
    len(L)
    import os
    L=os.listdir("/Users/gregoryleibon/Citation_Network/Data/original_citations_from_folder1t8")
    len(L)
    np.sum(judicial.usid_nice.isin(L))
    import os
    L=os.listdir("/Users/gregoryleibon/Citation_Network/Data/Allen_update")
    len(L)
    '''
     #list of case files in our directory
    Docs_We_Have=os.listdir(data_dir + "original_citations_from_folder1t8")	
     #overlapping set of files in metadata and folder
    Docs_We_Have2=judicial.usid_nice[judicial.usid_nice.isin(Docs_We_Have)]
	# header word distrobution over topic
	# topic distribution of 
	# new post 1946-ish latest and greatest 
	# ~ 7 thousand   
	# Headers 
    phi = pd.read_csv(data_dir+ 'fit-US-flattened-yearprefix-split-munich-phi.csv',index_col=0)
     #topic distribution over case
    theta = pd.read_csv(data_dir+ 'fit-US-flattened-yearprefix-split-munich-theta-merged.csv',index_col=0)
    '''
    theta['fullid']=theta.index
    theta_index=theta.index
    theta['usid'] = theta.apply(lambda row: '_'.join(row['fullid'].split('_')[-1].split('.')[:3]),1)
    theta = theta.reset_index().drop('fullid',1).drop('index',1)
    theta_m = theta.groupby('usid').aggregate(np.mean)      
     #list of doc titles in theta
    '''
    Docs_In_Comp_Data=np.array(theta.index)
    #create data frame with these theta titles plus associated file names
    Dict0={}
    Dict0['usid']=Docs_In_Comp_Data
    Dict0['usid_nice']=Docs_In_Comp_Data+'.txt'
    Docs_In_Comp_Data=pd.DataFrame(Dict0)
    #get overlap between theta titles and earlier titles
    Docs_We_Have3 = Docs_In_Comp_Data.usid_nice[Docs_In_Comp_Data.usid_nice.isin(Docs_We_Have2)]
    #### PERHAPS INSTEAD : Docs_We_Have3 = Docs_In_Comp_Data.usid_nice[Docs_In_Comp_Data.usid_nice.isin(judicial['usid_nice'])]
    #relevant set of metadata
    Our_Docs= judicial[judicial.usid_nice.isin(Docs_We_Have3)]
    #sort metadata by year
    Our_Docs=Our_Docs.sort_index(by='year',ascending=False)
    #create simple index title from mfile name for metadata
    Our_Docs['usid_index']=Our_Docs.usid_nice.str.replace('.txt','')
    #give theta these indices
    theta=theta.join(Our_Docs[['usid_index','usid']].set_index('usid_index'))
    #take all non null titles
    Our_theta=theta[theta.usid.isnull()==False]
    #remove this index
    del Our_theta['usid']
    # restict by topic 
    ##A=np.array(Our_theta)
    ##np.mean(A>0.001) 
    # simple get it down to good topics (45) .... 
    Our_Docs=Our_Docs.set_index('usid')
    # Let's rid via restricted citations 
    # then rid the smallest topics 
    # let ud keep the Topic percent 
    #topics both citing and cited in current set of Docs
    Our_Cites=allcites[(allcites.citation.isin(Our_Docs.caseid)) & (allcites.contains.isin(Our_Docs.caseid))]	
    Our_Cites['temp']=1
    #number of cites in this doc, CHECK
    G=Our_Cites.groupby(['citation'])['temp'].sum() 
    G.name='count_contains'
    G=pd.DataFrame(G)
    #add this count to metadata
    Our_Docs=Our_Docs.join(G,on='caseid')
    #number of times this doc is cited, CHECK
    Our_Cites['temp']=1
    G=Our_Cites.groupby(['contains'])['temp'].sum() 
    G.name='count_citations'
    G=pd.DataFrame(G)
    Our_Docs=Our_Docs.join(G,on='caseid')
    # connect symmetrically! (Consider both directions)
    del Our_Cites['temp']
    Our_Cites_Sym= pd.concat([Our_Cites.rename(columns={'citation':'case1','contains':'case2'}),Our_Cites.rename(columns={'citation':'case2','contains':'case1'})])
    Our_Cites_Sym=Our_Cites_Sym.drop_duplicates(['case1','case2'])
    # Our_Cites_Sym=Our_Cites_Sym.join(J,on='caseid')
    #count connections in both directions ) and add to metadata table
    Our_Cites_Sym['temp']=1
    G=Our_Cites_Sym.groupby(['case1'])['temp'].sum() 
    G.name='count_connectionsA'
    G=pd.DataFrame(G)
    Our_Docs=Our_Docs.join(G,on='caseid')
    Our_Docs['count_connectionsA'][Our_Docs['count_connectionsA'].isnull()]=0
    Our_Cites_Sym['temp']=1
    G=Our_Cites_Sym.groupby(['case2'])['temp'].sum() 
    G.name='count_connectionsB'
    G=pd.DataFrame(G)
    Our_Docs=Our_Docs.join(G,on='caseid')
    Our_Docs['count_connectionsB'][Our_Docs['count_connectionsB'].isnull()]=0
    Our_Docs['count_connections']=Our_Docs['count_connectionsA']+Our_Docs['count_connectionsB']
    ##hist(np.array(Our_Docs['count_connections'])) 
    return [Our_Docs, Our_theta, phi, Our_Cites,  Our_Cites_Sym]

def create_cite_transition(Our_Docs,Our_Cites): 
	# well if I'm keeping them all 
	N= len(Our_Docs) 
     
	W_cited  = Our_Cites.copy()
         #connect trans mat index based on 'citation', i.e. doc doing the citing
	W_cited = join_replace(W_cited,Our_Docs[['caseid','tran_mat_index']],'citation','caseid') 
	W_cited = W_cited.rename(columns={'tran_mat_index':'tran_mat_index_from'})
         #do same for doc being cited
	W_cited = join_replace(W_cited,Our_Docs[['caseid','tran_mat_index']],'contains','caseid') 
	W_cited = W_cited.rename(columns={'tran_mat_index':'tran_mat_index_to'})
         #transition matrix for each document citing all other documents
	W_cited['weight']=1
	T_cited = np.zeros([N,N])
	T_cited[W_cited.tran_mat_index_from.astype(int),W_cited.tran_mat_index_to.astype(int)] = W_cited.weight
	T_cited = normalize(T_cited,N)
 #contruct transition matrix for each document being cited by other documents
	W_cited_by  = Our_Cites.copy()
	W_cited_by = join_replace(W_cited_by,Our_Docs[['caseid','tran_mat_index']],'citation','caseid') 
	W_cited_by = W_cited_by.rename(columns={'tran_mat_index':'tran_mat_index_to'})
	W_cited_by = join_replace(W_cited_by,Our_Docs[['caseid','tran_mat_index']],'contains','caseid') 
	W_cited_by = W_cited_by.rename(columns={'tran_mat_index':'tran_mat_index_from'})
	W_cited_by['weight']=1
	T_cited_by = np.zeros([N,N])
	T_cited_by[W_cited_by.tran_mat_index_from.astype(int),W_cited_by.tran_mat_index_to.astype(int)] = W_cited_by.weight
	T_cited_by = normalize(T_cited_by,N)
	return [W_cited, T_cited, W_cited_by, T_cited_by]

def create_sim_matrix(M_T,M_C,Our_Docs,Our_theta): 
	N= len(Our_Docs) 
	T_sim = np.zeros([N,N])
	Topics = Our_theta.axes[1]
	Cases = Our_theta.axes[0]
	Here=0
	for case0 in Our_theta.index:
		Here=Here+1
		print(str(Here) + " out of " + str(len(Our_theta)))
		#case0 = Our_theta.index.tolist()[0]
		C = np.array(Our_theta[Our_theta.index==case0])[0]
		AS = np.argsort(-C)
		Top_Topics = Topics[AS[0:M_T:1]]
		for topic0 in Top_Topics:
			#topic0=Top_Topics[0]
			D  = np.array(Our_theta[topic0])
			AS = np.argsort(-D)
			Top_Cases_Per_Topic = Cases[AS[0:M_T:1]]
			tran_mat_index_case0 = Our_Docs.tran_mat_index[Our_Docs.usid_index==case0].tolist()[0]
			tran_mat_index_Top_Cases_Per_Topic = Our_Docs.tran_mat_index[Our_Docs.usid_index.isin(Top_Cases_Per_Topic)]
			T_sim[np.array(tran_mat_index_case0),np.array(tran_mat_index_Top_Cases_Per_Topic)] = 1 + T_sim[np.array(tran_mat_index_case0),np.array(tran_mat_index_Top_Cases_Per_Topic)] 
	Ind = np.arange(0,len(T_sim),1)
	T_sim[Ind,Ind] = 0*Ind 
	T_sim = normalize(T_sim,N)
	return T_sim

def master_transition_matrix(T_cited,T_cited_by,T_sim,p_cited,p_cited_by,p_sim):
	T =  p_sim*T_sim + p_cited*T_cited + p_cited_by*T_cited_by
	#np.sum(T,1)
      #set diagnol to zero
	Ind = np.arange(0,len(T),1)
	T[Ind,Ind] = 0*Ind 
	N= len(T)
	T = normalize(T,N)
	return T

def acurate_resolvent(T,r):
	Eye = np.eye(len(T))
	R = np.linalg.solve(Eye-r*T,Eye)
	R_norm = (1-r)*R
	return R_norm  

def form_resolvent(T,r):
	Eye = np.eye(len(T))
	R = np.linalg.solve(Eye-r*T,Eye)
	#Curvature  = np.log(np.diag(R -1))
	return R

def partial_resolvent(T,r,Top):
    N= len(T)
    Tnow=T
    rnow=r
    R= np.eye(N)
    R= R + rnow*Tnow
    for k in np.arange(1,Top,1):
        t0=time.time()
        rnow=rnow*r
        T0 = np.dot(T,Tnow)
        R= R + rnow*T0
        print(str(time.time()-t0))
    Rnorm = ((1-r)/(1-r**(Top+1)))*R 
    return Rnorm 

def compute_page_dist(R_norm): 
	for k in np.arange(0,len(R_norm),1):
		t0=time.time()
		V = R_norm[k,:]
		np.sum(V) 
		Tile_V = np.tile(V.reshape([1,len(V)]),[len(V),1])
		Tile_V.shape 
		Dist_k = np.sum(abs(R_norm - Tile_V),1)
		if k == 0:
			Dist_mat = Dist_k.reshape([1,len(V)])
		else:
			Dist_mat = np.concatenate([Dist_mat,Dist_k.reshape([1,len(V)])])
		print(str(k) + " out of " + str(len(R_norm)))
		print("current size " + str(Dist_mat.shape) )
	return Dist_mat

def compute_page_dist_one_step(R_all,p): 
	# take as step distribution  avoid crazy impact of point
	N=len(R_all) 
	R_1 =  R_all-np.eye(N)
 
	for k in range(0,N):#np.arange(0,len(R_1),1):
		t0=time.time()
		V = R_1[k,:]
		Tile_V = np.tile(V.reshape([1,N]),[N,1])
		#each row is R_i - R_k
		Dist_k = (np.sum(abs(R_1 - Tile_V)**p,1))**(1/(1.0*p))
		if k == 0:
			Dist_mat = Dist_k.reshape([1,len(V)])
		else:
			Dist_mat = np.concatenate([Dist_mat,Dist_k.reshape([1,len(V)])])
             #t1=time.time() - t0
		print(str(k) + " out of " + str(len(R_1)))
		print("current size " + str(Dist_mat.shape) )
	return Dist_mat

def compute_page_dist_one_step_preallocate_l2(R_all,p): 
	# take as step distribution  avoid crazy impact of point
    N=len(R_all) 
    R_1 =  R_all-np.eye(N)
    Distmat = np.zeros([N,N])
    for k in range(0,N):#np.arange(0,len(R_1),1):
        
        V = R_1[k,:]
        Tile_V = np.tile(V.reshape([1,N]),[N,1])
        t1=time.time() -t0
        #each row is R_i - R_k
        t0=time.time()
        Dist_k = (np.sum((R_1 - Tile_V)**p,1))**(1/(1.0*p))
        t1=time.time() -t0
        
        Distmat[k,:] = Dist_k
        
        print(str(k) + " out of " + str(len(R_1)))
        print("current size " + str(Dist_mat.shape) )
    return Dist_mat
 
def compute_page_dist_fast(R_all,p):
    # take as step distribution  avoid crazy impact of point
    N=len(R_all) 
    R_1 =  R_all-np.eye(N)
    
     
def compute_page_dist_one_step_l2(R_all): 
	# take as step distribution  avoid crazy impact of point
   
    N=len(R_all) 
    R_1 =  R_all-np.eye(N)
    Dist_mat = np.zeros([N,N])
    for k in range(0,N):#np.arange(0,len(R_1),1):
        t0=time.time()
        Tile_V = np.tile(R_1[k,:],[N,1])    
       
        Dist_mat[k,:] = np.linalg.norm(R_1 - Tile_V,2,1)
        t1=time.time() - t0
        print(k)
    
    return Dist_mat


def hist_Page_Dist(fig_num, Page_Dist): 
	plt.close(fig_num)
	plt.figure(fig_num)
	plt.hist(Page_Dist[Page_Dist>0],200)
	#title('Page Dist (one step) , MT =' + str(M_T) + ', MC ='   + str(M_C)  + ', r =' + str(r_name)) 
	plt.title('Page Dist on the Legal Corpus, with p=' + str(2))  
	plt.xlabel('Page Dist')
	plt.ylabel('Count')
	plt.xlim([np.min(Page_Dist),0.42])
	# TAKINF A BREAK TO FORM AN EXMAPLE
	return 'done'

def bending_on_date(date0, Drain_Pud_Info, Our_Docs, T_all):
	Window  = Drain_Pud_Info['Window']
	r = Drain_Pud_Info['r']
	n0 = [-np.inf,date0] 
	n1 = [date0+1,date0+ Window] 
	n2 = [date0+Window+1,date0 + 2*Window] 
	# def Explore_Temporal_Impact(n0,n1,n2,key_pertile, Our_Docs, Page_Dist):
	ep=10**(-10)
	Tran2 = Our_Docs.tran_mat_index[(Our_Docs.year>=(n2[0]-ep)) & (Our_Docs.year<=(n2[1]+ep))]
	Tran1 = Our_Docs.tran_mat_index[(Our_Docs.year>=(n1[0]-ep)) & (Our_Docs.year<=(n1[1]+ep))]
	Tran0 = Our_Docs.tran_mat_index[(Our_Docs.year>=(n0[0]-ep)) & (Our_Docs.year<=(n0[1]+ep))]
	T_0 = T_all[Tran0,:]
	T_0 = T_0[:,Tran0]
	T_01  = T_all[Tran0,:]
	T_01 = T_01[:,Tran1]
	T_010 = np.dot(T_01,T_01.transpose())
	T_induced  = normalize(T_0 +  T_010,len(T_0))
	T_restricted = normalize(T_0,len(T_0))
	[R_induced, k_induced] = form_resolvent(T_induced,r)
	[R_restricted, k_restricted] = form_resolvent(T_restricted,r)
	bending = k_induced - k_restricted
	Bending={}
	Bending['bending'] = bending
	Bending['k_restricted'] = k_restricted
	Bending['Tran0'] = Tran0
	Bending['Tran1'] = Tran1
	Bending['Tran2'] = Tran2
	return Bending

def puddling_on_date_fixed_impact(impact_key_pertile, Drain_Pud_Info, Page_Dist, Bending):
	bending_key_pertile = Drain_Pud_Info['bending_key_pertile']
	bending = Bending['bending'] 
	Tran0 = Bending['Tran0']
	Tran1 = Bending['Tran1']
	Tran2 = Bending['Tran2']
	A = Page_Dist[Tran0,:]
	A = A[:,Tran2]
	M = A.min(1)
	p_5 = np.percentile(M,impact_key_pertile)
	Impact2 = Tran0[(M<p_5)]
	p_5_c = np.percentile(M,(100-impact_key_pertile))
	Impact2_Low = Tran0[(M>p_5_c)]
	A = Page_Dist[Tran0,:]
	A = A[:,Tran1]
	M = A.min(1)
	p_5 = np.percentile(M,impact_key_pertile)
	Impact1 = Tran0[(M<p_5)]
	p_5_c = np.percentile(M,(100-impact_key_pertile))
	Impact1_Low = Tran0[(M>p_5_c)]
	p_q1 = np.percentile(bending,bending_key_pertile)
	p_q3 = np.percentile(bending,100-bending_key_pertile)
	Draiange = Tran0[bending<=p_q1]
	Puddling = Tran0[bending>=p_q3]
	Coincident_High_Impact = 100*(1-len(set(Impact1).difference(set(Impact2)))/(1.0*len(Impact1)))
	I1D = set(Impact1).intersection(set(Draiange))
	I1P = set(Impact1).intersection(set(Puddling))
	Impact_Data = {}
	Impact_Data['Impact1'] = Impact1
	Impact_Data['Impact1_Low'] = Impact1_Low
	Impact_Data['Impact2'] = Impact2
	Impact_Data['Impact2_Low'] = Impact2_Low
	Impact_Data['I1D'] =I1D
	Impact_Data['I1P'] = I1P
	Impact_Data['Coincident_High_Impact'] = Coincident_High_Impact
	Impact_Data['p_q1'] = p_q1
	Impact_Data['p_q3'] = p_q3
	return Impact_Data

def puddles_drainage_over_time(Drain_Pud_Info, T_all, Page_Dist, Our_Docs):
	date0_list = Drain_Pud_Info['date0_list']
	impact_key_pertile_list = Drain_Pud_Info['impact_key_pertile_list']
	Window  = Drain_Pud_Info['Window']
	bending_key_pertile = Drain_Pud_Info['bending_key_pertile']
	PIm2GIm1 = {}
	PIm2GIm1D = {}
	PIm2GIm1P = {}
	for date0 in date0_list:
		Bending = bending_on_date(date0, Drain_Pud_Info, Our_Docs, T_all)
		PIm2GIm1[str(date0)] = []
		PIm2GIm1D[str(date0)] = []
		PIm2GIm1P[str(date0)] = []
		for impact_key_pertile in impact_key_pertile_list:
			Impact_Data = puddling_on_date_fixed_impact(impact_key_pertile, Drain_Pud_Info, Page_Dist, Bending )
			PIm2GIm1[str(date0)].append(100*len(set(Impact_Data['Impact2']).intersection(set(Impact_Data['Impact1'])))/(1.0*len(Impact_Data['Impact1'])))
			print('Date:' + str(date0) + ', Predict Impact:' + str(PIm2GIm1[str(date0)][-1]) + 'if random would be '+ str(impact_key_pertile))
			PIm2GIm1D[str(date0)].append(100*len(set(Impact_Data['Impact2']).intersection(set(Impact_Data['I1D'])))/(1.0*len(Impact_Data['I1D'])))
			print('Date:' + str(date0) + ', Predict Impact Darinage:' + str(PIm2GIm1D[str(date0)][-1]) + 'if random would be '+ str(impact_key_pertile))
			PIm2GIm1P[str(date0)].append(100*len(set(Impact_Data['Impact2']).intersection(set(Impact_Data['I1P'])))/(1.0*len(Impact_Data['I1P'])))
			print('Date:' + str(date0) + ', Predict Impact Puddling:' + str(PIm2GIm1P[str(date0)][-1]) + 'if random would be '+ str(impact_key_pertile))
	for date0 in date0_list:
		PIm2GIm1[str(date0)] = np.array(PIm2GIm1[str(date0)])
		PIm2GIm1D[str(date0)] = np.array(PIm2GIm1D[str(date0)])
		PIm2GIm1P[str(date0)] = np.array(PIm2GIm1P[str(date0)])
	Drain_Pud_Info['PIm2GIm1'] = PIm2GIm1
	Drain_Pud_Info['PIm2GIm1P']  = PIm2GIm1P
	Drain_Pud_Info['PIm2GIm1D']  = PIm2GIm1D 
	# notice smaple from last date (stupid)
	Drain_Pud_Info['bending']  = Bending['bending']
	Drain_Pud_Info['k_restricted']  = Bending['k_restricted']
	return Drain_Pud_Info

def get_example(Our_Docs, Page_Dist, Per_Small, Per_Large):
		TF =  ((Page_Dist>=Per_Large))
		indexes=np.indices(Page_Dist.shape)
		Ind=indexes[:,TF]
		l=len(Ind[0])
		M = np.concatenate([Ind[0].reshape(l,1),Ind[1].reshape(l,1)],axis=1)
		Our_Docs[['usid_index','tran_mat_index','count_connections']][0:10:1]
		F=pd.DataFrame({'tran_mat_index1':M[:,0],'tran_mat_index2':M[:,1]})
		F['score']= Page_Dist[M[:,0],M[:,1]]
		Total_Close = np.sum((Page_Dist<Per_Small),1)
		F['total_close1'] = Total_Close[Ind[0].reshape(l,1)]
		F['total_close2'] = Total_Close[Ind[1].reshape(l,1)]
		F = join_replace(F,Our_Docs[['usid_index','tran_mat_index','count_connections']],'tran_mat_index1','tran_mat_index')
		F=F.rename(columns={'usid_index':'usid_index1','count_connections':'count_connections1'})
		F = join_replace(F,Our_Docs[['usid_index','tran_mat_index','count_connections']],'tran_mat_index2','tran_mat_index')
		F=F.rename(columns={'usid_index':'usid_index2','count_connections':'count_connections2'})
		return F

def get_exmaple_regions(Page_Dist, Our_Docs, F,Per_Small, Per_Large0):
	Cut = 3 
	ind0 = np.round(np.random.uniform(0,len(F)))
	print(ind0)
	tA = F.tran_mat_index1.tolist()[int(ind0)]
	tB = F.tran_mat_index2.tolist()[int(ind0)]
	TFA = (Page_Dist[tA,:]<Per_Small) & (Page_Dist[tB,:]>Per_Large0)
	indexes=np.indices(Page_Dist.shape)
	indA = indexes[1,0,TFA]
	indApd=pd.DataFrame({'tran_mat_index':indA})
	indApd['Region'] = 'A'
	indApd['dist_to_A'] = Page_Dist[tA,indA] 
	indApd['dist_to_B'] = Page_Dist[tB,indA]
	indApd=indApd.sort_index(by = 'dist_to_A')
	indApd=indApd[0:Cut:1]
	TFB = (Page_Dist[tB,:]<Per_Small) & (Page_Dist[tA,:]>Per_Large0)
	indexes=np.indices(Page_Dist.shape)
	indB =indexes[1,0,TFB]
	indBpd=pd.DataFrame({'tran_mat_index':indB})
	indBpd['Region'] = 'B'
	indBpd['dist_to_A'] = Page_Dist[tA,indB] 
	indBpd['dist_to_B'] = Page_Dist[tB,indB]
	indBpd=indBpd.sort_index(by = 'dist_to_B')
	indBpd=indBpd[0:Cut:1]
	Info = pd.concat([indApd,indBpd])
	Info = join_replace(Info,Our_Docs[['usid_index','tran_mat_index','year','parties']],'tran_mat_index')
	MN = np.min(np.array([np.sum(TFA),np.sum(TFB)]))
	if MN<Cut: print("Too small")
	print(Info.to_string())
	return Info
 
def graphs_from_papers(fig_num, date0, Pick_Out, Drain_Pud_Info ):
	def graph_drainage_effect(fig_num, date0, impact_key_pertile_list, PIm2GIm1, PIm2GIm1P, PIm2GIm1D  ):
		plt.close(fig_num)
		plt.figure(fig_num)
		plt.plot(impact_key_pertile_list,PIm2GIm1[str(date0)],'b') 
		plt.plot(impact_key_pertile_list,PIm2GIm1P[str(date0)],'r')
		plt.plot(impact_key_pertile_list,PIm2GIm1D[str(date0)],'k')
		plt.title("The Drainage and Puddling Effect")
		plt.xlabel("Initial Impact Probability")
		plt.ylabel("Future Impact Probability")
		plt.xlim([0,55])
		return "done" 
	def graph_increase_over_rand(fig_num, date0, impact_key_pertile_list, PIm2GIm1):
		plt.close(fig_num)
		plt.figure(fig_num)
		X=np.array(PIm2GIm1[str(date0)])-np.array(impact_key_pertile_list)
		plt.plot(np.array(impact_key_pertile_list),X,'b') 
		#plot([20,20],[0,np.max(X)],'r')
		plt.title("Increase over Random due to Impact")
		plt.xlabel("Initial Impact Probability")
		plt.ylabel("Increase")
		return "done" 
	def graph_imapct_as_determiend_by_effect(Pick_Out, date0_list, fig_num, date0, impact_key_pertile_list, PIm2GIm1, PIm2GIm1P, PIm2GIm1D):
		print("Pertile we are using " + str(impact_key_pertile_list[Pick_Out]))
		Im1=[]
		Im1P=[]
		Im1D=[]
		for date0 in date0_list:
			Im1.append(PIm2GIm1[str(date0)][Pick_Out])
			Im1P.append(PIm2GIm1P[str(date0)][Pick_Out])
			Im1D.append(PIm2GIm1D[str(date0)][Pick_Out])
		Im1 = np.array(Im1)
		Im1P = np.array(Im1P)
		Im1D = np.array(Im1D)
		# could restrict!
		TF = (date0_list>=1979) 
		plt.close(fig_num)
		plt.figure(fig_num)
		plt.plot(date0_list[TF],Im1[TF],'b') 
		plt.plot(date0_list[TF],Im1P[TF],'r')
		plt.plot(date0_list[TF],Im1D[TF],'k')
		# title("The Drainage Effect 2 (red: drainage, black: puddling, Impact = "+ str(impact_key_pertile_list[Pick_Out]) + ")")
		plt.title("The Drainage Effect")
		plt.xlabel("Date")
		plt.ylabel("Future Impact Probability")
		return "done"
	def bending_hist(fig_num, bending):
		# there is real bending 
		plt.close(fig_num)
		plt.figure(fig_num)
		Bins = 100
		H = plt.hist(bending[abs(bending)<10**15],Bins)
		#plot([p_q1,p_q1],[0,np.max(H[0])])
		#plot([p_q3,p_q3],[0,np.max(H[0])])
		plt.hist(bending[abs(bending)<10**15],Bins)
		plt.xlim([np.min(bending[abs(bending)<10**15]),1.5])
		plt.ylim([0,np.max(H[0])])
		plt.title('Bending')
		plt.xlabel("Bending")
		plt.ylabel("Count")
		return "done" 
	def curvature_hist(fig_num, bending):
		# there is real bending 
		plt.close(fig_num)
		plt.figure(fig_num)
		Bins = 50
		H = plt.hist(k_restricted[abs(k_restricted)<10**15],Bins)
		#plot([p_q1,p_q1],[0,np.max(H[0])])
		#plot([p_q3,p_q3],[0,np.max(H[0])])
		plt.hist(k_restricted[abs(k_restricted)<10**15],Bins)
		plt.xlim([-7,-2])
		plt.ylim([0,np.max(H[0])])
		plt.title('Curvature')
		plt.xlabel('Curvature')
		plt.ylabel("Count")
		return "done"
	impact_key_pertile_list = Drain_Pud_Info['impact_key_pertile_list'] 
	PIm2GIm1  = Drain_Pud_Info['PIm2GIm1']
	PIm2GIm1P = Drain_Pud_Info['PIm2GIm1P']
	PIm2GIm1D = Drain_Pud_Info['PIm2GIm1D']
	bending = Drain_Pud_Info['bending']
	k_restricted = Drain_Pud_Info['k_restricted']
	date0_list = Drain_Pud_Info['date0_list']
	fig_num0 = fig_num
	graph_drainage_effect(fig_num0, date0, impact_key_pertile_list, PIm2GIm1, PIm2GIm1P, PIm2GIm1D  )
	fig_num0 = fig_num0+1
	graph_increase_over_rand(fig_num0, date0, impact_key_pertile_list, PIm2GIm1)
	fig_num0 = fig_num0+1
	graph_imapct_as_determiend_by_effect(Pick_Out, date0_list, fig_num0, date0, impact_key_pertile_list, PIm2GIm1, PIm2GIm1P, PIm2GIm1D)
	fig_num0 = fig_num0+1
	bending_hist(fig_num0, bending)
	fig_num0 = fig_num0+1
	curvature_hist(fig_num0, k_restricted)
	return "done"

def plot_date_differecne_high_low_impact(fig_num, Impact_Data, Our_Docs): 
	plt.close(fig_num)
	plt.figure(fig_num) 
	Y_Low = np.array(Our_Docs.year[Our_Docs.tran_mat_index.isin(Impact_Data['Impact1_Low'])])
	Y_Low = np.sort(Y_Low)
	prob_Low = 100*np.arange(0,len(Y_Low),1)/(1.0*len(Y_Low))
	Y_High = np.array(Our_Docs.year[Our_Docs.tran_mat_index.isin(Impact_Data['Impact1'])])
	Y_High = np.sort(Y_High)
	prob_High = 100*np.arange(0,len(Y_High),1)/(1.0*len(Y_High))
	plt.plot(Y_Low,prob_Low)
	plt.plot(Y_High,prob_High,'r')
	plt.title('Cases Impacted by cases in (1990,1995]')
	plt.xlabel('Year')
	plt.ylabel('Percent of impacted cases seen so far')
	return "done" 


def collect_interesting_date_ex(High_Or_Low, n_now, n_then, Our_Docs, Page_Dist, Keep):
	# collect some exmaples 
	ep=10**(-10)
	Tran_now = Our_Docs.tran_mat_index[(Our_Docs.year>=(n_now[0]-ep)) & (Our_Docs.year<=(n_now[1]+ep))]
	Tran_then = Our_Docs.tran_mat_index[(Our_Docs.year>=(n_then[0]-ep)) & (Our_Docs.year<=(n_then[1]+ep))]
	A = Page_Dist[Tran_then,:]
	A = A[:,Tran_now]
	M = A.min(1)
	AS = np.argsort(M)
	tran_thenS = []
	tran_now_0S = []
	tran_now_1S = []
	tran_now_2S = []
	value_0S = []
	value_1S = []
	value_2S = []
	if High_Or_Low=='High':
		range_it = np.arange((len(AS)-Keep),len(AS),1)
	else:
		range_it = np.arange(0,Keep,1)
	for k in range_it:
		extreme_value = M[AS[k]]
		tran_then = Tran_then[AS[k]]
		AS2 = np.argsort(Page_Dist[Tran_then[AS[k]],Tran_now])
		tran_now_0 = Tran_now[AS2[0]]
		value_0 = Page_Dist[tran_then,tran_now_0]
		tran_now_1 = Tran_now[AS2[1]]
		value_1 = Page_Dist[tran_then,tran_now_1]
		tran_now_2 = Tran_now[AS2[2]]
		value_2 = Page_Dist[tran_then,tran_now_2]
		print [extreme_value, value_0, value_1, value_2]
		tran_thenS.append(tran_then)
		tran_now_0S.append(tran_now_0)
		tran_now_1S.append(tran_now_1)
		tran_now_2S.append(tran_now_2)
		value_0S.append(value_0)
		value_1S.append(value_1)
		value_2S.append(value_2)
	Exmaple_Date  = pd.DataFrame({'tran_then':tran_thenS,
		'tran_now_0':tran_now_0S,'value_0':value_0S,
		'tran_now_1':tran_now_1S,'value_1':value_1S,
		'tran_now_2':tran_now_2S,'value_2':value_2S})
	return Exmaple_Date 

def join_and_rename(name, Exmaple_Date, Our_Docs ): 
	Example_Date = Exmaple_Date.join(Our_Docs[['tran_mat_index','year','parties','caseid','usid_nice']].set_index('tran_mat_index'), on = ('tran'+name))
	Example_Date = Exmaple_Date.rename(columns={'year':('year'+name),'parties':('parties'+name),'caseid':('caseid'+name),'usid_nice':('usid_nice'+name)})
	return Example_Date
 
 
 