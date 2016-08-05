# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:31:11 2015

@author: d29905p
"""

import numpy as np
import pandas as pd
import time
import itertools
import pickle
from sklearn.manifold import MDS
import os
os.chdir("O:/documents/bendinglaw/src/bending_the_law/scripts/")
from bendinglaw import *

'''
script to create files necessary for Markov chains, PageDist and metadata for analysis and web application

'''


data_dir='C:/users/d29905p/documents/bendinglaw/'

#load citation and LDA data
[Our_Docs, Our_theta, phi, Our_Cites,  Our_Cites_Sym] = load_data()


#create citation transition matrices
Our_Docs['tran_mat_index'] = np.arange(0,len(Our_Docs),1)
[W_cited, T_cited, W_cited_by, T_cited_by] = create_cite_transition(Our_Docs,Our_Cites)


#create titles dictionary (case to transition matrix index)
Our_Docs['usid_title'] = Our_Docs.apply(lambda x: str(x['usid_index']).replace('_',' ') + ': ' + str(x['parties']),1)
O=Our_Docs[['tran_mat_index','parties','year','usid_index','usid_title']].to_dict('records')
O_dict = {key:o for key,o in zip(range(0,len(O)),O)}
pickle.dump(O_dict, open(data_dir+'titles2_rev.p','wb'),2)
#create titleskey dictionary (transition matrix index to metadata dirctionary )
titleskey = {record['usid_title']:key for key, record in O_dict.items()}
pickle.dump(titleskey, open(data_dir+'titleskey2_rev.p','wb'),2)
#create metadata dictionary (case title to metadata)
Our_Docs['count_contains'] = Our_Docs['count_contains'].replace(np.nan,0)
Our_Docs['count_citations'] = Our_Docs['count_citations'].replace(np.nan,0)
K= Our_Docs[['tran_mat_index','parties','year','count_contains','count_citations','usid_index','usid_title']].to_dict('records')
K_dict = {record['usid_title']:record for record in K}
pickle.dump(K_dict, open(data_dir+'metadata2_rev.p','wb'),2)



#create or load similarity matrix
Create_Sim = True
M_T = 10
M_C = 10
if Create_Sim:
    t0=time.time()		
    T_sim = create_sim_matrix(M_T,M_C,Our_Docs,Our_theta)
    print(str(time.time()-t0) + " time for, M_T= " + str(M_T) + " , and M_C = " + str(M_C))
    pickle.dump(T_sim,open(data_dir+ 'T_sim2_rev' + '_' + str(M_T) + '_'  + str(M_C)  + '.p', 'wb'))
else: 
    T_sim=pickle.load(open(data_dir+ 'T_sim2_rev' + '_' + str(M_T) + '_'  + str(M_C)  + '.p', 'rb'))

# set r value (see paper)	

r = 1/2.0
r_name = '1Over2'
L_p = 2

#enumerate weighting combinations
combos = [[1,1,1]]    
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            if not(i==j and j==k):
                combos.append([i,j,k])
                
            
                
#create PageDistMatrices (will take a long time, ~hour per weighting)
for p_set in combos:
     p_cited = p_set[0]/sum(p_set)
     p_cited_by = p_set[1]/sum(p_set)
     p_sim = p_set[2]/sum(p_set)
     T_all = master_transition_matrix(T_cited,T_cited_by,T_sim,p_cited,p_cited_by,p_sim)
     
		## r = 1/2.0
		##R_norm   = acurate_resolvent(T_all,r)
     R_all= form_resolvent(T_all,r)
		## Page_Dist_Original = compute_page_dist(R_norm) 
     Page_Dist = compute_page_dist_one_step(R_all,L_p)
     ##pickle.dump(T_all,open(data_dir+ 'T_all' + '_' + '_'.join([str(p) for p in p_set]) +'_' + str(M_T) + '_'  + str(M_C)  + '.p', 'wb'))
     ##pickle.dump(R_all,open(data_dir+ 'R_all' + '_' + '_'.join([str(p) for p in p_set]) +'_' +str(M_T) + '_'  + str(M_C)+'_' + str(r_name) + '.p', 'wb'))
     pickle.dump(Page_Dist, open(data_dir+ 'Page_Dist2_rev_1' + '_' + '_'.join([str(p) for p in p_set]) +'_' + str(M_T) + '_'  + str(M_C)+ '_'  + str(L_p) +'_' + str(r_name) + '.p', 'wb'))

    


#create lookup dictionaries for top 30 most similar cases, storing simliar cases and scores thereof for each case in order of similarity 
for combo in combos:
    print((combo[0],combo[1],combo[2]))
    #get page dist matrix
    infile = 'Page_Dist2_rev_1_%s_%s_%s_10_10_2_1Over2.p' % (combo[0],combo[1],combo[2])
    Dist_mat = pickle.load(open(data_dir + infile,'rb'))
    N=Dist_mat.shape[0]
    #file names for output titles and scores
    outfile_titles = 'case_lookup_rev_%s%s%s.p'% (combo[0],combo[1],combo[2])
    outfile_scores = 'case_lookup_rev_scores_%s%s%s.p'% (combo[0],combo[1],combo[2])
    
    rows = []   
    scores = []
    #for each case...
    for row_ind in range(0,N):
        #get associated case row.
        row = Dist_mat[row_ind,:]
        #get 30 cases with minimum page distance
        how_many = 30
        top_N = np.argpartition(row, how_many)[:how_many]
        top_N_sorted=top_N[np.argsort(row[top_N])]
        #append indices of these cases
        rows.append(top_N_sorted)
        #append scores of these cases
        scores.append(row[top_N_sorted])
    # convert to numpy arrays  
    dist_lookup=np.array(rows)
    score_lookup = np.array(scores)
    #dictionaries for linking case index to case indices and respective scores of top 30 most similar cases
    dist_dict = {key:dist_lookup[key,:].tolist() for key in range(0,dist_lookup.shape[0])}
    dist_dict_scores = {key:score_lookup[key,:].tolist() for key in range(0,dist_lookup.shape[0])}
    ##pickle.dump(dist_dict, open(data_dir+'case_lookup.p','wb'))
    pickle.dump(dist_dict, open(data_dir+outfile_titles,'wb'),2)
    pickle.dump(dist_dict_scores, open(data_dir+outfile_scores,'wb'),2)



#get case all case pairs relevant for a certain rating and store their similarity scores
#revise in several different formats for efficient storage
for combo in combos:
    print(combo)
    infile = 'Page_Dist_1_%s_%s_%s_10_10_2_1Over2.p' % (combo[0],combo[1],combo[2])
    Dist_mat = pickle.load(open(data_dir + infile,'rb'))
    N=Dist_mat.shape[0]
    infile_titles = 'case_lookup_%s%s%s.p'% (combo[0],combo[1],combo[2])
    dist_dict = pickle.load(open(data_dir + infile_titles,'rb'))
    full_dist_pairs=set()
    t0=time.time()
    for row_ind in range(0,N):       
            
        full_dist_pairs = full_dist_pairs | set(itertools.combinations(dist_dict[row_ind],2))
    t1=time.time()-t0
    print(len(full_dist_pairs))
    full_dist_pairs = list(set([tuple(sorted(pair)) for pair in full_dist_pairs if pair[0]!=pair[1]]))            
    print(len(full_dist_pairs))
    print(t1)
    score_list = [list(pair) + [Dist_mat[pair[0],pair[1]]] for pair in full_dist_pairs]       
    
    pickle.dump(score_list, open(data_dir+'row_col_dist_sort_%s%s%s.p'% (combo[0],combo[1],combo[2]),'wb'),2)

#revise row col dists
for combo in combos:
   score_list = pickle.load(open(data_dir+'row_col_dist_sort_%s%s%s.p'% (combo[0],combo[1],combo[2]),'rb'))
   print('reducing...')
   print(len(score_list))
   newlist = pd.DataFrame(score_list).drop_duplicates(subset=[0,1]).values.tolist()               
   print(len(newlist))
   pickle.dump(newlist, open(data_dir+'row_col_dist_sortunique_%s%s%s.p'% (combo[0],combo[1],combo[2]),'wb'),2)
#make row col a numpy
for combo in combos:
   score_list = pickle.load(open(data_dir+'row_col_dist_sortunique_%s%s%s.p'% (combo[0],combo[1],combo[2]),'rb'))
   
   npy = np.array(score_list)               
   npy.dump(data_dir+'row_col_dist_%s%s%s.npy'% (combo[0],combo[1],combo[2]))
#make numpy a dict
for combo in combos:
   npy=np.load(data_dir+'row_col_dist_%s%s%s.npy'% (combo[0],combo[1],combo[2]))               
   dict_npy = {(int(row[0]),int(row[1])):row[2] for row in npy}  
   pickle.dump(dict_npy,open(data_dir+'row_col_dict_%s%s%s.p'% (combo[0],combo[1],combo[2]), 'wb'), 2)
# nump to int
for combo in combos:
    npy=np.load(data_dir+'row_col_dist_%s%s%s.npy'% (combo[0],combo[1],combo[2]))               
    if npy.max()<65000:   
        npy[:,2]*=100000
        new_mat = npy.astype('uint16')
        new_mat.dump(data_dir+'row_col_dist_int_%s%s%s.npy'% (combo[0],combo[1],combo[2]))
    else:
        print('ERROR')
        print(combo)
'''
#save above as text  
for combo in combos:
    print(combo)
    npy=np.load(data_dir+'row_col_dist_int_%s%s%s.npy'% (combo[0],combo[1],combo[2]))
    np.savetxt(data_dir +'row_col_dist_int_%s%s%s.txt' % (combo[0],combo[1],combo[2]),npy,fmt='%i')
'''

#transform citations to trans_mat index, save citation data as a text file
docs_id_indexed = Our_Docs.set_index('caseid')
docs_id_indexed['trans_index'] = list(range(0,docs_id_indexed.shape[0]))
def get_transdex(cell):
    try:
        return docs_id_indexed.loc[cell]['trans_index']
    except KeyError:
        return np.nan
trans_cites = Our_Cites.applymap(get_transdex).as_matrix()
trans_cites.dump(data_dir+'cite_network.npy')
cite_net = np.load(data_dir+'cite_network.npy') 
np.savetxt(data_dir +'cite_network.txt',cite_net, fmt='%i')


#load data for test MDS
combo = [1,1,1]
tcit = combo[0]
tcitby = combo[1]
tsim = combo[2]        
query = '410 US 113: Roe v. Wade'
titleskey=pickle.load(open(data_dir+'titleskey.p','rb'))
titles=pickle.load(open(data_dir+'titles.p','rb'))        
#title to metadata dict
metadata =pickle.load(open(data_dir+'metadata.p','rb'))
#find relevant similar cases
if tcit == tcitby and tcitby == tsim:
    #index to top similar indices
    case_lookup =pickle.load(open(data_dir+ 'case_lookup_111.p','rb'))
else:
    case_lookup =pickle.load(open(data_dir+ 'case_lookup_%s%s%s.p' % (tcit,tcitby,tsim),'rb'))
#get ids of cases
caselist_ids = case_lookup[titleskey[query]]
caselist = [titles[caseid]['usid_title'] for caseid in caselist_ids]
meta = metadata[query]
data = {'caselist':caselist}
data.update(meta)
#get possible relevant distances from database
   #case_pairs=list(itertools.combinations(caselist_ids,2))
##reduced_dist_table = pickle.load(open(data_dir + 'row_col_dist_sortunique_%s%s%s.p' % (combo[0],combo[1],combo[2]),'rb'))
##dist_frame=pd.DataFrame(reduced_dist_table).groupby([0,1])
#get relevant page distances for cases abobe
npy_dists = np.load(data_dir+'row_col_dist_int_%s%s%s.npy'% (combo[0],combo[1],combo[2]))
reduced_dists = npy_dists[np.in1d(npy_dists[:,0], caselist_ids)&np.in1d(npy_dists[:,1],caselist_ids)]
#create distmat
dist_mat = np.zeros([30,30])          
mat_indices = [sorted(pair) for pair in itertools.combinations(range(0,30),2) if pair[0]!=pair[1]]
for inds in mat_indices:
    i=inds[0]
    j=inds[1]           
    #dist = dist_frame.get_group(tuple(sorted([caselist_ids[i],caselist_ids[j]]))).iloc[0,2]
    new_inds = sorted([caselist_ids[i],caselist_ids[j]])
    dist=reduced_dists[np.logical_and(reduced_dists[:,0]==new_inds[0], reduced_dists[:,1]==new_inds[1])][0,2]
    dist_mat[i,j]=dist
    dist_mat[j,i]=dist
#set MDS object
mds=  MDS(eps=1e-6,dissimilarity='precomputed')
#scale, 2d
pos = mds.fit(dist_mat).embedding_
#get nodes in json dict format
nodes = []
for i in range(0,30):
    case_data = titles[caselist_ids[i]]
    #query case vs all others
    group = 1 if i==0 else 2
    value = 3 if i==0 else 2
    nodes.append({'id':i+1, 'label': case_data['parties'], 'title':'Case ID: ' + case_data['usid_index'].replace('_','') + '<br>' + 'Year: ' + str(case_data['year']),'value': value, 'group':group,'x':pos[i,0], 'y':pos[i,1]} )
#get edges in json dict
trans_cites = np.load(data_dir+'cite_network.npy')
reduced_trans_cites  = trans_cites[np.in1d(trans_cites[:,0], caselist_ids) & np.in1d(trans_cites[:,1], caselist_ids) ]       
edges = [{'from': cite[0],'to':cite[1]} for cite in reduced_trans_cites.tolist()]
'''    
D=set(itertools.combinations(dist_dict[row_ind],2))
MAT = np.zeros((30,30))
for i in dist_dict[row_ind]:
for j in dist_dict[row_ind]:
    
for pair in D:
   ''' 
#test s3 bucket
from boto.s3.connection import S3Connection
conn = S3Connection('AKIAJCRXIMEOSHFWBFTQ','uQvi9IshPQcTga7hZq4CCJEpOmnH/flQrTuDlgUS')
bucket = conn.get_bucket('bendinglawbucket')
combo = [1,1,1]
from io import BytesIO
k=np.load(BytesIO(bucket.get_key("row_col_dist_int_%s%s%s.npy"% (combo[0],combo[1],combo[2])).get_contents_as_string()))



#create titles, titleskey, and metadata files
Our_Docs['usid_title'] = Our_Docs.apply(lambda x: str(x['usid_index']).replace('_',' ') + ': ' + str(x['parties']),1)
O=Our_Docs[['tran_mat_index','parties','year','usid_index','usid_title']].to_dict('records')
O_dict = {key:o for key,o in zip(range(0,len(O)),O)}
pickle.dump(O_dict, open(data_dir+'titles2.p','wb'),2)
titleskey = {record['usid_title']:key for key, record in O_dict.items()}
pickle.dump(titleskey, open(data_dir+'titleskey2.p','wb'),2)

Our_Docs['count_contains'] = Our_Docs['count_contains'].replace(nan,0)
Our_Docs['count_citations'] = Our_Docs['count_citations'].replace(nan,0)
K= Our_Docs[['tran_mat_index','parties','year','count_contains','count_citations','usid_index','usid_title']].to_dict('records')

K_dict = {record['usid_title']:record for record in K}
pickle.dump(K_dict, open(data_dir+'metadata2.p','wb'),2)


#remove titles from metadata and titles key files
meta2 = {key.split(':')[0]:value for key, value in metadata.items()}
tk = {key.split(':')[0]:value for key, value in titleskey.items()}
pickle.dump(meta2, open(data_dir+ 'metadata_2.p','wb'),2)
pickle.dump(tk, open(data_dir +'titleskey_2.p','wb'),2)



#####


# -*- coding: utf-8 -*-
"""
Created on Wed Mar 16 17:50:40 2016

@author: d29905p
"""
import os
import numpy as np
os.chdir("O:/Documents/bendinglaw/src/bending_the_law/lawsite_nogit")
rcd = np.load('row_col_dist_int_010.npy')
import pickle
cl=pickle.load(open('case_lookup_010.p', 'rb'))
cl=pickle.load(open('case_lookup_010.p', 'rb'))

combos = [[1,1,1]]    
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            if not(i==j and j==k):
                combos.append([i,j,k])
                
redundancy_dict = {}
already_represented = []
for combo in combos:
    if [c/2 for c in combo] in already_represented:
        redundancy_dict[tuple(combo)]= [c/2 for c in combo]
    else:
        redundancy_dict[tuple(combo)] = combo
        already_represented.append(combo)

