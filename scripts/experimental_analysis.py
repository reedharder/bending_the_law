# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 11:51:54 2015

@author: d29905p
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

data_dir='C:/users/d29905p/documents/bendinglaw/'


student_data = pd.read_csv(data_dir +'student_data1.csv' )
titleskey=pickle.load(open(data_dir+'titleskey2.p','rb'))
titles=pickle.load(open(data_dir+'titles2.p','rb'))        
#title to metadata dict
metadata =pickle.load(open(data_dir+'metadata2.p','rb'))
#dict to lookup by US index
id_title_lookup = {value['usid_index']:key for key, value in titles.items()}
#clean up student inputs
def clean_case(cell):
    return "_".join(cell.replace('U.S.','US').split()[:3])
def case_index(cell):
    try:
        return id_title_lookup[cell]
    except KeyError:
        print(cell)
        return -9999
student_data.iloc[:,1:]=student_data.iloc[:,1:].applymap(clean_case)
student_data_index = student_data.copy()
student_data_index.iloc[:,1:]=student_data_index.iloc[:,1:].applymap(case_index)
targets = list(set(student_data['Target Case'].tolist()))

#get overlapping case count
stud_grps = student_data.iloc[:,1:].groupby('Target Case')
for target in targets:
    cases = stud_grps.get_group(target).drop('Target Case',axis=1)
    cases = pd.Series(cases.values.flatten())
    print(cases.value_counts())

#combos to sort through
combos = [[1,1,1]]    
for i in range(0,3):
    for j in range(0,3):
        for k in range(0,3):
            if not(i==j and j==k):
                combos.append([i,j,k])

#intialize Dataframes for each target case               
df_dict = {}
stud_grps = student_data.iloc[:,1:].groupby('Target Case')
for target in targets:
    cases = stud_grps.get_group(target).drop('Target Case',axis=1)
    cases = pd.Series(cases.values.flatten())
    ##print(cases.value_counts())
    df =  pd.DataFrame(cases.value_counts()).reset_index()
    df['found_count'] = df[0]
    df = df.drop(0,1)
    df['found_case'] = df['index']
    df = df.drop('index',1)
    df_dict[target] = df

#function to get rank of found case given by our algorithm for a given target case and weighting
def get_rank(row,rowindex):
    try:
        return np.where(dist_lookup[rowindex,:]==id_title_lookup[row['found_case']])[0]
    except KeyError:        
        return -9999
        
#for each combo and target case, find rank of student fo
for combo in combos:
    
    infile = 'Page_Dist2_1_%s_%s_%s_10_10_2_1Over2.p' % (combo[0],combo[1],combo[2])
    try:
        Dist_mat = pickle.load(open(data_dir + infile,'rb'))
    except:
        print(combo)
    else:
        N=Dist_mat.shape[0]
        stud_grps = student_data.groupby('Target Case')
        rows = []   
        scores = []
        for row_ind in [id_title_lookup[case] for case in targets]:
            row = Dist_mat[row_ind,:]
            how_many = N-1
            top_N = np.argpartition(row, how_many)[:how_many]
            top_N_sorted=top_N[np.argsort(row[top_N])]
            rows.append(top_N_sorted)
            scores.append(row[top_N_sorted])
        dist_lookup=np.array(rows)
        score_lookup = np.array(scores)
        
        dist_dict = {key:dist_lookup[key,:].tolist() for key in range(0,dist_lookup.shape[0])}
        dist_dict_scores = {key:score_lookup[key,:].tolist() for key in range(0,dist_lookup.shape[0])}
        for i,target in enumerate(targets):
            df = df_dict[target]
            df['_'.join([str(cm) for cm in combo])] = df.apply(get_rank,axis=1,rowindex=i).iloc[:,0]
            df_dict[target] = df
#save these dataframes to csv
for target in targets:
    df_dict[target].to_csv(data_dir + 'target_' + target +'.csv')
    
#load data frames, perform overlap analysis (count of cases less than 20)
    
for target in targets:
     df = pd.read_csv(data_dir + 'target_' + target +'.csv')
     shared_df = df.copy()
     shared_df.iloc[:,3:] = df.iloc[:,3:].applymap(lambda x: 1 if x<=20 else 0)
     shared_df.iloc[:,3:] = shared_df.apply(lambda x: x.iloc[3:]*x['found_count'],1)       
     out_df=df.append(pd.Series([np.nan, np.nan, 'Shared with cutoff 20'] + shared_df.iloc[:,3:].apply(sum,0).tolist(), index = [col for col in df.columns]), ignore_index=True )
     shared_df = df.copy()
     shared_df.iloc[:,3:] = df.iloc[:,3:].applymap(lambda x: 1 if x<=30 else 0)
     shared_df.iloc[:,3:] = shared_df.apply(lambda x: x.iloc[3:]*x['found_count'],1)       
     out_df=df.append(pd.Series([np.nan, np.nan, 'Shared with cutoff 30'] + shared_df.iloc[:,3:].apply(sum,0).tolist(), index = [col for col in df.columns]), ignore_index=True )