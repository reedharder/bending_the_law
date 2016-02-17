# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:36:35 2015

@author: d29905p
"""

'''
scripts for analyis of page dist data

'''
import numpy as np
from  matplotlib.pyplot import *
import time
import pickle
from bendlinglaw import *

[Our_Docs, Our_theta, phi, Our_Cites,  Our_Cites_Sym] = load_data()
Our_Docs['tran_mat_index'] = np.arange(0,len(Our_Docs),1)
[W_cited, T_cited, W_cited_by, T_cited_by] = create_cite_transition(Our_Docs,Our_Cites)
print(str(time.time()-t0) + " time to laod data and create cite matricies" )
Create_Sim = True
#Create_Sim = False
M_T = 10
M_C = 10
if Create_Sim:
    t0=time.time()
		
    T_sim = create_sim_matrix(M_T,M_C,Our_Docs,Our_theta)
    print(str(time.time()-t0) + " time for, M_T= " + str(M_T) + " , and M_C = " + str(M_C))
    pickle.dump(T_sim,open(data_dir+ 'T_sim2' + '_' + str(M_T) + '_'  + str(M_C)  + '.p', 'wb'))
else: 
    T_sim=pickle.load(open(data_dir+ 'T_sim2' + '_' + str(M_T) + '_'  + str(M_C)  + '.p', 'rb'))

# might need some sort or restart, but maybe not since we are palying russioan roullette
r = 5/6.0
r_name = '5Over6'

r = 1/3.0
r_name = '1Over3'

r = 1/2.0
r_name = '1Over2'
L_p = 2

	
#load page dist data
T_all = pickle.load(open(data_dir+ 'T_all' + '_' + str(M_T) + '_'  + str(M_C) + '.p', 'rb'))
# could get 
# R_all = pickle.load(open(data_dir+ 'R_all' + '_' + str(M_T) + '_'  + str(M_C) +'_' + str(r_name) + '.p', 'rb'))
Page_Dist = pickle.load(open(data_dir+ 'Page_Dist_1' + '_' + str(M_T) + '_'  + str(M_C) + '_'  + str(L_p)+'_' + str(r_name) + '.p', 'rb'))



#run analysis
fig_num0 = 1
hist_Page_Dist(fig_num0, Page_Dist)
	# gathering exmaples to va;idate page dist 
V = Page_Dist[Page_Dist>0]
Per_Small = np.percentile(V,35)
Per_Large = np.percentile(V,80)
Per_Large0 = np.percentile(V,65)
Per_conn = np.percentile(Our_Docs.count_connections,30) 
F = get_example(Our_Docs, Page_Dist, Per_Small, Per_Large)
F = F[(F.total_close1>5) & (F.total_close2>5) & (F.count_connections2>Per_conn) & (F.count_connections2>Per_conn)]
Info  = get_exmaple_regions(Page_Dist, Our_Docs, F,Per_Small, Per_Large0)

Create_Puddles = False
Create_Puddles = True
# original
run_id = ''
run_id = '(new1)'
if Create_Puddles:
    Drain_Pud_Info = {}     
    Drain_Pud_Info['M_T'] = M_T
    Drain_Pud_Info['M_C'] = M_C
    Drain_Pud_Info['L_p'] = L_p
    Drain_Pud_Info['r_name'] = '1Over2'
    Drain_Pud_Info['r'] = r
    Drain_Pud_Info['impact_key_pertile_list'] = np.arange(5,105,5)
    Drain_Pud_Info['bending_key_pertile'] = 30 
    Drain_Pud_Info['date0_list'] = np.arange(1972,1993,1)
    Drain_Pud_Info['Window']= 5
    # takeslike 20 miniutes to run output in memepry
    Drain_Pud_Info = puddles_drainage_over_time(Drain_Pud_Info, T_all, Page_Dist, Our_Docs )
    pickle.dump(Drain_Pud_Info, open(data_dir+ 'Drain_Pud_Info' + '_' + str(M_T) + '_'  + str(M_C)+ '_'  + str(L_p) +'_' + str(r_name) + run_id + '.p', 'wb'))
else:
    Drain_Pud_Info = pickle.load(open(data_dir+ 'Drain_Pud_Info' + '_' + str(M_T) + '_'  + str(M_C) + '_'  + str(L_p)+'_' + str(r_name) + run_id + '.p', 'rb'))
		# return [date0_list, impact_key_pertile_list, PIm2GIm1, PIm2GIm1P, PIm2GIm1D, bending, k_restricted]
fig_num0 = fig_num0 + 1
date0 = 1990
Pick_Out=3
graphs_from_papers(fig_num0, date0, Pick_Out, Drain_Pud_Info )
# There are two major goals
# 1. FInd High and low impact regions 
# 2. High and low drainage regions in amoung high impact 
impact_key_pertile = Drain_Pud_Info['impact_key_pertile_list'][Pick_Out]
Bending = bending_on_date(date0, Drain_Pud_Info, Our_Docs, T_all)

	
Impact_Data = puddling_on_date_fixed_impact(impact_key_pertile, Drain_Pud_Info, Page_Dist, Bending)
 # find high and low impact as function of date compare good region 

fig_num0 = 20
fig_num0 = fig_num0 + 1
plot_date_differecne_high_low_impact(fig_num0, Impact_Data, Our_Docs)

	# find year controlled 
	# high impact low 
	# low impact in same rangeg 
	# find l

beg_date = 1980
end_date = 1985
TF_date = (Our_Docs.year>=beg_date)  &  (Our_Docs.year<=end_date)
OD_Low = Our_Docs[TF_date & (Our_Docs.tran_mat_index.isin(Impact_Data['Impact1_Low']))]
OD_High = Our_Docs[TF_date & (Our_Docs.tran_mat_index.isin(Impact_Data['Impact1']))]
[len(OD_Low),len(OD_High)]

beg_date = 1960
end_date = 1965
TF_date = (Our_Docs.year>=beg_date)  &  (Our_Docs.year<=end_date)
OD_Low = Our_Docs[TF_date & (Our_Docs.tran_mat_index.isin(Impact_Data['Impact1_Low']))]
OD_High = Our_Docs[TF_date & (Our_Docs.tran_mat_index.isin(Impact_Data['Impact1']))]
[len(OD_Low),len(OD_High)]


# run Bending_The_Law2.py
n_then = [1960,1965] 
n_now = [date0+1,date0+ Window] 
High_Or_Low='Low'
Keep=100    
Exmaple_Date = collect_interesting_date_ex(High_Or_Low, n_now, n_then, Our_Docs, Page_Dist, Keep)
Exmaple_Date = join_and_rename('_then', Exmaple_Date, Our_Docs )
Exmaple_Date = join_and_rename('_now_0', Exmaple_Date, Our_Docs )
Exmaple_Date = join_and_rename('_now_1', Exmaple_Date, Our_Docs )
Exmaple_Date = join_and_rename('_now_2', Exmaple_Date, Our_Docs )
N=0
Exmaple_Date[N:(N+10):1][['parties_then','year_then', 'parties_now_0','year_now_0','parties_now_1','year_now_1','parties_now_2','year_now_2']]
Exmaple_Date[Exmaple_Date.index.isin([0,2,8])][['parties_then','year_then', 'parties_now_0','year_now_0','parties_now_1','year_now_1','parties_now_2','year_now_2']] 


n_then = [1960,1965] 
n_now = [date0+1,date0+ Window] 
High_Or_Low='High'
Keep=100
Exmaple_Date = collect_interesting_date_ex(High_Or_Low, n_now, n_then, Our_Docs, Page_Dist, Keep)
Exmaple_Date = join_and_rename('_then', Exmaple_Date, Our_Docs )
Exmaple_Date = join_and_rename('_now_0', Exmaple_Date, Our_Docs )
Exmaple_Date = join_and_rename('_now_1', Exmaple_Date, Our_Docs )
Exmaple_Date = join_and_rename('_now_2', Exmaple_Date, Our_Docs )
N=0
Exmaple_Date[N:(N+10):1][['parties_then','year_then', 'parties_now_0','year_now_0','parties_now_1','year_now_1','parties_now_2','year_now_2']]

	# last thing is High and low drainage regios 
	# pobbaity:High impact Driaingage 
	# Low Impact 


close(3)
figure(3)
A = Page_Dist[Tran_then,:]
A = A[:,Tran_now]
M = A.min(1)
hist(M,20)



[100*len(set(Impact_Data['Impact2']).intersection(set(Impact_Data['Impact1'])))/(1.0*len(Impact_Data['Impact1'])),		
 100*len(set(Impact_Data['Impact2']).intersection(set(Impact_Data['I1D'])))/(1.0*len(Impact_Data['I1D'])),
100*len(set(Impact_Data['Impact2']).intersection(set(Impact_Data['I1P'])))/(1.0*len(Impact_Data['I1P']))]

[100*len(set(Impact_Data['Impact2_Low']).intersection(set(Impact_Data['Impact1'])))/(1.0*len(Impact_Data['Impact1'])),		
 100*len(set(Impact_Data['Impact2_Low']).intersection(set(Impact_Data['I1D'])))/(1.0*len(Impact_Data['I1D'])),
100*len(set(Impact_Data['Impact2_Low']).intersection(set(Impact_Data['I1P'])))/(1.0*len(Impact_Data['I1P']))]