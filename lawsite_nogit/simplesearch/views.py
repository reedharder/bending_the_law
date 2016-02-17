from django.shortcuts import render
import pickle
import json 
from django.http import HttpResponse
from django.views.decorators.csrf import ensure_csrf_cookie



# Create your views here.
@ensure_csrf_cookie
def search(request):       
  
    if request.method == 'POST':
        if request.POST['posttype']=='query':
            #case title to index dict
            titleskey =pickle.load(open('titleskey_2.p','rb'))
            ##titleskey = pickle.loads(bucket.get_key("titleskey.p").get_contents_as_string())
            #index to metadata dict 
            titles =pickle.load(open('titles.p','rb'))
            ##titles = pickle.loads(bucket.get_key("titles.p").get_contents_as_string())
            #title to metadata dict
            metadata =pickle.load(open('metadata_2.p','rb'))
            ##metadata = pickle.loads(bucket.get_key("metadata.p").get_contents_as_string())
            tcit = request.POST['tcit']
            tcitby = request.POST['tcitby']
            tsim = request.POST['tsim']
            if tcit == tcitby and tcitby == tsim:
                #index to top similar indices
                case_lookup =pickle.load(open('case_lookup_111.p','rb'))
            else:
                case_lookup =pickle.load(open('case_lookup_%s%s%s.p' % (tcit,tcitby,tsim),'rb'))
            query = request.POST['querycase']
            query = query.split(':')[0]
            caselist_ids = case_lookup[titleskey[query]]
            caselist = [titles[caseid]['usid_title'] for caseid in caselist_ids][1:]
            caselist_usids = [case.split(':')[0].replace(' US ','/') for case in caselist]                      
            meta = metadata[query]
            data = {'caselist':caselist,'caselist_usids':caselist_usids }
            data.update(meta)
            return HttpResponse(json.dumps(data), content_type='application/json')
        elif request.POST['posttype']=='mds2d':
            import itertools
            import numpy as np
            from sklearn.manifold import MDS                         
            tcit = request.POST['tcit']
            tcitby = request.POST['tcitby']
            tsim = request.POST['tsim']
            if tcit == tcitby and tcitby == tsim:
                tcit = 1
                tcitby = 1
                tsim = 1
      #load data for test MDS
            combo = [tcit,tcitby,tsim]          
            query = request.POST['querycase'] 
            query = query.split(':')[0]
            titleskey=pickle.load(open('titleskey_2.p','rb'))
            titles=pickle.load(open('titles.p','rb'))        
            #title to metadata dict
            metadata =pickle.load(open('metadata_2.p','rb'))
           
            if tcit == tcitby and tcitby == tsim:
                #index to top similar indices
                case_lookup =pickle.load(open('case_lookup_111.p','rb'))
            else:
                case_lookup =pickle.load(open('case_lookup_%s%s%s.p' % (tcit,tcitby,tsim),'rb'))
           
            caselist_ids = case_lookup[titleskey[query]]
            caselist = [titles[caseid]['usid_title'] for caseid in caselist_ids]
            meta = metadata[query]
            data = {'caselist':caselist}
            data.update(meta)
            #get possible relevant distances from database            
            npy_dists  = np.load('row_col_dist_int_%s%s%s.npy' % (combo[0],combo[1],combo[2]))            
            
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
                nodes.append({'id':case_data['tran_mat_index'], 'label': case_data['parties'], 'title':'Case ID: ' + case_data['usid_index'].replace('_','') + '<br>' + 'Year: ' + str(case_data['year']),'value': value, 'group':group,'x':pos[i,0]/30, 'y':pos[i,1]/30} )
            #get edges in json dict
            #trans_cites = np.load('cite_network.npy')
            trans_cites=np.load('cite_network.npy')
                        
            reduced_trans_cites  = trans_cites[np.in1d(trans_cites[:,0], caselist_ids) & np.in1d(trans_cites[:,1], caselist_ids) ]       
            edges = [{'from': cite[0],'to':cite[1]} for cite in reduced_trans_cites.tolist()]
            return HttpResponse(json.dumps({'nodes':nodes,'edges':edges}), content_type='application/json')
            
    else:
        titleskey =pickle.load(open('titleskey.p','rb'))
       
        return render(request, 'simplesearch/search.html', {'cases':sorted(titleskey.keys())} )
