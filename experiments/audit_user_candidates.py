"""TRAIN-only candidate/ranking bottleneck audit and item-balanced UU graph."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from build_semantic_user_graphs import normalize, anchor_history, cosine_rows, sha

WINDOWS=(1,4,16,64,256)
KS=(5,10,20,40,80)


def item_balanced(u, ranked, r, item_ties, k):
    """Round-robin distinct shared-item buckets. Rank inside each by original score.

    Every selected neighbor is unique. Bucket order uses random item keys, never ID.
    Multiple-anchor neighbors are credited to the first bucket selecting them.
    """
    anchors=r.indices[r.indptr[u]:r.indptr[u+1]]
    buckets={int(i):[] for i in anchors}
    for v in ranked:
        for i in np.intersect1d(anchors,r.indices[r.indptr[v]:r.indptr[v+1]],assume_unique=True):
            buckets[int(i)].append(int(v))
    order=sorted((i for i in buckets if buckets[i]),key=lambda i:item_ties[i])
    cursor={i:0 for i in order};selected=[];seen=set()
    while len(selected)<k:
        progress=False
        for i in order:
            bucket=buckets[i]
            while cursor[i]<len(bucket) and bucket[cursor[i]] in seen:cursor[i]+=1
            if cursor[i]<len(bucket):
                v=bucket[cursor[i]];cursor[i]+=1
                selected.append(v);seen.add(v);progress=True
                if len(selected)==k:break
        if not progress:break
    return np.array(selected,dtype=np.int64)


def concentration(u,chosen,r,item_degree):
    anchors=r.indices[r.indptr[u]:r.indptr[u+1]]
    possible=int(np.count_nonzero(item_degree[anchors]>1))
    if not len(chosen):return None
    credit=np.zeros(len(anchors))
    for v in chosen:
        common=np.intersect1d(anchors,r.indices[r.indptr[v]:r.indptr[v+1]],assume_unique=True)
        if len(common):credit[np.searchsorted(anchors,common)]+=1/len(common)
    share=credit/len(chosen)
    return dict(max_anchor_share=float(share.max()),anchor_hhi=float((share**2).sum()),
                eligible_anchor_coverage=float(np.count_nonzero(credit)/possible) if possible else 0.)


def rank_info(sorted_ids, sorted_scores, target):
    locations=np.flatnonzero(sorted_ids==target)
    if not len(locations):return None
    location=int(locations[0]);score=sorted_scores[location]
    # Exact computed-score ties, no tolerance-induced merging.
    low=int(np.count_nonzero(sorted_scores>score))+1
    high=int(np.count_nonzero(sorted_scores>=score))
    return dict(rank=location+1,low=low,high=high,ties=high-low+1)


def summarize_pairs(records):
    total=len(records);present=[x for x in records if x is not None]
    count=len(present)
    result=dict(eligible_pairs=total,candidate_pairs=count,
                candidate_coverage=count/total if total else None,
                candidate_missing_fraction=(total-count)/total if total else None)
    if not count:return result
    result['conditional_rank_quantiles']=dict(zip(('p10','median','p90','p99'),np.quantile([x['rank'] for x in present],[.1,.5,.9,.99]).tolist()))
    result['conditional_mean_tie_count']=float(np.mean([x['ties'] for x in present]))
    result['topk']={}
    for k in KS:
        chosen=sum(x['rank']<=k for x in present)
        result['topk'][str(k)]=dict(
            fraction_of_all_pairs_selected=chosen/total,
            fraction_of_candidates_selected=chosen/count,
            fraction_of_candidates_tie_cutoff=float(np.mean([x['low']<=k<x['high'] for x in present])),
            expected_candidate_recall_random_ties=float(np.mean([np.clip((k-x['low']+1)/x['ties'],0,1) for x in present])),
            optimistic_candidate_recall=float(np.mean([x['low']<=k for x in present])),
            pessimistic_candidate_recall=float(np.mean([x['high']<=k for x in present])))
    return result


def graph_summary(graph, degree, pair_counts, eligible_counts, concentrations):
    u,v=graph.nonzero()
    result=dict(edges=int(graph.nnz),users_with_neighbors=int(np.count_nonzero(np.diff(graph.indptr))),
                edge_fraction_within_window={str(w):float(np.mean(np.abs(u-v)<=w)) if len(u) else None for w in WINDOWS},
                exact_offset_pair_recall={str(w):pair_counts[w]/eligible_counts[w] if eligible_counts[w] else None for w in WINDOWS})
    result['anchor_concentration']={}
    for key in ('max_anchor_share','anchor_hhi','eligible_anchor_coverage'):
        vals=[x[key] for x in concentrations if x is not None]
        result['anchor_concentration'][key]=dict(mean=float(np.mean(vals)),median=float(np.median(vals)),p90=float(np.quantile(vals,.9))) if vals else None
    result['note']='Anchor credit is split equally among all shared items for each selected neighbor; not unique causal attribution.'
    return result


def audit(r,features,k=20,block=128,seed=2026,out=None):
    r=r.astype(np.float32).tocsr();r.sum_duplicates();r.eliminate_zeros();r.data[:]=1;r.sort_indices()
    degree=np.diff(r.indptr);item_degree=np.asarray(r.sum(0)).ravel()
    rng=np.random.default_rng(seed);ties=rng.random(r.shape[0]);item_ties=rng.random(r.shape[1])
    results={};graphs={}
    matrices={'shared_cosine':r,'semantic_anchor':anchor_history(r,features)}
    for name,x in matrices.items():
        print('Auditing '+name,flush=True)
        z=cosine_rows(x)
        records={w:[] for w in WINDOWS};candidate_counts=[]
        names=[name]+(['item_balanced'] if name=='shared_cosine' else [])
        buffers={n:dict(row=[],col=[],weight=[],concentration=[],hits={w:0 for w in WINDOWS}) for n in names}
        eligible={w:0 for w in WINDOWS}
        for start in range(0,r.shape[0],block):
            scores=(z[start:start+block]@z.T).tocsr()
            for local in range(scores.shape[0]):
                u=start+local
                if not degree[u]:continue
                v=scores.indices[scores.indptr[local]:scores.indptr[local+1]]
                w=scores.data[scores.indptr[local]:scores.indptr[local+1]]
                keep=(v!=u)&(w>0);v,w=v[keep],w[keep]
                order=np.lexsort((ties[v],-w));v,w=v[order],w[order]
                candidate_counts.append(len(v))
                choices={name:v[:k]}
                if name=='shared_cosine':choices['item_balanced']=item_balanced(u,v,r,item_ties,k)
                score_by_user=dict(zip(v.tolist(),w.tolist()))
                for n,selected in choices.items():
                    b=buffers[n];b['row'].extend([u]*len(selected));b['col'].extend(selected.tolist())
                    b['weight'].extend(score_by_user[int(j)] for j in selected)
                    b['concentration'].append(concentration(u,selected,r,item_degree))
                for offset in WINDOWS:
                    for target in (u-offset,u+offset):
                        if 0<=target<r.shape[0] and degree[target]:
                            eligible[offset]+=1
                            records[offset].append(rank_info(v,w,target))
                            for n,selected in choices.items():buffers[n]['hits'][offset]+=int(target in selected)
            if start%2048==0:print(f'{name}: {min(start+block,r.shape[0])}/{r.shape[0]} users',flush=True)
        ranking={str(w):summarize_pairs(records[w]) for w in WINDOWS}
        for n,b in buffers.items():
            graph=sp.csr_matrix((b['weight'],(b['row'],b['col'])),shape=(r.shape[0],)*2,dtype=np.float32)
            graphs[n]=graph
            results[n]=graph_summary(graph,degree,b['hits'],eligible,b['concentration'])
            if n==name:
                results[n]['ranking']=ranking
                results[n]['candidate_count_quantiles']=np.quantile(candidate_counts,[0,.5,.9,1]).tolist()
            if out is not None:
                sp.save_npz(out/(n+'.npz'),graph)
                row_sum=np.asarray(graph.sum(1)).ravel()
                sp.save_npz(out/(n+'_rownorm.npz'),sp.diags(1/np.maximum(row_sum,1e-12))@graph)
    return dict(graphs=results,note='TRAIN-only. Exact-offset pairs are all valid directed pairs, not the prior sampled pairs. '
                'Rank statistics condition on score-positive candidates. Top-k optimistic/pessimistic bounds vary tie resolution, not a proposed ID-aware graph. '
                'Item-balanced selection uses equal-turn shared-item buckets and original cosine scores, with random entity tie keys. '
                'Graphs use no ID distance; exact ID relabeling requires co-permuting tie keys. No recommendation accuracy is measured.'),graphs


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',required=True);p.add_argument('--dataset',default='baby')
    p.add_argument('--output',default='night_runs/baby-user-rank-1')
    p.add_argument('--k',type=int,default=20);p.add_argument('--block',type=int,default=128);p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if min(args.k,args.block)<1:p.error('k and block must be positive')
    folder=Path(args.data_path)/args.dataset;out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    paths=[folder/(args.dataset+'.inter'),folder/'image_feat.npy',folder/'text_feat.npy']
    df=pd.read_csv(paths[0],sep='\t',usecols=['userID','itemID','x_label'])
    edges=df[df.x_label==0][['userID','itemID']].drop_duplicates()
    if edges.empty or (edges.to_numpy()<0).any():raise ValueError('Invalid training edges')
    features=[np.load(f,allow_pickle=False).astype(np.float32) for f in paths[1:]]
    if any(x.ndim!=2 or not np.isfinite(x).all() for x in features):raise ValueError('Invalid feature arrays')
    if len(features[0])!=len(features[1]) or edges.itemID.max()>=len(features[0]):raise ValueError('Feature indexing mismatch')
    f=normalize(np.concatenate([normalize(x) for x in features],axis=1))
    r=sp.csr_matrix((np.ones(len(edges)),(edges.userID,edges.itemID)),shape=(int(edges.userID.max())+1,len(f)))
    report,_=audit(r,f,args.k,args.block,args.seed,out)
    report.update(arguments=vars(args),training_edges=int(r.nnz),input_sha256={str(f):sha(f) for f in paths})
    (out/'report.json').write_text(json.dumps(report,indent=2))
    print('Finished: '+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
