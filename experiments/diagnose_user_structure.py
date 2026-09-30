"""TRAIN-only structure diagnostics and ID-independent user graph construction."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def unit_rows(matrix):
    norms=np.sqrt(np.asarray(matrix.multiply(matrix).sum(1)).ravel())
    return sp.diags(1/np.maximum(norms,1e-12)) @ matrix


def pair_stats(r, a, b, degree, item_degree, signatures):
    shared=np.asarray(r[a].multiply(r[b]).sum(1)).ravel()
    rare=np.asarray(r[a].multiply(r[b]).multiply(1/np.maximum(item_degree,1)).sum(1)).ravel()
    return dict(pairs=len(a),any_shared=float(np.mean(shared>0)), shared_items=float(shared.mean()),
                jaccard=float(np.mean(shared/np.maximum(degree[a]+degree[b]-shared,1))),
                cosine=float(np.mean(shared/np.sqrt(degree[a]*degree[b]))),
                resource_allocation=float(rare.mean()),
                degree_abs_difference=float(np.abs(degree[a]-degree[b]).mean()),
                neighbor_popularity_profile_cosine=float(np.mean(np.sum(signatures[a]*signatures[b],axis=1))))


def build_graph(r, metric, k, rng, block=128):
    """No ID distances: scores depend solely on training neighbor sets.

    Tie-breaking uses independent random entity keys, not numeric ID order.
    """
    degree=np.asarray(r.sum(1)).ravel()
    item_degree=np.asarray(r.sum(0)).ravel()
    if metric=='cosine':
        x=unit_rows(r)
    elif metric=='resource_cosine':
        x=unit_rows(r.multiply(1/np.sqrt(np.maximum(item_degree,1))).tocsr())
    else:
        x=r
    tie=rng.random(r.shape[0])
    rows=[];cols=[];weights=[]
    for start in range(0,r.shape[0],block):
        scores=(x[start:start+block]@x.T).tocsr()
        for local in range(scores.shape[0]):
            u=start+local
            v=scores.indices[scores.indptr[local]:scores.indptr[local+1]]
            s=scores.data[scores.indptr[local]:scores.indptr[local+1]].copy()
            keep=(v!=u)&(s>0);v,s=v[keep],s[keep]
            if metric=='jaccard':s=s/(degree[u]+degree[v]-s)
            if len(v)>k:
                threshold=np.partition(s,-k)[-k];keep=s>=threshold;v,s=v[keep],s[keep]
            order=np.lexsort((tie[v],-s))[:k];v,s=v[order],s[order]
            rows.extend([u]*len(v));cols.extend(v.tolist());weights.extend(s.tolist())
    return sp.csr_matrix((weights,(rows,cols)),shape=(r.shape[0],r.shape[0]))


def diagnose(r, offsets, pairs, k, seed, permutations, block, save=None):
    r=r.astype(np.float32).tocsr();r.sum_duplicates();r.eliminate_zeros();r.data[:]=1
    degree=np.asarray(r.sum(1)).ravel();item_degree=np.asarray(r.sum(0)).ravel()
    users=np.flatnonzero(degree)
    if len(users)<3:raise ValueError('Need at least three active users')
    rng=np.random.default_rng(seed)
    bins=np.floor(np.log2(np.maximum(item_degree,1))).astype(int)
    item_to_bin=sp.csr_matrix((np.ones(len(bins)),(np.arange(len(bins)),bins)),shape=(len(bins),bins.max()+1))
    signature=(r@item_to_bin).toarray()
    signature/=np.maximum(np.linalg.norm(signature,axis=1,keepdims=True),1e-12)
    degree_groups={d:users[degree[users]==d] for d in np.unique(degree[users])}
    pair_results={}
    for offset in offsets:
        available=users[users+offset<r.shape[0]]
        available=available[degree[available+offset]>0]
        if not len(available):continue
        a=rng.choice(available,min(pairs,len(available)),replace=False);b=a+offset
        matched=[];fallback=0
        for u,v in zip(a,b):
            candidates=degree_groups[degree[v]]
            candidates=candidates[(candidates!=u)&(candidates!=v)]
            if not len(candidates):
                candidates=users[(users!=u)&(users!=v)]
                gap=np.abs(degree[candidates]-degree[v]);candidates=candidates[gap==gap.min()];fallback+=1
            matched.append(rng.choice(candidates))
        random=rng.choice(users,len(a))
        while np.any(random==a):
            bad=random==a;random[bad]=rng.choice(users,bad.sum())
        pair_results[str(offset)]=dict(near=pair_stats(r,a,b,degree,item_degree,signature),
             random=pair_stats(r,a,random,degree,item_degree,signature),
             degree_matched=pair_stats(r,a,np.array(matched),degree,item_degree,signature),
             nearest_degree_fallback_pairs=fallback)
    graph_results={}
    for metric in ('cosine','jaccard','resource_cosine'):
        print('Building '+metric,flush=True)
        graph=build_graph(r,metric,k,np.random.default_rng(seed),block)
        u,v=graph.nonzero();dist=np.abs(u-v)
        stats=dict(edges=graph.nnz,users_with_neighbors=int(np.count_nonzero(np.diff(graph.indptr))))
        for window in (1,4,16,64,256):
            observed=float(np.mean(dist<=window)) if len(dist) else None
            null=[]
            for _ in range(permutations):
                labels=np.arange(r.shape[0]);labels[users]=rng.permutation(users)
                null.append(float(np.mean(np.abs(labels[u]-labels[v])<=window)) if len(dist) else 0.)
            null_mean=float(np.mean(null))
            stats[str(window)]=dict(edge_fraction_within_window=observed, shuffled_label_mean=null_mean,
                shuffled_label_interval_95=np.quantile(null,[.025,.975]).tolist(),
                enrichment=observed/null_mean if null_mean and observed is not None else None)
        graph_results[metric]=stats
        if save is not None:sp.save_npz(save/(metric+'.npz'),graph)
    return dict(users=int(len(users)),training_pairs=int(r.nnz),seed=seed,k=k,pairs=pair_results,graphs=graph_results,
      note='TRAIN-only; ID proximity is used only in evaluation, never in edge scoring or selection. '
           'Graphs are directed top-k with raw positive scores, no self edges; no-overlap users may be isolated. '
           'Pair statistics are descriptive and pairs are not independent. Shuffled intervals are null reference intervals, not confidence intervals. '
           'Common-neighbor similarity does not establish isomorphism or causal user cohorts. No ranking accuracy is measured.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',required=True);p.add_argument('--dataset',default='baby')
    p.add_argument('--output',default='night_runs/baby-structure-1')
    p.add_argument('--pairs',type=int,default=10000);p.add_argument('--k',type=int,default=20)
    p.add_argument('--seed',type=int,default=2026);p.add_argument('--permutations',type=int,default=100)
    p.add_argument('--block',type=int,default=128);p.add_argument('--save-graphs',action='store_true')
    args=p.parse_args()
    if min(args.pairs,args.k,args.permutations,args.block)<1:p.error('Numeric budgets must be positive')
    path=Path(args.data_path)/args.dataset/(args.dataset+'.inter')
    df=pd.read_csv(path,sep='\t',usecols=['userID','itemID','x_label'])
    train=df[df.x_label==0][['userID','itemID']].drop_duplicates()
    if train.empty:raise ValueError('Empty training split')
    r=sp.csr_matrix((np.ones(len(train)),(train.userID,train.itemID)),shape=(int(train.userID.max())+1,int(train.itemID.max())+1))
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'report.json').exists():raise ValueError('Choose a new output directory')
    result=diagnose(r,[1,2,4,8,16,32,64,128,256],args.pairs,args.k,args.seed,args.permutations,args.block,out if args.save_graphs else None)
    result['dataset']=args.dataset
    (out/'report.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
