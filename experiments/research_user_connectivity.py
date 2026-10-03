"""TRAIN-only short-path study of complete item groups and semantic bridges.

Graphs never use ID distance. The same sampled anchors/pairs evaluate all graphs.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from build_semantic_user_graphs import normalize, item_transition, sha

OFFSETS=(1,4,16,64,256)


def undirected_item_graph(features,k,block,ties):
    s=item_transition(features,k,block,ties).tocsr()
    s.setdiag(0);s.eliminate_zeros()
    s=s.astype(bool)
    return s.maximum(s.T).tocsr()


def rewire(graph,seed,swaps_per_edge=5):
    """Undirected double-edge swaps preserve each item's exact II degree.

    A finite-swap control, not a claim of uniform sampling over all such graphs.
    """
    rng=np.random.default_rng(seed)
    a,b=sp.triu(graph,k=1).nonzero()
    edges=list(zip(a.tolist(),b.tolist()));edge_set=set(edges)
    target=len(edges)*swaps_per_edge;success=0;attempts=0
    while success<target and attempts<max(1,target*20) and len(edges)>1:
        attempts+=1
        x,y=rng.integers(len(edges),size=2)
        if x==y:continue
        u,v=edges[x];w,z=edges[y]
        if rng.integers(2):w,z=z,w
        if len({u,v,w,z})!=4:continue
        e1=tuple(sorted((u,z)));e2=tuple(sorted((w,v)))
        if e1 in edge_set or e2 in edge_set:continue
        edge_set.remove(edges[x]);edge_set.remove(edges[y]);edge_set.add(e1);edge_set.add(e2)
        edges[x]=e1;edges[y]=e2;success+=1
    rows=[];cols=[]
    for u,v in edges:rows.extend((u,v));cols.extend((v,u))
    out=sp.csr_matrix((np.ones(len(rows),dtype=bool),(rows,cols)),shape=graph.shape)
    if not np.array_equal(np.diff(out.indptr),np.diff(graph.indptr)):raise AssertionError('Degree mismatch')
    return out,dict(successful_swaps=success,target_swaps=target,attempts=attempts,
                    original_edge_fraction_remaining=len(edge_set & set(zip(a.tolist(),b.tolist())))/len(edges) if edges else 0.)


def paired_controls(degree,anchors,seed):
    rng=np.random.default_rng(seed);active=np.flatnonzero(degree)
    buckets={d:active[degree[active]==d] for d in np.unique(degree[active])}
    pairs={}
    for offset in OFFSETS:
        rows=[];near=[];random=[];matched=[];fallback=0
        for row,u in enumerate(anchors):
            v=u+offset
            if v>=len(degree) or not degree[v]:continue
            pool=active[(active!=u)&(active!=v)]
            exact=buckets[degree[v]];exact=exact[(exact!=u)&(exact!=v)]
            if not len(exact):
                gap=np.abs(degree[pool]-degree[v]);exact=pool[gap==gap.min()];fallback+=1
            rows.append(row);near.append(v);random.append(rng.choice(pool));matched.append(rng.choice(exact))
        pairs[str(offset)]=dict(rows=np.array(rows,dtype=int),near=np.array(near,dtype=int),
             random=np.array(random,dtype=int),degree_matched=np.array(matched,dtype=int),degree_match_fallback=fallback)
    return pairs


def paths_from_anchors(adjacency,anchors,n_users,block=32):
    """Exact Boolean walk reachability <=2/3/4 edges; no dense all-pairs matrix."""
    for start in range(0,len(anchors),block):
        users=anchors[start:start+block]
        frontier=sp.csr_matrix((np.ones(len(users),dtype=bool),(np.arange(len(users)),users)),shape=(len(users),adjacency.shape[0]))
        reached=frontier.copy()
        for hop in range(1,5):
            frontier=(frontier@adjacency).tocsr().astype(bool)
            reached=reached.maximum(frontier).tocsr()
            if hop>=2:
                uu=reached[:,:n_users].tolil()
                uu[np.arange(len(users)),users]=False
                uu=uu.tocsr();uu.eliminate_zeros()
                yield start,hop,uu


def evaluate(r,item_graph,anchors,pairs,block):
    n_users,n_items=r.shape
    adjacency=sp.bmat([[sp.csr_matrix((n_users,n_users),dtype=bool),r.astype(bool)],
                      [r.T.astype(bool),item_graph]],format='csr',dtype=bool)
    _,labels=connected_components(adjacency,directed=False)
    active=np.flatnonzero(np.diff(r.indptr))
    counts=np.bincount(labels[active]);largest=int(counts.max()) if len(counts) else 0
    hits={str(h):{offset:{kind:0 for kind in ('near','random','degree_matched')} for offset in pairs} for h in (2,3,4)}
    degrees={str(h):[] for h in (2,3,4)}
    for start,hop,reach in paths_from_anchors(adjacency,anchors,n_users,block):
        degrees[str(hop)].extend(np.diff(reach.indptr).tolist())
        for offset,pair in pairs.items():
            mask=(pair['rows']>=start)&(pair['rows']<start+reach.shape[0])
            local=pair['rows'][mask]-start
            for kind in ('near','random','degree_matched'):
                if len(local):hits[str(hop)][offset][kind]+=int(np.asarray(reach[local,pair[kind][mask]]).sum())
        if hop==4 and start%(block*8)==0:print(f'  paths: {min(start+block,len(anchors))}/{len(anchors)} anchors',flush=True)
    result=dict(item_edges_undirected=int(item_graph.nnz//2),largest_component_user_fraction=largest/len(active),hops={})
    for hop in (2,3,4):
        per_pair={}
        for offset,pair in pairs.items():
            total=len(pair['rows'])
            rates={kind:hits[str(hop)][offset][kind]/total if total else None for kind in ('near','random','degree_matched')}
            rates.update(pairs=total,degree_match_fallback=pair['degree_match_fallback'])
            if total:
                rates['near_minus_matched']=rates['near']-rates['degree_matched']
                rates['near_over_matched']=rates['near']/rates['degree_matched'] if rates['degree_matched'] else None
            same={kind:float(np.mean(labels[anchors[pair['rows']]]==labels[pair[kind]])) if total else None for kind in ('near','random','degree_matched')}
            rates['same_component']=same
            per_pair[offset]=rates
        deg=np.array(degrees[str(hop)])
        result['hops'][str(hop)]=dict(mean_reachable_other_users=float(deg.mean()),
            mean_fraction_active_users_reachable=float(deg.mean()/max(1,len(active)-1)),
            reachable_count_quantiles=np.quantile(deg,[.1,.5,.9]).tolist(),pairs=per_pair)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path',required=True);parser.add_argument('--dataset',default='baby')
    parser.add_argument('--output',default='night_runs/baby-connectivity-1')
    parser.add_argument('--anchors',type=int,default=2048);parser.add_argument('--block',type=int,default=32)
    parser.add_argument('--item-k',type=int,nargs='+',default=[5,10]);parser.add_argument('--null-repeats',type=int,default=2)
    parser.add_argument('--seed',type=int,default=2026)
    args=parser.parse_args()
    if min(args.anchors,args.block,args.null_repeats,*args.item_k)<1:parser.error('Counts must be positive')
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    start=time.time()
    try:
        (output/'status.json').write_text(json.dumps(dict(state='running')))
        folder=Path(args.data_path)/args.dataset
        paths=[folder/(args.dataset+'.inter'),folder/'image_feat.npy',folder/'text_feat.npy']
        df=pd.read_csv(paths[0],sep='\t',usecols=['userID','itemID','x_label'])
        train=df[df.x_label==0][['userID','itemID']].drop_duplicates()
        if train.empty or (train.to_numpy()<0).any():raise ValueError('Invalid train interactions')
        feats=[np.load(p,allow_pickle=False).astype(np.float32) for p in paths[1:]]
        if any(x.ndim!=2 or not np.isfinite(x).all() for x in feats):raise ValueError('Invalid features')
        if len(feats[0])!=len(feats[1]) or train.itemID.max()>=len(feats[0]):raise ValueError('Feature indexing mismatch')
        f=normalize(np.concatenate([normalize(x) for x in feats],axis=1))
        r=sp.csr_matrix((np.ones(len(train),dtype=bool),(train.userID,train.itemID)),shape=(int(train.userID.max())+1,len(f)))
        active=np.flatnonzero(np.diff(r.indptr))
        if len(active)<3:raise ValueError('Need three active users')
        rng=np.random.default_rng(args.seed)
        anchors=np.sort(rng.choice(active,min(args.anchors,len(active)),replace=False))
        pairs=paired_controls(np.diff(r.indptr),anchors,args.seed+1)
        np.savez_compressed(output/'query_pairs.npz',anchors=anchors,
            **{offset+'_'+key:value for offset,pair in pairs.items() for key,value in pair.items() if isinstance(value,np.ndarray)})
        ties=rng.random(len(f))
        report=dict(arguments=vars(args),training_edges=int(r.nnz),active_users=len(active),sampled_anchors=len(anchors),
                    input_sha256={str(p):sha(p) for p in paths},
                    source_sha256={p.name:sha(p) for p in (Path(__file__),Path(__file__).with_name('build_semantic_user_graphs.py'))},variants={},
                    note='TRAIN-only undirected UI graph plus optional undirected II edges. Hops count physical edges: UIU=2, UIIU=3, UIUIU/UIIIU=4. '
                    'All original UI edges retained, no UU top-k truncation. Same uniform anchor sample and matched query pairs across variants. '
                    'Per-anchor reachability is exact; reported averages/paired rates are sample estimates. No confidence intervals or ranking accuracy. '
                    'Null II graphs use finite degree-preserving rewiring, not guaranteed uniform samples. ID offsets used solely for evaluation.')
        def run(name,graph,metadata=None):
            print('Evaluating '+name,flush=True)
            report['variants'][name]=evaluate(r,graph,anchors,pairs,args.block)
            if metadata:report['variants'][name]['rewiring']=metadata
            if graph.nnz:sp.save_npz(output/(name+'_item_graph.npz'),graph)
            (output/'report.json').write_text(json.dumps(report,indent=2))
        run('full_shared_items',sp.csr_matrix((len(f),len(f)),dtype=bool))
        for k in sorted(set(args.item_k)):
            graph=undirected_item_graph(f,k,128,ties)
            run(f'semantic_k{k}',graph)
            for repeat in range(args.null_repeats):
                null,info=rewire(graph,args.seed+1000*k+repeat)
                run(f'degree_rewired_k{k}_r{repeat}',null,info)
        (output/'status.json').write_text(json.dumps(dict(state='complete',seconds=time.time()-start)))
        print('Finished: '+str(output/'report.json'),flush=True)
    except BaseException as error:
        (output/'status.json').write_text(json.dumps(dict(state='failed',error=repr(error),seconds=time.time()-start)))
        raise


if __name__=='__main__':main()
