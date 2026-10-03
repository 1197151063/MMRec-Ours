"""Four TRAIN-only UU graph hypotheses. Numeric ID distance is diagnostic only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp


def normalize(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


def cosine_rows(x):
    n=np.sqrt(np.asarray(x.multiply(x).sum(1)).ravel())
    return (sp.diags(1/np.maximum(n,1e-12))@x).tocsr()


def select_graph(x,k,block,ties,overlap=False):
    degree=np.asarray(x.sum(1)).ravel()
    z=x if overlap else cosine_rows(x)
    rows=[];cols=[];weights=[]
    for start in range(0,x.shape[0],block):
        s=(z[start:start+block]@z.T).tocsr()
        for local in range(s.shape[0]):
            u=start+local
            v=s.indices[s.indptr[local]:s.indptr[local+1]]
            w=s.data[s.indptr[local]:s.indptr[local+1]].copy()
            mask=(v!=u)&(w>0);v,w=v[mask],w[mask]
            if overlap:w/=np.maximum(np.minimum(degree[u],degree[v]),1)
            if len(w)>k:
                threshold=np.partition(w,-k)[-k];mask=w>=threshold;v,w=v[mask],w[mask]
            choice=np.lexsort((ties[v],-w))[:k];v,w=v[choice],w[choice]
            rows.extend([u]*len(v));cols.extend(v.tolist());weights.extend(w.tolist())
    return sp.csr_matrix((weights,(rows,cols)),shape=(x.shape[0],)*2,dtype=np.float32)


def anchor_history(r,features,block=512):
    """Each interaction weighted by agreement with the user's OTHER items."""
    total=np.asarray(r@features)
    coo=r.tocoo();affinity=np.empty(r.nnz,dtype=np.float32)
    for start in range(0,r.nnz,block):
        u=coo.row[start:start+block];i=coo.col[start:start+block]
        other=normalize(total[u]-features[i])
        similarity=np.sum(other*features[i],axis=1).clip(-1,1)
        affinity[start:start+len(u)]=.5+.5*similarity
    return sp.csr_matrix((affinity,(coo.row,coo.col)),shape=r.shape)


def item_transition(features,k,block,ties):
    if k>=len(features):raise ValueError('item-k must be less than item count')
    rows=[];cols=[];weights=[]
    for start in range(0,len(features),block):
        scores=features[start:start+block]@features.T
        for offset,w in enumerate(scores):
            i=start+offset;w[i]=-np.inf
            candidates=np.flatnonzero(w>0)
            if len(candidates)>k:
                cut=np.partition(w[candidates],-k)[-k];candidates=candidates[w[candidates]>=cut]
            order=np.lexsort((ties[candidates],-w[candidates]))[:k];js=candidates[order]
            values=w[js];values=values/max(values.sum(),1e-12)
            rows.extend([i]*len(js));cols.extend(js.tolist());weights.extend(values.tolist())
    s=sp.csr_matrix((weights,(rows,cols)),shape=(len(features),)*2,dtype=np.float32)
    # Isolated semantic rows remain identity rows.
    identity_weight=np.where(np.asarray(s.sum(1)).ravel()>0,.5,1.)
    return .5*s+sp.diags(identity_weight)


def edge_summary(graph,r):
    coo=graph.tocoo();dist=np.abs(coo.row-coo.col)
    shared=[]
    for start in range(0,len(coo.data),4096):
        u=coo.row[start:start+4096];v=coo.col[start:start+4096]
        shared.extend(np.asarray(r[u].multiply(r[v]).sum(1)).ravel().tolist())
    return dict(edges=graph.nnz,users_with_neighbors=int(np.count_nonzero(np.diff(graph.indptr))),
                fraction_sharing_training_item=float(np.mean(np.array(shared)>0)) if shared else None,
                id_windows={str(d):float(np.mean(dist<=d)) if len(dist) else None for d in (1,4,16,64,256)})


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-path',required=True);p.add_argument('--dataset',default='baby')
    p.add_argument('--output',default='night_runs/baby-semantic-uu-1')
    p.add_argument('--k',type=int,default=20);p.add_argument('--item-k',type=int,default=10)
    p.add_argument('--block',type=int,default=128);p.add_argument('--seed',type=int,default=2026)
    args=p.parse_args()
    if min(args.k,args.item_k,args.block)<1:p.error('k, item-k and block must be positive')
    folder=Path(args.data_path)/args.dataset;out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    paths=[folder/(args.dataset+'.inter'),folder/'image_feat.npy',folder/'text_feat.npy']
    df=pd.read_csv(paths[0],sep='\t',usecols=['userID','itemID','x_label'])
    edges=df[df.x_label==0][['userID','itemID']].drop_duplicates()
    if edges.empty or (edges.to_numpy()<0).any():raise ValueError('Invalid training edges')
    visual=np.load(paths[1],allow_pickle=False).astype(np.float32)
    text=np.load(paths[2],allow_pickle=False).astype(np.float32)
    if len(visual)!=len(text) or edges.itemID.max()>=len(visual):raise ValueError('Feature/index mismatch')
    if not np.isfinite(visual).all() or not np.isfinite(text).all():raise ValueError('Nonfinite features')
    features=normalize(np.concatenate((normalize(visual),normalize(text)),axis=1))
    r=sp.csr_matrix((np.ones(len(edges),dtype=np.float32),(edges.userID,edges.itemID)),shape=(int(edges.userID.max())+1,len(features)))
    rng=np.random.default_rng(args.seed);ties=rng.random(r.shape[0]);item_ties=rng.random(r.shape[1])
    report=dict(arguments=vars(args),input_sha256={str(f):sha(f) for f in paths},training_edges=r.nnz,
                graphs={},note='No ID distances in graph construction. Self edges excluded. '
                'Equal scores use random entity tie keys; exact relabeling requires co-permuting those keys. '
                'ID enrichment is descriptive, never a tuning/selection objective. No ranking gains measured.')
    for name in ('shared_cosine','overlap','semantic_anchor','semantic_bridge'):
        print('Building '+name,flush=True)
        if name=='semantic_anchor':x=anchor_history(r,features)
        elif name=='semantic_bridge':x=(r@item_transition(features,args.item_k,args.block,item_ties)).tocsr()
        else:x=r
        graph=select_graph(x,args.k,args.block,ties,overlap=name=='overlap')
        sp.save_npz(out/(name+'.npz'),graph)
        row_sum=np.asarray(graph.sum(1)).ravel()
        sp.save_npz(out/(name+'_rownorm.npz'),sp.diags(1/np.maximum(row_sum,1e-12))@graph)
        report['graphs'][name]=edge_summary(graph,r)
        (out/'report.json').write_text(json.dumps(report,indent=2))
        print(name+': '+json.dumps(report['graphs'][name]),flush=True)
    print('Finished: '+str(out/'report.json'),flush=True)


if __name__=='__main__':main()
