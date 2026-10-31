import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import json,time,shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from common import load_inputs,parse_answer
from features import design
from training_labels import labels
from tree_selector import features,tree_predict
start=time.perf_counter();root=Path('runs');root.mkdir(exist_ok=True);tb,ta=load_inputs('train');vb,va=load_inputs('validation');Y=labels(tb,'train');vy=labels(vb,'validation');e=ta['embeddings'].reshape(-1,1024);e=e/np.maximum(np.linalg.norm(e,axis=1,keepdims=True),1e-8);mean=e.mean(0);cov=(e-mean).T@(e-mean)/len(e);vals,P=np.linalg.eigh(cov);P=P[:,-32:].astype('float32');X=features(tb,ta,mean,P);V=features(vb,va,mean,P)
valid=np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in vb]);hard=lambda b:np.array([len(set(a for c in r['candidates'] if (a:=parse_answer(c['text'])) is not None))>=2 for r in b]);ht=hard(tb);hv=hard(vb)
with np.load('training_inputs/baseline.npz') as m:
 d=np.clip((design(vb,va,'combined')-m['mean'])/m['scale'],-8,8);base=(np.tanh(np.tanh(d@m['W1']+m['b1'])@m['W2']+m['b2'])@m['u']+m['bias']).reshape(-1,4)
norm=lambda s:(s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),.1)
def metric(z):
 hit=vy[np.arange(len(vy)),np.where(valid,z,-1e30).argmax(1)];return {'hard_validation_accuracy':float(hit[hv].mean()*100),'validation_accuracy':float(hit.mean()*100)}
records=[];bestkey=(-1,-1);best=None
for subset in ['all','hard']:
 ix=np.repeat(np.ones(len(tb),dtype=bool) if subset=='all' else ht,4);x=X[ix];y=Y.ravel()[ix];edges=[np.unique(np.quantile(x[:,j],np.linspace(0,1,33)[1:-1])) for j in range(x.shape[1])];bins=np.array([np.searchsorted(ed,x[:,j]) for j,ed in enumerate(edges)],dtype=np.uint8).T
 for rate in [.05,.15]:
  bias=float(np.log((y.mean()+1e-5)/(1-y.mean()+1e-5)));z=np.full(len(y),bias);v=np.full(len(V),bias);trees=[]
  for iteration in range(1,161):
   p=expit(z);g=y-p;h=np.maximum(p*(1-p),.01)
   def grow(ix,depth):
    G=g[ix].sum();H=h[ix].sum();value=float(np.clip(G/(H+5),-3,3))
    if depth==0 or len(ix)<80:return {'value':value}
    gain=0;choice=None
    for j,ed in enumerate(edges):
     if not len(ed):continue
     gb=np.bincount(bins[ix,j],weights=g[ix],minlength=len(ed)+1);hb=np.bincount(bins[ix,j],weights=h[ix],minlength=len(ed)+1);cb=np.bincount(bins[ix,j],minlength=len(ed)+1);gl=gb.cumsum()[:-1];hl=hb.cumsum()[:-1];cl=cb.cumsum()[:-1];score=gl**2/(hl+5)+(G-gl)**2/(H-hl+5)-G**2/(H+5);score[(cl<40)|(len(ix)-cl<40)]=-1e9;k=int(score.argmax())
     if score[k]>gain:gain=score[k];choice=(j,k)
    if choice is None:return {'value':value}
    j,k=choice;m=bins[ix,j]<=k;return {'feature':j,'threshold':float(edges[j][k]),'left':grow(ix[m],depth-1),'right':grow(ix[~m],depth-1)}
   node=grow(np.arange(len(y)),3);trees.append(node);z+=rate*tree_predict(x,node);v+=rate*tree_predict(V,node)
   if iteration%20==0:
    for w in [.25,.5,.75,1.]:
     met=metric(w*norm(v.reshape(-1,4))+(1-w)*norm(base));rec={'subset':subset,'rate':rate,'trees':iteration,'tree_weight':w,**met};records.append(rec);key=(met['hard_validation_accuracy'],met['validation_accuracy'])
     if key>bestkey:bestkey=key;best=(json.loads(json.dumps(trees)),bias,rate,rec)
  print(json.dumps(records[-1]),flush=True)
trees,bias,rate,rec=best;bundle=root/'model_bundle';bundle.mkdir(exist_ok=True);np.savez(bundle/'transforms.npz',mean=mean,P=P);(bundle/'trees.json').write_text(json.dumps({'bias':bias,'rate':rate,'trees':trees}));shutil.copyfile('training_inputs/baseline.npz',bundle/'baseline.npz');cfg={'public_bank_development_exposed':True,'test_references_used_for_fitting':False,**rec};(bundle/'config.json').write_text(json.dumps(cfg,indent=2));(root/'experiments.json').write_text(json.dumps(records,indent=2));(root/'training_record.json').write_text(json.dumps({'seconds':time.perf_counter()-start,'fits':4,'max_trees':160,'PCA_components':32,'depth':3,'configuration':cfg},indent=2));print(json.dumps(cfg,indent=2),flush=True)
