"""Larger verifier, not a pretrained LLM. No test input/reference access.
Fits training labels; selection uses hard-validation then overall validation.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import time,json,shutil
from pathlib import Path
import numpy as np
from scipy.special import expit
from common import load_inputs,parse_answer
from features import design
from training_labels import labels
from deep_selector import raw

def sm(z):
 z=z-z.max(1,keepdims=True);e=np.exp(z);return e/e.sum(1,keepdims=True)
def main():
 start=time.perf_counter();root=Path('runs');pool=root/'candidates';pool.mkdir(parents=True,exist_ok=True)
 tb,ta=load_inputs('train');vb,va=load_inputs('validation');Y=labels(tb,'train').astype('float32');vy=labels(vb,'validation')
 valid=lambda bank:np.array([[parse_answer(c['text']) is not None for c in r['candidates']] for r in bank]);tv=valid(tb);vv=valid(vb)
 hard=lambda bank:np.array([len(set(a for c in r['candidates'] if (a:=parse_answer(c['text'])) is not None))>=2 for r in bank]);ht=hard(tb);hv=hard(vb)
 x=design(tb,ta,'combined');v=design(vb,va,'combined');mean=x.mean(0);scale=np.maximum(x.std(0),.001);X=np.clip((x-mean)/scale,-8,8).reshape(len(tb),4,-1);V=np.clip((v-mean)/scale,-8,8);d=X.shape[-1]
 def metric(z):
  hit=vy[np.arange(len(vy)),np.where(vv,z,-1e30).argmax(1)];return {'hard_validation_accuracy':float(hit[hv].mean()*100),'validation_accuracy':float(hit.mean()*100)}
 records=[]
 for subset in ['all','hard']:
  ixall=np.arange(len(X)) if subset=='all' else np.flatnonzero(ht)
  for seed in [7540,7541]:
   tick=time.perf_counter();rng=np.random.default_rng(seed);names=['W1','b1','W2','b2','u','bias'];pars=[rng.normal(0,1/np.sqrt(d),(d,128)).astype('float32'),np.zeros(128,dtype='float32'),rng.normal(0,1/np.sqrt(128),(128,64)).astype('float32'),np.zeros(64,dtype='float32'),rng.normal(0,1/np.sqrt(64),64).astype('float32'),np.zeros(1,dtype='float32')];m=[np.zeros_like(p) for p in pars];s=[np.zeros_like(p) for p in pars];step=0;bestkey=(-1,-1);best=None;history=[];decay=.001;dropout=.1
   for epoch in range(1,101):
    for ix in np.array_split(rng.permutation(ixall),20):
     W1,b1,W2,b2,u,bias=pars;xf=X[ix].reshape(-1,d);h1=np.tanh(xf@W1+b1);mask=(rng.random(h1.shape)>=dropout).astype('float32')/(1-dropout);h1d=h1*mask;h2=np.tanh(h1d@W2+b2);z=(h2@u+bias[0]).reshape(-1,4)
     zs=np.where(tv[ix],z,-1e9);zp=np.where(Y[ix]>0,zs,-1e9);delta=.5*(expit(z)-Y[ix])/(len(ix)*4)+.5*(sm(zs)-sm(zp))/len(ix);dz=delta.ravel();du=h2.T@dz+decay*u;dh2=dz[:,None]*u*(1-h2*h2);gW2=h1d.T@dh2+decay*W2;gb2=dh2.sum(0);dh1=(dh2@W2.T)*mask*(1-h1*h1);grads=[xf.T@dh1+decay*W1,dh1.sum(0),gW2,gb2,du,np.array([dz.sum()])];step+=1
     for p,g,mm,ss in zip(pars,grads,m,s):
      g=np.clip(g,-5,5);mm*=.9;mm+=.1*g;ss*=.999;ss+=.001*g*g;p-=.0005*(mm/(1-.9**step))/(np.sqrt(ss/(1-.999**step))+1e-8)
    if epoch%5==0:
     params=dict(zip(names,pars));z=raw(V,params).reshape(-1,4);met=metric(z);history.append({'epoch':epoch,**met});key=(met['hard_validation_accuracy'],met['validation_accuracy'])
     if key>bestkey:bestkey=key;best=([p.copy() for p in pars],epoch,met)
   pp,ep,met=best;name=f'deep_{subset}_{seed}.npz';np.savez(pool/name,mean=mean,scale=scale,**dict(zip(names,pp)))
   rec={'file':name,'training_subset':subset,'seed':seed,'hidden_units':[128,64],'dropout':dropout,'L2':decay,'learning_rate':.0005,'max_epochs':100,'selected_epoch':ep,'parameter_count':sum(p.size for p in pp),'seconds':time.perf_counter()-tick,'history':history,**met};records.append(rec);(root/'experiments.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps({k:v for k,v in rec.items() if k!='history'}),flush=True)
 best=max(records,key=lambda a:(a['hard_validation_accuracy'],a['validation_accuracy']));bundle=root/'model_bundle';bundle.mkdir(exist_ok=True);shutil.copyfile(pool/best['file'],bundle/best['file']);cfg={'file':best['file'],'architecture':'2084-128-64-1 tanh MLP','selection':'hard validation then overall validation','public_test_feedback_used_in_development':True,'test_reference_answers_used_for_training':False,'selected_epoch':best['selected_epoch'],'seed':best['seed'],'hard_validation_accuracy':best['hard_validation_accuracy'],'validation_accuracy':best['validation_accuracy']};(bundle/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');(root/'training_record.json').write_text(json.dumps({'total_seconds':time.perf_counter()-start,'fits':4,'parameter_count':best['parameter_count'],'asset_bytes':sum(p.stat().st_size for p in bundle.iterdir()),'configuration':cfg},indent=2)+'\n');print(json.dumps(cfg,indent=2),flush=True)
if __name__=='__main__':main()
