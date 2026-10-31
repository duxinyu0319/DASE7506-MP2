import json
from pathlib import Path
import numpy as np
from features import evidence,design
from common import parse_answer

def features(bank,arrays,mean,P):
 e=arrays['embeddings'];e=e/np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-8);a=(e-mean)@P;f=evidence(bank);x=np.concatenate([a,a-a.mean(1,keepdims=True),f,f-f.mean(1,keepdims=True)],axis=-1);return x.reshape(-1,x.shape[-1])
def tree_predict(x,node):
 out=np.empty(len(x));stack=[(node,np.arange(len(x)))]
 while stack:
  n,ix=stack.pop()
  if 'value' in n:out[ix]=n['value'];continue
  m=x[ix,n['feature']]<=n['threshold'];stack.extend([(n['left'],ix[m]),(n['right'],ix[~m])])
 return out
def predict(bank,arrays,checkpoint):
 if not bank:return []
 root=Path(checkpoint);cfg=json.loads((root/'config.json').read_text())
 with np.load(root/'transforms.npz') as m:x=features(bank,arrays,m['mean'],m['P'])
 t=json.loads((root/'trees.json').read_text());z=np.full(len(x),t['bias'])
 for node in t['trees']:z+=t['rate']*tree_predict(x,node)
 z=z.reshape(-1,4)
 with np.load(root/'baseline.npz') as m:
  d=np.clip((design(bank,arrays,'combined')-m['mean'])/m['scale'],-8,8);base=(np.tanh(np.tanh(d@m['W1']+m['b1'])@m['W2']+m['b2'])@m['u']+m['bias']).reshape(-1,4)
 norm=lambda s:(s-s.mean(1,keepdims=True))/np.maximum(s.std(1,keepdims=True),.1)
 z=cfg['tree_weight']*norm(z)+(1-cfg['tree_weight'])*norm(base)
 for i,r in enumerate(bank):
  for j,c in enumerate(r['candidates']):
   if parse_answer(c['text']) is None:z[i,j]=-1e30
 return z.argmax(1).tolist()
