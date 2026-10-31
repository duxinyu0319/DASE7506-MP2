"""Two-hidden-layer selector, trained from scratch on supplied training data."""
from pathlib import Path
import json
import numpy as np
from common import parse_answer
from features import design

def raw(x,m):return np.tanh(np.tanh(x@m['W1']+m['b1'])@m['W2']+m['b2'])@m['u']+m['bias']
def predict(bank,arrays,checkpoint):
 if not bank:return []
 root=Path(checkpoint);cfg=json.loads((root/'config.json').read_text())
 with np.load(root/cfg['file'],allow_pickle=False) as m:
  x=np.clip((design(bank,arrays,'combined')-m['mean'])/m['scale'],-8,8);z=raw(x,m).reshape(len(bank),4)
 for i,r in enumerate(bank):
  for j,c in enumerate(r['candidates']):
   if parse_answer(c['text']) is None:z[i,j]=-1e30
 return z.argmax(1).tolist()
