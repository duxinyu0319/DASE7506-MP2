"""Input-only features. No reference access."""
import re
import numpy as np
from fractions import Fraction
from common import parse_answer
FEATURE_NAMES=['parseable','agreement','diversity','length','words','lines','digits','equals','question_length','integer','answer_magnitude','negative','boxed','hash','steps','uncertain','interpretation','relative_length']
def evidence(bank):
    out=[]
    for r in bank:
        ts=[c['text'] for c in r['candidates']];aa=[parse_answer(t) for t in ts];avg=max(np.mean([len(t) for t in ts]),1)
        for t,a in zip(ts,aa):
            v=float(Fraction(a)) if a is not None else 0;low=t.lower()
            out.append([a is not None,aa.count(a)/4 if a is not None else 0,len(set(x for x in aa if x is not None))/4,np.log1p(len(t)),np.log1p(len(t.split())),np.log1p(t.count('\n')),np.log1p(len(re.findall(r'\d+',t))),np.log1p(t.count('=')),np.log1p(len(r['question'])),v==int(v),np.log1p(min(abs(v),1e15)),v<0,'\\boxed' in t,'####' in t,np.log1p(low.count('step')),any(w in low for w in ['however','wait','mistake','incorrect']),any(w in low for w in ['assum','interpret','ambig']),len(t)/avg])
    return np.array(out,dtype=np.float32).reshape(len(bank),4,-1)
def design(bank,arrays,mode):
    e=arrays['embeddings'].astype(np.float32);e=e/np.maximum(np.linalg.norm(e,axis=-1,keepdims=True),1e-8)
    if mode=='embedding': x=e
    elif mode=='relative': x=np.concatenate([e,e-e.mean(axis=1,keepdims=True)],axis=-1)
    elif mode=='evidence': x=evidence(bank)
    elif mode=='combined':
        f=evidence(bank);x=np.concatenate([e,e-e.mean(axis=1,keepdims=True),f,f-f.mean(axis=1,keepdims=True)],axis=-1)
    else:raise ValueError(mode)
    return x.reshape(-1,x.shape[-1])
def majority(bank):
    picks=[]
    for r in bank:
        a=[parse_answer(c['text']) for c in r['candidates']];picks.append(int(np.argmax([a.count(v) if v is not None else -1 for v in a])))
    return picks
