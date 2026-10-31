"""Training/validation supervision helper. Not imported by inference."""
import numpy as np
from common import DATA,read_jsonl,parse_answer
def labels(bank,split):
    if split not in ('train','validation'):raise ValueError('Training helpers do not accept test references')
    refs=read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    assert [r['id'] for r in refs]==[r['id'] for r in bank]
    return np.array([[int((a:=parse_answer(c['text'])) is not None and a==parse_answer(r['answer'])) for c in b['candidates']] for b,r in zip(bank,refs)],dtype=float)
