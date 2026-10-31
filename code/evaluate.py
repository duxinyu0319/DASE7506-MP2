"""Fresh official prediction and independent scoring, including external timing."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import argparse,subprocess,sys,time,json,platform
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--split',choices=['validation','test'],default='test');args=p.parse_args();base=Path(__file__).resolve().parent;out=base/'results';out.mkdir(exist_ok=True);start=time.perf_counter()
for cmd in [[sys.executable,'predict.py','--method','submission.py','--checkpoint','checkpoint','--split',args.split,'--output',f'results/{args.split}-predictions.json'],[sys.executable,'score_predictions.py','--split',args.split,'--predictions',f'results/{args.split}-predictions.json','--output',f'results/{args.split}-score.json']]:subprocess.run(cmd,cwd=base,check=True)
meta={'split':args.split,'fresh_prediction_plus_independent_scoring_seconds':time.perf_counter()-start,'python':platform.python_version(),'hardware':{'system':platform.system(),'architecture':platform.machine(),'os_release':platform.release(),'processor':platform.processor(),'macOS_version':platform.mac_ver()[0]},'threads':4,'reference_Xeon_verified':False}
(out/f'{args.split}-resources.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2))
