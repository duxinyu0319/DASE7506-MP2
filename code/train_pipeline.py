"""Reproduce the final two-stage training process. No test evaluation is run."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='4'
import subprocess,sys,shutil,json
from pathlib import Path
base=Path(__file__).resolve().parent
work=base/'training_work'/'neural';work.mkdir(parents=True,exist_ok=True)
subprocess.run([sys.executable,str(base/'check_data.py')],cwd=base,check=True)
subprocess.run([sys.executable,str(base/'train_deep.py')],cwd=work,check=True)
config=json.loads((work/'runs/model_bundle/config.json').read_text());name=config['file']
if name!='deep_hard_7541.npz':print('NOTE: numeric differences changed selected neural checkpoint:',name)
(base/'training_inputs').mkdir(exist_ok=True);shutil.copyfile(work/'runs/model_bundle'/name,base/'training_inputs/baseline.npz')
subprocess.run([sys.executable,str(base/'train_trees.py')],cwd=base,check=True)
shutil.copytree(base/'runs/model_bundle',base/'checkpoint',dirs_exist_ok=True)
print('Training complete; evaluate the frozen checkpoint separately. Existing checkpoint has been overwritten.')
