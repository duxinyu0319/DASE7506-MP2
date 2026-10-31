# DASE7506 MP2 - final tree/neural selector

**Official test: 381/467 = 81.5845824411135%. Validation: 747/798 = 93.60902255639098%.**
This is the strongest measured public-bank model from the recorded studies. Test reference answers were not used for fitting parameters or transformations. Read report.pdf and AI_DISCLOSURE.md. 

## Deliverables
- `report.pdf`: five-page English report.
- `code/submission.py`: required predict(bank, arrays, checkpoint) interface.
- `code/tree_selector.py`, `features.py`: actual final inference.
- `code/checkpoint/`: every inference asset; frozen, no retraining needed.
- `code/results/test-score.json`: official JSON; submit its accuracy field.
- `code/results/test-resources.json`: external fresh-process prediction + scoring time.
- `code/train_pipeline.py`, `train_deep.py`, `train_trees.py`, `training_labels.py`: complete final model training pipeline.
- `evidence/`: original training/search records, selected earlier results, interface/immutability checks, and model/asset hashes.
- `RELEASE_SHA256.json`: exact local release file identities; not a remote Git commit.
- `SUBMISSION_STEPS_中文.md`: manual GitHub and course submission checklist.

## Install (Python 3.12)
The lightweight ZIP omits the original supplied data. Copy the whole **unchanged** official `data` directory into `code/data`, preserving candidates/, manifest.json and references. Obtain it from https://github.com/xudongwu-0/DASE7506-MP2 . Preserve LICENSE.txt and NOTICE.txt when redistributing supplied data. From the extracted root:
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r code/requirements-method.txt
python -m pip check
cd code
python check_data.py
```
Pinned executed environment: Python3.12.10, NumPy2.5.4, SciPy1.18.1. SciPy is training-only; frozen inference needs NumPy. Installation requires package access; prediction/scoring is offline afterward. No API key or additional pretrained weights are needed. See INSTALL_MAC.md for troubleshooting.

## Reproduce submitted score WITHOUT training
From `code/` with the environment active:
```bash
python evaluate.py --split test
```
This caps OMP/OpenBLAS/MKL/VECLIB threads at four and launches a fresh official prediction process and independent scorer. It includes interpreter/import startup, loading and preprocessing in an external wall clock. Exact underlying commands:
```bash
python predict.py --method submission.py --checkpoint checkpoint --split test --output results/test-predictions.json
python score_predictions.py --split test --predictions results/test-predictions.json --output results/test-score.json
```
For validation: `python evaluate.py --split validation`.
Expected test: correct381, n467, accuracy81.5845824411135. 

## Reproduce training
Back up `checkpoint/` before retraining. From `code/`:
```bash
python train_pipeline.py
```
Stage1 trains four neural configurations (subsets all/hard; seeds7540/7541), each100epochs; the chosen train-only neural model is hard subset, seed7541, epoch5. Architecture2084-128-64-1; Adam0.0005; L2=0.001; dropout0.1. Stage2 trains four boosted-tree configurations (subsets all/hard; rates0.05/0.15), each up to160trees, testing four fusion weights every20trees. Train-only PCA32 plus relative and evidence features gives100tree inputs. Tree depth3, min40samples per leaf, Hessian L2=5. Selected hard subset,140trees,rate0.15,tree weight0.25 plus neural0.75 after within-question logit standardization. Validation criterion: hard-validation accuracy then overall-validation accuracy. Training pipeline does not invoke test prediction or scoring. All means,scales,PCA and fitted parameters are estimated from training only. Numeric library differences may change close validation ties; use the packaged frozen checkpoint for exact score reproduction.

`training_inputs/baseline.npz` is the same frozen neural comparator used by the final tree training. The full pipeline regenerates it. It is training-only and is not loaded by submission.py; inference uses checkpoint/baseline.npz. Selected inference assets total1,339,071bytes (about1.277MiB). Even both copies together remain well below16MiB.

## Audit inputs
```bash
python predict.py --method submission.py --checkpoint checkpoint --split audit --data-dir /path/to/audit_inputs --output results/audit-predictions.json
```
The method consumes passed bank/arrays and supports new bank sizes. IDs are not features. No reference file, lookup table or cached prediction is read by inference. The official parser/data/scorer remain unchanged.

## Resources, disclosure, version and submission
Local measurement: Apple M5 Max,36GB RAM,macOS26.4,four-thread caps; external test prediction+scoring0.218969seconds. Reference Xeon Platinum8457C eligibility has not been measured. Stage1 internal training/search18.10seconds,Stage2 internal25.49seconds,excluding interpreter/import startup. This is CPU-only work. 

Upload this release to a public repository, record its exact commit/release and matching checkpoint. Code link should identify that frozen version. Checkpoint link must provide all four files in code/checkpoint/; a separate MP2_Final_Checkpoint.zip is also provided. Fill your student ID and official test accuracy on the course site, then actually create the GitHub issue. This package has not been uploaded to your GitHub account and no course issue has been created. Deadline:31October2026,23:59,UTC+8. Instructor audit/peer reproduction remains authoritative.

## Student local reproduction
The student reproduced 381/467 (81.5845824411135%) on Apple M5 Max, Mac17,6, 36 GiB RAM, macOS26.4 ARM64, Python3.12.10. Fresh prediction plus independent scoring took 0.21896933300013188 seconds. Actual machine metadata is in code/results/local-machine.json. Reference Xeon verification remains pending.
