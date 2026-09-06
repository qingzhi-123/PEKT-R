# PEKT-R

This repository implements PEKT-R, an adaptive programming exercise
recommendation algorithm that combines programming error representation,
knowledge tracing, and error-correction-aware ranking.

The implementation follows the paper pipeline:

1. Build a multi-label programming-error vector from judge status, code,
   problem tags, and judge feedback.
2. Fuse problem, knowledge concept, answer result, judge status, and error
   features into an error-aware diagnostic Transformer.
3. Predict each learner's dynamic knowledge mastery state.
4. Rank candidate exercises with:

```text
Score(u, q) = alpha * KMatch(u, q)
            + beta  * DMatch(u, q)
            + gamma * ECorrect(u, q)
```

## Data

The BePKT dataset is expected at:

```powershell
data\BePKT
```

The dataset is not included in this repository. Download BePKT separately and
place it under `data\BePKT` before preprocessing.

## Quick Start

Install dependencies in the existing virtual environment:

```powershell
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install -e .
```

Preprocess BePKT:

```powershell
.\.venv\Scripts\python -m pektr.preprocess_bepkt --data-dir data\BePKT --out-dir artifacts\bepkt
```

Train PEKT-R:

```powershell
.\.venv\Scripts\python -m pektr.train --artifact-dir artifacts\bepkt --epochs 20 --batch-size 64
```

Evaluate with leave-one-out ranking:

```powershell
.\.venv\Scripts\python -m pektr.evaluate --artifact-dir artifacts\bepkt --checkpoint checkpoints\pektr.pt --k 10 --num-negatives 100
```

Tune on GPU:

```powershell
.\.venv\Scripts\python -m pektr.tune --artifact-dir artifacts\bepkt --trials 8 --epochs 4 --batch-size 128 --device cuda
```

Search ranking weights for a trained checkpoint:

```powershell
.\.venv\Scripts\python -m pektr.score_search --artifact-dir artifacts\bepkt --checkpoint checkpoints\pektr.pt --split val --device cuda
```

## RTX 5070 Ti Tuned Configuration

The current tuned checkpoint is:

```powershell
checkpoints\pektr_5070ti_best.pt
```

The tuned checkpoint is not included in the Git repository. Reproduce it with
the training command below.

It was trained on the local RTX 5070 Ti with:

```powershell
.\.venv\Scripts\pektr-train.exe --artifact-dir artifacts\bepkt --checkpoint checkpoints\pektr_5070ti_d128.pt --epochs 30 --batch-size 256 --max-seq-len 128 --d-model 128 --n-heads 4 --n-layers 2 --dim-feedforward 256 --dropout 0.1 --lr 0.0005 --weight-decay 0.00005 --device cuda
```

Use the tuned ranking weights:

```powershell
--alpha 0.05 --beta 0.20 --gamma 0.75 --rho 0.75
```

The final tuning record is stored in:

```powershell
artifacts\tuning\final_5070ti_config.json
```

Recommend exercises for a learner:

```powershell
.\.venv\Scripts\python -m pektr.recommend --artifact-dir artifacts\bepkt --checkpoint checkpoints\pektr.pt --user-id 34 --top-n 10
```

## Notes

BePKT does not provide hand-labeled fine-grained programming-error labels.
The preprocessor therefore creates paper-aligned weak labels from raw judge
status, judge details, problem tags, and code heuristics. If manually labeled
error data or a CodeT5+/PLCodeBERT classifier is available, replace the
`error_vectors` field emitted by preprocessing and the PEKT-R model can use it
directly.
