# L2B-ProBio

L2B-ProBio classifies bacterial samples as probiotic or non-probiotic. The main model combines a convolutional network over 32 bp 2bRAD reads with ten gene features derived from Prokka annotations. This repository is a compact release of the research code: it keeps the executable pipeline, the exact 8-mer vocabulary, the matching gene-feature artifacts, and the seed-1 hybrid checkpoint while omitting duplicate checkpoints, caches, logs, and large raw datasets.

## Repository contents

```text
assets/
  checkpoints/hybrid_seed1.pt    pretrained hybrid CNN
  gene_features/                 matching top-10 gene weights
  kmer/kmer_vocab.json           8-mer vocabulary
examples/                        two tiny inputs for a smoke test
l2bprobio/                       training, evaluation, and prediction code
scripts/run_prokka.sh            FASTA-to-Prokka annotation helper
```

The pretrained checkpoint is about 48 MB, below GitHub's 100 MB per-file limit. The raw 2bRAD reads, genome FASTA files, and per-genome Prokka outputs are not included. They were not present as distributable source data in the working project and are much larger than the code. Exact paper metrics require those original inputs.

Verify the packaged binary and data assets at any time with `shasum -a 256 -c SHA256SUMS`.

## Environment

Python 3.10 and PyTorch 2.2 or newer are supported. Create the Conda environment from the repository root:

```bash
conda env create -f environment.yml
conda activate l2b-probio
```

For an existing Python environment, install PyTorch for the desired CPU/CUDA platform and then install this project:

```bash
python -m pip install -e .
```

Prokka is only needed when annotations have not already been generated. A separate environment avoids dependency conflicts:

```bash
conda create -n prokka -c conda-forge -c bioconda prokka
conda activate prokka
bash scripts/run_prokka.sh /path/to/genomes /path/to/prokka_outputs 8
```

Each FASTA filename must use the same sample ID as its 2bRAD text file. The helper produces `/path/to/prokka_outputs/<sample_id>/<sample_id>.tsv`.

## Input layout

For training and evaluation, put one 32 bp read on each line and organize samples by label:

```text
data/reads/
  probiotic/
    sample_001.txt
  non_probiotic/
    sample_002.txt
```

Prediction accepts a flat directory:

```text
data/unlabeled/
  sample_101.txt
  sample_102.txt
```

Reads longer than 32 bp are truncated to their first 32 bases. Reads shorter than 32 bp are rejected. By default, 4,096 reads are selected deterministically per sample; samples with fewer reads are resampled with replacement.

## Verify the packaged model

Inspect the checkpoint architecture:

```bash
python -m l2bprobio inspect
```

It should report a vocabulary size of 65,537, a read dimension of 128, and a gene-feature dimension of 10. Run the included two-sample smoke test on CPU:

```bash
python -m l2bprobio predict \
  --reads-dir examples/reads \
  --prokka-dir examples/prokka \
  --sample-reads 16 \
  --batch-size 2 \
  --device cpu \
  --output results/demo_predictions.tsv
```

This checks data loading, gene-feature extraction, checkpoint loading, and inference. The demo sequences are synthetic and are not intended to validate biological performance.

## Reproduce the main evaluation

With the original labeled 2bRAD files and matching Prokka outputs in place, evaluate the packaged seed-1 hybrid model with the original 80/10/10 stratified split:

```bash
python -m l2bprobio evaluate \
  --reads-dir data/reads \
  --prokka-dir data/prokka \
  --checkpoint assets/checkpoints/hybrid_seed1.pt \
  --gene-feature-dir assets/gene_features \
  --vocab assets/kmer/kmer_vocab.json \
  --seed 1 \
  --split 0.8,0.1,0.1 \
  --sample-reads 4096 \
  --batch-size 64 \
  --device auto \
  --output-dir results/pretrained_seed1
```

The command writes `metrics.json` and `predictions.tsv`. Metrics include accuracy, balanced accuracy, F1, precision, recall, ROC AUC, and the confusion matrix. Sample discovery is sorted by filename before the seeded stratified split, matching the original pipeline.

## Retrain the main model

The original main configuration used seed 1, 200 epochs, 128-dimensional read embeddings, AdamW at `1e-4`, 4,096 reads per sample, and the ten-gene hybrid branch:

```bash
python -m l2bprobio train \
  --reads-dir data/reads \
  --prokka-dir data/prokka \
  --hybrid \
  --gene-feature-dir assets/gene_features \
  --vocab assets/kmer/kmer_vocab.json \
  --seed 1 \
  --split 0.8,0.1,0.1 \
  --sample-reads 4096 \
  --batch-size 128 \
  --read-dim 128 \
  --epochs 200 \
  --learning-rate 1e-4 \
  --device cuda \
  --output checkpoints/hybrid_seed1_retrained.pt
```

The best validation-balanced-accuracy checkpoint is saved. Evaluate it by replacing `--checkpoint` in the evaluation command. Use `--device cpu` when CUDA is unavailable; training will be considerably slower.

## Predict new samples

Generate Prokka annotations from matching genome FASTA files first, then run:

```bash
python -m l2bprobio predict \
  --reads-dir data/unlabeled \
  --prokka-dir data/prokka_unlabeled \
  --checkpoint assets/checkpoints/hybrid_seed1.pt \
  --sample-reads 4096 \
  --batch-size 16 \
  --output results/predictions.tsv
```

The output contains the sample ID, probiotic probability, and thresholded class. The decision threshold is 0.5.

