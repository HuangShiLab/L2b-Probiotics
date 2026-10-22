#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 FASTA_DIR OUTPUT_DIR [THREADS]" >&2
  exit 2
fi

fasta_dir=$1
output_dir=$2
threads=${3:-8}
mkdir -p "$output_dir"

shopt -s nullglob
fastas=("$fasta_dir"/*.fa "$fasta_dir"/*.fna "$fasta_dir"/*.fasta "$fasta_dir"/*.fsa)
if (( ${#fastas[@]} == 0 )); then
  echo "No FASTA files found in $fasta_dir" >&2
  exit 1
fi

for fasta in "${fastas[@]}"; do
  filename=$(basename "$fasta")
  stem=${filename%.*}
  sample_output="$output_dir/$stem"
  if [[ -f "$sample_output/$stem.tsv" ]]; then
    echo "Skipping $stem (annotation already exists)"
    continue
  fi
  prokka \
    --outdir "$sample_output" \
    --prefix "$stem" \
    --cpus "$threads" \
    "$fasta"
done

