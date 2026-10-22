from __future__ import annotations

import csv
import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ACCESSION_PREFIXES = ("NZ_", "NC_", "GCA_", "GCF_", "CP", "AE", "AP", "CM", "LR")
HYPOTHETICAL = re.compile(r"\bhypothetical protein\b", re.IGNORECASE)


def sample_stem(sample_id: str) -> str:
    name = Path(sample_id).name
    for suffix in (".txt", ".fa", ".fna", ".fasta", ".fsa", ".tsv", ".gff"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return name


def normalize_sample_id(sample_id: str) -> str:
    stem = sample_stem(sample_id)
    return stem if stem.startswith(ACCESSION_PREFIXES) else stem.split("_", 1)[0]


def find_annotation(root: str | Path, sample_id: str) -> Path | None:
    root = Path(root)
    raw_stem = sample_stem(sample_id)
    stems = dict.fromkeys((raw_stem, normalize_sample_id(sample_id)))
    for stem in stems:
        bases = [root / f"{stem}_prokka" / stem, root / stem / stem, root / stem]
        for suffix in (".tsv", ".gff"):
            for base in bases:
                candidate = Path(f"{base}{suffix}")
                if candidate.exists():
                    return candidate
    return None


def _simple_product(product: str) -> str | None:
    product = product.strip()
    if not product or HYPOTHETICAL.search(product):
        return None
    token = product.replace("-", "_")
    return token if re.fullmatch(r"[A-Za-z]\w{1,30}", token) else None


def parse_tsv(path: Path) -> Counter[str]:
    genes: list[str] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            return Counter()
        columns = {name.lower(): name for name in reader.fieldnames}
        for row in reader:
            if row.get(columns.get("ftype", "")) != "CDS":
                continue
            gene = row.get(columns.get("gene", ""), "").strip()
            if gene:
                genes.append(gene)
            product = _simple_product(row.get(columns.get("product", ""), ""))
            if product:
                genes.append(product)
    return Counter(genes)


def parse_gff(path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with path.open() as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9 or fields[2] != "CDS":
                continue
            attributes = fields[8]
            gene_match = re.search(r"(?:^|;)gene=([^;]+)", attributes)
            if gene_match:
                counts[gene_match.group(1).strip()] += 1
                continue
            product_match = re.search(r"(?:^|;)product=([^;]+)", attributes)
            if product_match:
                product = _simple_product(product_match.group(1))
                if product:
                    counts[product] += 1
    return counts


class GeneFeatureBank:
    def __init__(self, artifacts_dir: str | Path, annotations_dir: str | Path):
        artifacts_dir = Path(artifacts_dir)
        self.annotations_dir = Path(annotations_dir)
        with (artifacts_dir / "gene_lor.json").open() as handle:
            self.log_odds = json.load(handle)
        with (artifacts_dir / "topk.json").open() as handle:
            self.genes = json.load(handle)
        self.gene_index = {gene: index for index, gene in enumerate(self.genes)}
        self.scaler = None
        scaler_path = artifacts_dir / "feature_scaler.pkl"
        if scaler_path.exists():
            with scaler_path.open("rb") as handle:
                self.scaler = pickle.load(handle)

    @property
    def feature_dim(self) -> int:
        return len(self.genes)

    def vector(self, sample_id: str) -> np.ndarray:
        annotation = find_annotation(self.annotations_dir, sample_id)
        if annotation is None:
            raise FileNotFoundError(
                f"No Prokka TSV/GFF found for {sample_id} below {self.annotations_dir}"
            )
        counts = parse_tsv(annotation) if annotation.suffix == ".tsv" else parse_gff(annotation)
        vector = np.zeros(self.feature_dim, dtype=np.float32)
        for gene, count in counts.items():
            index = self.gene_index.get(gene)
            if index is not None:
                vector[index] = count * float(self.log_odds.get(gene, 0.0))
        if self.scaler is not None:
            vector = np.asarray(self.scaler.transform(vector.reshape(1, -1))).ravel()
        return vector.astype(np.float32)

    def batch(self, sample_ids: list[str], device: torch.device) -> torch.Tensor:
        matrix = np.stack([self.vector(sample_id) for sample_id in sample_ids])
        return torch.as_tensor(matrix, dtype=torch.float32, device=device)

