from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset


CLASS_DIRS = {"non_probiotic": 0, "probiotic": 1}


def load_vocab(path: str | Path) -> dict[str, int]:
    with Path(path).open() as handle:
        vocab = json.load(handle)
    if "<UNK>" not in vocab:
        raise ValueError(f"Vocabulary {path} has no <UNK> token")
    return vocab


def discover_labeled_samples(root: str | Path) -> tuple[list[Path], list[int]]:
    root = Path(root)
    records: list[tuple[Path, int]] = []
    for dirname, label in CLASS_DIRS.items():
        records.extend((path, label) for path in (root / dirname).glob("*.txt"))
    records.sort(key=lambda item: (item[0].name, str(item[0])))
    if not records:
        raise FileNotFoundError(
            f"No labeled .txt samples found below {root}; expected "
            "non_probiotic/ and probiotic/ directories"
        )
    paths, labels = zip(*records)
    return list(paths), list(labels)


def discover_unlabeled_samples(root: str | Path) -> list[Path]:
    paths = sorted(Path(root).glob("*.txt"))
    if not paths:
        raise FileNotFoundError(f"No .txt samples found directly in {root}")
    return paths


class ReadDataset(Dataset):
    """Encode each 32 bp read as four non-overlapping 8-mer IDs."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        vocab_path: str | Path,
        labels: Sequence[int] | None = None,
        sample_reads: int = 4096,
        seed: int = 1,
    ) -> None:
        if sample_reads <= 0:
            raise ValueError("sample_reads must be positive")
        self.paths = [Path(path) for path in paths]
        self.labels = list(labels) if labels is not None else None
        if self.labels is not None and len(self.labels) != len(self.paths):
            raise ValueError("paths and labels must have the same length")
        self.vocab = load_vocab(vocab_path)
        self.unknown = self.vocab["<UNK>"]
        self.sample_reads = sample_reads
        self.seed = seed

    def __len__(self) -> int:
        return len(self.paths)

    def _rng(self, path: Path) -> random.Random:
        digest = hashlib.sha256(f"{self.seed}:{path.name}".encode()).digest()
        return random.Random(int.from_bytes(digest[:8], "little"))

    def _sample_sequences(self, path: Path) -> list[str]:
        with path.open() as handle:
            sequences = [line.strip().upper() for line in handle if line.strip()]
        too_short = sum(len(sequence) < 32 for sequence in sequences)
        if too_short:
            raise ValueError(f"{path} contains {too_short} reads shorter than 32 bp")
        if not sequences:
            raise ValueError(f"{path} contains no reads")
        rng = self._rng(path)
        if len(sequences) >= self.sample_reads:
            return rng.sample(sequences, self.sample_reads)
        return sequences + rng.choices(sequences, k=self.sample_reads - len(sequences))

    def _encode(self, path: Path) -> torch.Tensor:
        rows = []
        for sequence in self._sample_sequences(path):
            rows.append(
                [
                    self.vocab.get(sequence[offset : offset + 8], self.unknown)
                    for offset in range(0, 32, 8)
                ]
            )
        return torch.tensor(rows, dtype=torch.long)

    def __getitem__(self, index: int):
        path = self.paths[index]
        reads = self._encode(path)
        if self.labels is None:
            return reads, path.stem
        return reads, torch.tensor(self.labels[index], dtype=torch.float32), path.stem


def make_labeled_dataset(
    root: str | Path,
    vocab_path: str | Path,
    sample_reads: int,
    seed: int,
) -> ReadDataset:
    paths, labels = discover_labeled_samples(root)
    return ReadDataset(paths, vocab_path, labels, sample_reads, seed)


def split_indices(
    labels: Sequence[int], split: Sequence[float] = (0.8, 0.1, 0.1), seed: int = 1
) -> dict[str, np.ndarray]:
    if len(split) != 3 or any(part <= 0 for part in split):
        raise ValueError("split must contain three positive fractions")
    total = float(sum(split))
    train_fraction, val_fraction, test_fraction = [part / total for part in split]
    indices = np.arange(len(labels))
    labels_array = np.asarray(labels)
    train, remainder = train_test_split(
        indices,
        stratify=labels_array,
        test_size=1.0 - train_fraction,
        random_state=seed,
    )
    test_share = test_fraction / (val_fraction + test_fraction)
    val, test = train_test_split(
        remainder,
        stratify=labels_array[remainder],
        test_size=test_share,
        random_state=seed,
    )
    return {"train": train, "val": val, "test": test}


def clone_subset(dataset: ReadDataset, indices: Sequence[int]) -> ReadDataset:
    clone = object.__new__(ReadDataset)
    clone.paths = [dataset.paths[index] for index in indices]
    clone.labels = (
        [dataset.labels[index] for index in indices] if dataset.labels is not None else None
    )
    clone.vocab = dataset.vocab
    clone.unknown = dataset.unknown
    clone.sample_reads = dataset.sample_reads
    clone.seed = dataset.seed
    return clone


def collate_labeled(batch):
    reads = torch.stack([item[0] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    sample_ids = [item[2] for item in batch]
    return reads, labels, sample_ids


def collate_unlabeled(batch):
    reads = torch.stack([item[0] for item in batch])
    sample_ids = [item[1] for item in batch]
    return reads, sample_ids

