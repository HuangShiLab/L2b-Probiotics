from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader

from .checkpoints import load_model
from .data import (
    ReadDataset,
    clone_subset,
    collate_labeled,
    collate_unlabeled,
    discover_unlabeled_samples,
    load_vocab,
    make_labeled_dataset,
    split_indices,
)
from .features import GeneFeatureBank
from .models import build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VOCAB = PROJECT_ROOT / "assets" / "kmer" / "kmer_vocab.json"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "assets" / "checkpoints" / "hybrid_seed1.pt"
DEFAULT_FEATURES = PROJECT_ROOT / "assets" / "gene_features"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def parse_split(value: str) -> tuple[float, float, float]:
    try:
        parts = tuple(float(part) for part in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("split must be comma-separated numbers") from error
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError("split must contain three positive numbers")
    return parts


def add_data_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--reads-dir", required=True, type=Path)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--sample-reads", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or mps")


def add_gene_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gene-feature-dir", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument(
        "--prokka-dir",
        required=True,
        type=Path,
        help="Directory containing per-sample Prokka TSV or GFF files",
    )


def make_feature_bank(args, expected_dim: int | None = None) -> GeneFeatureBank:
    bank = GeneFeatureBank(args.gene_feature_dir, args.prokka_dir)
    if expected_dim is not None and bank.feature_dim != expected_dim:
        raise ValueError(
            f"Checkpoint expects {expected_dim} gene features, but "
            f"{args.gene_feature_dir} defines {bank.feature_dim}"
        )
    return bank


def infer_labeled(model, loader, bank, device):
    labels: list[float] = []
    probabilities: list[float] = []
    sample_ids: list[str] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for reads, batch_labels, batch_ids in loader:
            reads = reads.to(device, non_blocking=device.type == "cuda")
            if bank is None:
                logits = model(reads)
            else:
                logits = model(reads, bank.batch(batch_ids, device))
            probabilities.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(batch_labels.tolist())
            sample_ids.extend(batch_ids)
    return (
        np.asarray(labels, dtype=int),
        np.asarray(probabilities),
        sample_ids,
        time.perf_counter() - started,
    )


def metric_report(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predictions = (probabilities > 0.5).astype(int)
    report = {
        "n_samples": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }
    if len(np.unique(labels)) == 2:
        report["roc_auc"] = float(roc_auc_score(labels, probabilities))
    return report


def command_evaluate(args) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    vocab = load_vocab(args.vocab)
    if len(vocab) != metadata["vocab_size"]:
        raise ValueError(
            f"Checkpoint expects vocabulary size {metadata['vocab_size']}, got {len(vocab)}"
        )
    full_dataset = make_labeled_dataset(
        args.reads_dir, args.vocab, args.sample_reads, args.seed
    )
    indices = split_indices(full_dataset.labels, args.split, args.seed)
    test_dataset = clone_subset(full_dataset, indices["test"])
    loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_labeled,
    )
    bank = (
        make_feature_bank(args, metadata["feature_dim"]) if metadata["hybrid"] else None
    )
    labels, probabilities, sample_ids, elapsed = infer_labeled(model, loader, bank, device)
    report = metric_report(labels, probabilities)
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "device": str(device),
            "elapsed_seconds": elapsed,
            "seed": args.seed,
            "sample_reads": args.sample_reads,
            "split": list(args.split),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "metrics.json").open("w") as handle:
        json.dump(report, handle, indent=2)
    with (args.output_dir / "predictions.tsv").open("w") as handle:
        handle.write("sample_id\tlabel\tprobability\tprediction\n")
        for sample_id, label, probability in zip(sample_ids, labels, probabilities):
            handle.write(
                f"{sample_id}\t{label}\t{probability:.8f}\t{int(probability > 0.5)}\n"
            )
    print(json.dumps(report, indent=2))
    print(f"Wrote evaluation outputs to {args.output_dir}")


def command_predict(args) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    model, metadata = load_model(args.checkpoint, device)
    vocab = load_vocab(args.vocab)
    if len(vocab) != metadata["vocab_size"]:
        raise ValueError(
            f"Checkpoint expects vocabulary size {metadata['vocab_size']}, got {len(vocab)}"
        )
    dataset = ReadDataset(
        discover_unlabeled_samples(args.reads_dir),
        args.vocab,
        sample_reads=args.sample_reads,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_unlabeled,
    )
    bank = make_feature_bank(args, metadata["feature_dim"]) if metadata["hybrid"] else None
    rows = []
    with torch.inference_mode():
        for reads, sample_ids in loader:
            reads = reads.to(device, non_blocking=device.type == "cuda")
            if bank is None:
                logits = model(reads)
            else:
                logits = model(reads, bank.batch(sample_ids, device))
            probabilities = torch.sigmoid(logits).cpu().tolist()
            rows.extend(zip(sample_ids, probabilities))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write("sample_id\tprobability\tprediction\n")
        for sample_id, probability in rows:
            label = "probiotic" if probability > 0.5 else "non_probiotic"
            handle.write(f"{sample_id}\t{probability:.8f}\t{label}\n")
    print(f"Wrote {len(rows)} predictions to {args.output}")


def validation_score(model, loader, bank, device) -> float:
    model.eval()
    labels, probabilities, _, _ = infer_labeled(model, loader, bank, device)
    return float(balanced_accuracy_score(labels, probabilities > 0.5))


def command_train(args) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    if device.type == "mps":
        print("Warning: exact reproducibility is best on CPU or CUDA; MPS kernels can differ.")
    vocab = load_vocab(args.vocab)
    full_dataset = make_labeled_dataset(
        args.reads_dir, args.vocab, args.sample_reads, args.seed
    )
    indices = split_indices(full_dataset.labels, args.split, args.seed)
    train_dataset = clone_subset(full_dataset, indices["train"])
    val_dataset = clone_subset(full_dataset, indices["val"])
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=collate_labeled,
        drop_last=len(train_dataset) > args.batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_labeled,
    )
    bank = make_feature_bank(args) if args.hybrid else None
    feature_dim = bank.feature_dim if bank is not None else None
    model = build_model(len(vocab), args.read_dim, feature_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    else:  # PyTorch 2.2
        scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for reads, labels, sample_ids in train_loader:
            reads = reads.to(device, non_blocking=device.type == "cuda")
            labels = labels.to(device, non_blocking=device.type == "cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                if bank is None:
                    logits = model(reads)
                else:
                    logits = model(reads, bank.batch(sample_ids, device))
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.detach().item() * len(labels)
            seen += len(labels)

        score = validation_score(model, val_loader, bank, device)
        mean_loss = running_loss / max(seen, 1)
        print(
            f"epoch={epoch:03d}/{args.epochs} loss={mean_loss:.6f} "
            f"val_balanced_accuracy={score:.6f}"
        )
        if score > best_score:
            best_score = score
            checkpoint = {
                "model": model.state_dict(),
                "metadata": {
                    "seed": args.seed,
                    "read_dim": args.read_dim,
                    "feature_dim": feature_dim,
                    "sample_reads": args.sample_reads,
                    "split": list(args.split),
                    "best_val_balanced_accuracy": best_score,
                    "epoch": epoch,
                },
            }
            torch.save(checkpoint, args.output)
            print(f"Saved best checkpoint to {args.output}")


def command_inspect(args) -> None:
    _, metadata = load_model(args.checkpoint, torch.device("cpu"))
    print(json.dumps(metadata, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="l2bprobio",
        description="Train and evaluate the L2B-ProBio 2bRAD probiotic classifier.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    evaluate = subparsers.add_parser("evaluate", help="evaluate a checkpoint on a labeled test split")
    add_data_arguments(evaluate)
    add_gene_arguments(evaluate)
    evaluate.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    evaluate.add_argument("--split", type=parse_split, default=(0.8, 0.1, 0.1))
    evaluate.add_argument("--output-dir", type=Path, default=Path("results/evaluation"))
    evaluate.set_defaults(handler=command_evaluate)

    predict = subparsers.add_parser("predict", help="predict unlabeled 2bRAD samples")
    add_data_arguments(predict)
    add_gene_arguments(predict)
    predict.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    predict.add_argument("--output", type=Path, default=Path("results/predictions.tsv"))
    predict.set_defaults(handler=command_predict)

    train = subparsers.add_parser("train", help="train the CNN or hybrid CNN")
    add_data_arguments(train)
    train.add_argument("--hybrid", action="store_true", help="enable the Prokka gene-feature branch")
    train.add_argument("--gene-feature-dir", type=Path, default=DEFAULT_FEATURES)
    train.add_argument("--prokka-dir", type=Path)
    train.add_argument("--split", type=parse_split, default=(0.8, 0.1, 0.1))
    train.add_argument("--read-dim", type=int, default=128)
    train.add_argument("--epochs", type=int, default=200)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--output", type=Path, default=Path("checkpoints/hybrid_seed1.pt"))
    train.set_defaults(handler=command_train)

    inspect = subparsers.add_parser("inspect", help="show checkpoint architecture metadata")
    inspect.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    inspect.set_defaults(handler=command_inspect)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if hasattr(args, "sample_reads") and args.sample_reads < 16:
        raise SystemExit("--sample-reads must be at least 16 for the four pooling layers")
    if args.command == "train" and args.hybrid and args.prokka_dir is None:
        raise SystemExit("--prokka-dir is required with --hybrid")
    args.handler(args)

