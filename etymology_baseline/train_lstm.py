"""Train the Section-3-style character LSTM on prepared etymology data.

Defaults not specified by the paper are recorded in config.json. This program
does not claim to reproduce the authors' original data or exact accuracy rule.
"""

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import platform
import random
import time

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

from prepare_etymology import COARSE_LABELS, FINE_LABELS, file_sha256, word_id, write_json


def load_words(directory):
    manifest = json.loads((directory / "preprocess_summary.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("format_version") != 1:
        raise ValueError("Prepared data is incomplete or uses an unsupported format.")
    parts, seen = {}, {}
    for split in ["train", "validation", "test"]:
        rows = []
        with (directory / f"{split}.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                key = word_id(row["language"], row["word"])
                if key != row["id"] or key in seen:
                    raise ValueError(f"Invalid or repeated word across partitions: {row['word']!r}")
                if not row["word"] or not row["fine_labels"]:
                    raise ValueError("A prepared word has no spelling or fine labels.")
                if set(row["fine_labels"]) - set(FINE_LABELS):
                    raise ValueError("Unknown fine label in prepared data.")
                seen[key] = split
                rows.append(row)
        expected = manifest["splits"][split]["words"]
        if len(rows) != expected:
            raise ValueError(f"{split} has {len(rows)} words, expected {expected}.")
        parts[split] = rows
    return parts, manifest


def task_samples(word_rows, task, target_names):
    samples = []
    for row in word_rows:
        if task == "parent":
            annotations = row["parent_languages_by_mechanism"].items()
        else:
            annotations = [(None, row[f"{task}_labels"])]
        for mechanism, names in annotations:
            if not names:
                continue
            mapped = set(names)
            if task == "parent":
                mapped = {name if name in target_names[:-1] else "<OTHER>" for name in names}
            samples.append({
                "id": row["id"], "language": row["language"], "word": row["word"],
                "mechanism": mechanism,
                "target": [int(name in mapped) for name in target_names],
            })
    return samples


def input_tokens(sample, multilingual):
    tokens = [f"lang:{sample['language']}"] if multilingual else []
    tokens.extend(f"char:{character}" for character in sample["word"])
    if sample["mechanism"] is not None:
        tokens.append(f"mechanism:{sample['mechanism']}")
    return tokens


class WordDataset(Dataset):
    def __init__(self, samples, vocabulary, multilingual):
        self.samples, self.vocabulary = samples, vocabulary
        self.multilingual = multilingual

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        ids = [self.vocabulary.get(token, 1) for token in input_tokens(sample, self.multilingual)]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(sample["target"], dtype=torch.float32), sample


def collate(batch):
    sequences, targets, samples = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    return pad_sequence(sequences, batch_first=True, padding_value=0), lengths, torch.stack(targets), samples


class CharacterLSTM(nn.Module):
    def __init__(self, vocabulary_size, output_size):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=False)
        self.output = nn.Linear(128, output_size)

    def forward(self, token_ids, lengths):
        embedded = self.embedding(token_ids)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        return self.output(hidden[-1])  # Raw logits; BCEWithLogitsLoss includes sigmoid.


class Metrics:
    def __init__(self, target_names):
        self.names = target_names
        self.n = self.correct = self.empty = self.label_correct = 0
        self.tp = torch.zeros(len(target_names), dtype=torch.long)
        self.fp = self.tp.clone()
        self.fn = self.tp.clone()
        self.support = self.tp.clone()

    def update(self, truth, predicted):
        truth, predicted = truth.detach().cpu().bool(), predicted.detach().cpu().bool()
        self.n += len(truth)
        self.correct += int((truth == predicted).all(dim=1).sum())
        self.label_correct += int((truth == predicted).sum())
        self.empty += int((~predicted.any(dim=1)).sum())
        self.tp += (truth & predicted).sum(dim=0)
        self.fp += (~truth & predicted).sum(dim=0)
        self.fn += (truth & ~predicted).sum(dim=0)
        self.support += truth.sum(dim=0)

    def result(self):
        precision = self.tp.double() / (self.tp + self.fp).clamp_min(1)
        recall = self.tp.double() / (self.tp + self.fn).clamp_min(1)
        f1 = 2 * self.tp.double() / (2 * self.tp + self.fp + self.fn).clamp_min(1)
        denominator = int((2 * self.tp + self.fp + self.fn).sum())
        return {
            "examples": self.n,
            "exact_match_accuracy": self.correct / max(1, self.n),
            "label_accuracy": self.label_correct / max(1, self.n * len(self.names)),
            "micro_f1": 2 * int(self.tp.sum()) / max(1, denominator),
            "macro_f1": float(f1.mean()),
            "empty_prediction_rate": self.empty / max(1, self.n),
            "per_label": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                                 "f1": float(f1[i]), "support": int(self.support[i])}
                          for i, name in enumerate(self.names)},
        }


def evaluate(model, loader, names, device, threshold, prediction_path=None):
    model.eval()
    metrics, by_language, total_loss = Metrics(names), {}, 0.0
    prediction_stream = prediction_path.open("w", encoding="utf-8") if prediction_path else None
    try:
        with torch.no_grad():
            for ids, lengths, targets, samples in loader:
                targets = targets.to(device)
                logits = model(ids.to(device), lengths)
                total_loss += float(nn.functional.binary_cross_entropy_with_logits(logits, targets)) * len(samples)
                probabilities = logits.sigmoid().cpu()
                predictions = probabilities >= threshold
                metrics.update(targets, predictions)
                for language in {sample["language"] for sample in samples}:
                    indices = [i for i, sample in enumerate(samples) if sample["language"] == language]
                    by_language.setdefault(language, Metrics(names)).update(targets[indices], predictions[indices])
                if prediction_stream:
                    for sample, probs, pred in zip(samples, probabilities.tolist(), predictions.tolist()):
                        record = {k: sample[k] for k in ["id", "language", "word", "mechanism"]}
                        record.update({
                            "gold_labels": [name for name, bit in zip(names, sample["target"]) if bit],
                            "predicted_labels": [name for name, bit in zip(names, pred) if bit],
                            "probabilities": dict(zip(names, probs)),
                        })
                        prediction_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if prediction_stream:
            prediction_stream.close()
    result = metrics.result()
    result["bce_loss"] = total_loss / max(1, metrics.n)
    result["per_language"] = {lang: value.result() for lang, value in sorted(by_language.items())}
    return result


def majority_baseline(train, test, names):
    counts, global_counts = {}, torch.zeros(len(names), dtype=torch.long)
    for sample in train:
        target = torch.tensor(sample["target"], dtype=torch.long)
        counts.setdefault(sample["language"], torch.zeros_like(target)).add_(target)
        global_counts += target
    metrics, by_language = Metrics(names), {}
    for sample in test:
        prediction = torch.zeros((1, len(names)), dtype=torch.bool)
        prediction[0, int(counts.get(sample["language"], global_counts).argmax())] = True
        truth = torch.tensor([sample["target"]], dtype=torch.bool)
        metrics.update(truth, prediction)
        by_language.setdefault(sample["language"], Metrics(names)).update(truth, prediction)
    result = metrics.result()
    result["per_language"] = {lang: value.result() for lang, value in sorted(by_language.items())}
    result["most_frequent_label_by_language"] = {
        lang: names[int(value.argmax())] for lang, value in sorted(counts.items())}
    return result


def train(args):
    parts, manifest = load_words(args.data)
    multilingual = len(manifest["eligible_words_by_language"]) > 1
    if args.task == "parent":
        frequencies = Counter()
        for row in parts["train"]:
            for parents in row["parent_languages_by_mechanism"].values():
                frequencies.update(parents)
        parent_names = sorted(frequencies, key=lambda name: (-frequencies[name], name))[:5]
        if not parent_names:
            raise ValueError("No parent-language training annotations are available.")
        names = parent_names + ["<OTHER>"]
    else:
        names = FINE_LABELS if args.task == "fine" else COARSE_LABELS
    samples = {split: task_samples(rows, args.task, names) for split, rows in parts.items()}
    del parts
    for split, values in samples.items():
        if not values:
            raise ValueError(f"No {args.task} examples in {split}. Use more data.")
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}. Choose a new --out directory.")
    args.out.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device_name == "auto":
        device_name = "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(device_name)
    tokens = sorted({token for sample in samples["train"] for token in input_tokens(sample, multilingual)})
    vocabulary = {"<PAD>": 0, "<UNK>": 1}
    vocabulary.update({token: i + 2 for i, token in enumerate(tokens)})
    config = {
        "task": args.task, "embedding_dim": 128, "hidden_dim": 128,
        "num_layers": 1, "bidirectional": False, "dropout": 0.0,
        "optimizer": "Adam", "learning_rate": 0.001, "loss": "BCEWithLogitsLoss",
        "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "seed": args.seed, "threshold": args.threshold, "threads": args.threads,
        "device": device_name, "multilingual": multilingual, "target_names": names,
        "vocabulary_size": len(vocabulary), "sequence_truncation": None,
        "sample_counts": {split: len(rows) for split, rows in samples.items()},
        "source_sha256": manifest["source_sha256"],
        "prepared_manifest_sha256": file_sha256(args.data / "preprocess_summary.json"),
        "split_seed": manifest["split_seed"], "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "unspecified_paper_choices": ["layers/direction", "dropout", "batch size", "epochs/patience",
                                     "seed", "threshold", "exact-match/F1 metrics", "text processing"],
        "parent_vocabulary_policy": "Top 5 from training annotations within this run; remaining languages OTHER.",
        "accuracy_definition": "Exact equality of predicted and gold label sets at the recorded threshold.",
        "comparability": manifest["comparability"],
    }
    write_json(args.out / "config.json", config)
    write_json(args.out / "vocabulary.json", vocabulary)
    generator = torch.Generator().manual_seed(args.seed)
    loaders = {
        split: DataLoader(WordDataset(rows, vocabulary, multilingual), batch_size=args.batch_size,
                          shuffle=(split == "train"), collate_fn=collate, num_workers=0,
                          generator=generator if split == "train" else None)
        for split, rows in samples.items()
    }
    model = CharacterLSTM(len(vocabulary), len(names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, stale = math.inf, 0, 0
    history = []
    print(f"Task: {args.task}; device: {device}; samples: {config['sample_counts']}", flush=True)
    for i, name in enumerate(names):
        support = sum(sample["target"][i] for sample in samples["train"])
        if not support:
            print(f"Training support for {name}: 0 (check dataset coverage).", flush=True)
    started = time.perf_counter()
    with (args.out / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["epoch", "train_bce", "validation_bce", "validation_exact_match", "validation_micro_f1", "seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            loss_sum = count = 0
            for batch_index, (ids, lengths, targets, _) in enumerate(loaders["train"], 1):
                optimizer.zero_grad(set_to_none=True)
                logits = model(ids.to(device), lengths)
                loss = criterion(logits, targets.to(device))
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became non-finite.")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(targets)
                count += len(targets)
                if batch_index % 200 == 0:
                    print(f"Epoch {epoch}, batch {batch_index}/{len(loaders['train'])}, BCE {loss_sum/count:.5f}", flush=True)
            validation = evaluate(model, loaders["validation"], names, device, args.threshold)
            row = {"epoch": epoch, "train_bce": loss_sum / count,
                   "validation_bce": validation["bce_loss"],
                   "validation_exact_match": validation["exact_match_accuracy"],
                   "validation_micro_f1": validation["micro_f1"],
                   "seconds": time.perf_counter() - epoch_start}
            history.append(row)
            writer.writerow(row)
            stream.flush()
            print(f"Epoch {epoch}: train BCE={row['train_bce']:.5f}; validation BCE={row['validation_bce']:.5f}; "
                  f"validation exact-match={row['validation_exact_match']:.4f}", flush=True)
            if validation["bce_loss"] < best_loss:
                best_loss, best_epoch, stale = validation["bce_loss"], epoch, 0
                torch.save({"model_state": model.state_dict(), "config": config, "vocabulary": vocabulary},
                           args.out / "best_model.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping: validation loss has stopped improving.", flush=True)
                    break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(args.out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, loaders["validation"], names, device, args.threshold)
    test = evaluate(model, loaders["test"], names, device, args.threshold, args.out / "test_predictions.jsonl")
    baseline = majority_baseline(samples["train"], samples["test"], names)
    report = {
        "status": "complete", "config": config, "best_epoch": best_epoch,
        "epochs_run": len(history), "best_validation_bce": best_loss,
        "training_and_validation_seconds": training_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "lstm_validation": validation, "lstm_test": test, "majority_test": baseline,
        "notes": ["Test examples are scored only after selecting the best checkpoint.",
                  "Exact-match and F1 definitions are explicit choices; paper accuracy details are incomplete.",
                  "Macro F1 includes zero-support labels as zero.",
                  "No test-set threshold tuning or empty-prediction argmax fallback is applied."],
    }
    write_json(args.out / "metrics.json", report)
    print(f"Best epoch: {best_epoch}")
    print(f"Test exact-match: LSTM={test['exact_match_accuracy']:.4f}; majority={baseline['exact_match_accuracy']:.4f}")
    print(f"Test micro F1: {test['micro_f1']:.4f}")
    print(f"Results: {args.out.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", choices=["fine", "coarse", "parent"], default="fine")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.patience, args.threads) < 1 or not 0 < args.threshold < 1:
        parser.error("Counts must be positive and threshold must be between 0 and 1.")
    try:
        train(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
