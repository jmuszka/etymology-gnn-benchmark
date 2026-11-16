"""Train a relational GNN (R-GCN) on the word-relation graph.

The model sees relations only: node languages, relation types and graph
structure. Spellings never enter the model. When a word is the prediction
target, every edge from its own entry is hidden, so the model works from the
word's children and recorded siblings (edges owned by other entries).
Labels, split, targets and metrics are the LSTM baseline's.
"""

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import platform
import random
import time

import numpy as np
import torch
from torch import nn
from torch_geometric.nn import RGCNConv

from lstm_baseline.prepare_etymology import COARSE_LABELS, FINE_LABELS, file_sha256, write_json
from lstm_baseline.train_lstm import Metrics, load_words, majority_baseline, task_samples


COMPOSITIONAL = {"affix", "compound", "prefix", "suffix"}


class Graph:
    """Typed edges in two sorted layouts: by owner (an entry's own rows) and by cited term."""

    def __init__(self, source, relation, target, node_language, num_relations):
        self.num_nodes, self.num_relations = len(node_language), num_relations
        self.node_language = node_language
        by_owner = np.lexsort((target, source))
        self.out_ptr = self._pointers(source[by_owner])
        self.out_target, self.out_relation = target[by_owner], relation[by_owner]
        by_cited = np.lexsort((source, target))
        self.in_ptr = self._pointers(target[by_cited])
        self.in_source, self.in_relation = source[by_cited], relation[by_cited]
        self.in_degree = np.diff(self.in_ptr)

    def _pointers(self, sorted_ids):
        return np.concatenate([[0], np.cumsum(np.bincount(sorted_ids, minlength=self.num_nodes))])


def load_graph(directory, languages, hidden_owners=None):
    """Return the graph and a (language, term) -> node lookup for the given languages."""
    summary = json.loads((directory / "graph_summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete" or summary.get("format_version") != 1:
        raise ValueError("Graph is incomplete or uses an unsupported format.")
    arrays = np.load(directory / "graph.npz")
    names = json.loads((directory / "nodes.json").read_text(encoding="utf-8"))
    source, relation, target = arrays["source"], arrays["relation"], arrays["target"]
    node_language = arrays["node_language"]
    wanted = [i for i, name in enumerate(names["languages"]) if name in languages]
    lookup = {(names["languages"][node_language[i]], names["terms"][i]): int(i)
              for i in np.flatnonzero(np.isin(node_language, wanted))}
    if hidden_owners is not None:
        hidden = np.fromiter((lookup[key] for key in hidden_owners), dtype=np.int64)
        keep = ~np.isin(source, hidden)
        source, relation, target = source[keep], relation[keep], target[keep]
    graph = Graph(source, relation, target, node_language, len(names["relations"]))
    return graph, lookup, names, summary


def ego_subgraph(graph, center, rng, fanout):
    """Two-hop neighbourhood of `center` that never uses an edge owned by `center`.

    Hop 1 is the entries that cite the centre. Hop 2 is those entries' own
    rows and the entries that cite them. Messages flow from the related term
    to the owner with the relation's id, and from the owner to the related
    term with the id shifted by the number of relations.
    """
    inverse = graph.num_relations
    local, order = {center: 0}, [center]
    sources, targets, kinds = [], [], []

    def local_id(node):
        index = local.get(node)
        if index is None:
            index = local[node] = len(order)
            order.append(node)
        return index

    def pick(low, high, limit):
        if high - low <= limit:
            return range(low, high)
        return (low + rng.choice(high - low, limit, replace=False)).tolist()

    hop_one = []
    for j in pick(graph.in_ptr[center], graph.in_ptr[center + 1], fanout[0]):
        citing = int(graph.in_source[j])
        if citing not in local:
            hop_one.append(citing)
        sources.append(local_id(citing))
        targets.append(0)
        kinds.append(int(graph.in_relation[j]) + inverse)
    for owner in hop_one:
        owner_local = local[owner]
        for j in pick(graph.out_ptr[owner], graph.out_ptr[owner + 1], fanout[1]):
            sources.append(local_id(int(graph.out_target[j])))
            targets.append(owner_local)
            kinds.append(int(graph.out_relation[j]))
        for j in pick(graph.in_ptr[owner], graph.in_ptr[owner + 1], fanout[1]):
            citing = int(graph.in_source[j])
            if citing == center:
                continue  # An edge from the centre's own entry: hidden.
            sources.append(local_id(citing))
            targets.append(owner_local)
            kinds.append(int(graph.in_relation[j]) + inverse)
    # Citation counts must not reveal the centre's own rows either.
    own = Counter(graph.out_target[graph.out_ptr[center]:graph.out_ptr[center + 1]].tolist()) if hop_one else {}
    degrees = [int(graph.in_degree[node]) - own.get(node, 0) for node in order]
    return order, sources, targets, kinds, degrees


def collate(samples, graph, fanout, seed, epoch):
    """Join the ego-subgraphs of a batch as disjoint graphs. epoch=None gives fixed evaluation samples."""
    nodes, degrees, sources, targets, kinds, centers = [], [], [], [], [], []
    for sample in samples:
        key = [seed, sample["node"]] if epoch is None else [seed, epoch, sample["node"]]
        order, s, t, k, d = ego_subgraph(graph, sample["node"], np.random.default_rng(key), fanout)
        offset = len(nodes)
        centers.append(offset)
        nodes.extend(order)
        degrees.extend(d)
        sources.extend(i + offset for i in s)
        targets.extend(i + offset for i in t)
        kinds.extend(k)
    is_center = torch.zeros(len(nodes), dtype=torch.long)
    is_center[centers] = 1
    return {
        "language": torch.from_numpy(graph.node_language[nodes]).long(),
        "is_center": is_center,
        "degree": torch.tensor(degrees, dtype=torch.float32).log1p().unsqueeze(1),
        "edge_index": torch.tensor([sources, targets], dtype=torch.long).reshape(2, -1),
        "edge_type": torch.tensor(kinds, dtype=torch.long),
        "center": torch.tensor(centers, dtype=torch.long),
        "mechanism": torch.tensor([sample["mechanism_id"] for sample in samples], dtype=torch.long),
        "target": torch.tensor([sample["target"] for sample in samples], dtype=torch.float32),
    }


class RelationalGNN(nn.Module):
    def __init__(self, num_languages, num_relations, output_size, hidden, bases, use_mechanism):
        super().__init__()
        self.language = nn.Embedding(num_languages, hidden)
        self.is_center = nn.Embedding(2, hidden)
        self.degree = nn.Linear(1, hidden)
        # Each relation has a forward and an inverse message type.
        self.conv1 = RGCNConv(hidden, hidden, 2 * num_relations, num_bases=bases)
        self.conv2 = RGCNConv(hidden, hidden, 2 * num_relations, num_bases=bases)
        self.mechanism = nn.Embedding(len(FINE_LABELS), hidden) if use_mechanism else None
        self.output = nn.Linear(hidden, output_size)

    def forward(self, batch):
        x = self.language(batch["language"]) + self.is_center(batch["is_center"]) + self.degree(batch["degree"])
        if self.mechanism is not None:
            # The known mechanism is a feature of the target node, like the LSTM's mechanism token,
            # so the layers can weigh the relatives differently for each mechanism.
            x = x.index_add(0, batch["center"], self.mechanism(batch["mechanism"]))
        x = self.conv1(x, batch["edge_index"], batch["edge_type"]).relu()
        x = self.conv2(x, batch["edge_index"], batch["edge_type"]).relu()
        return self.output(x[batch["center"]])  # Raw logits; BCEWithLogitsLoss includes sigmoid.


def slices_of(sample):
    """Names of the result slices a test example belongs to."""
    relatives = "with_relatives" if sample["relatives"] else "without_relatives"
    names = [relatives]
    if sample["mechanism"] is not None:
        group = "compositional" if sample["mechanism"] in COMPOSITIONAL else "borrowing_inherited"
        names += [group, f"{group}_{relatives}", sample["mechanism"]]
    return names


def to_device(batch, device):
    return {key: value.to(device) for key, value in batch.items()}


def evaluate(model, batches, names, device, threshold, prediction_path=None):
    """`batches` is a list of (samples, collated batch) pairs."""
    model.eval()
    metrics, sliced, total_loss = Metrics(names), {}, 0.0
    stream = prediction_path.open("w", encoding="utf-8") if prediction_path else None
    try:
        with torch.no_grad():
            for samples, batch in batches:
                batch = to_device(batch, device)
                logits = model(batch)
                total_loss += float(nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])) * len(samples)
                probabilities = logits.sigmoid().cpu()
                predictions = probabilities >= threshold
                truth = batch["target"].cpu()
                metrics.update(truth, predictions)
                membership = {}
                for i, sample in enumerate(samples):
                    for name in slices_of(sample):
                        membership.setdefault(name, []).append(i)
                for name, indices in membership.items():
                    sliced.setdefault(name, Metrics(names)).update(truth[indices], predictions[indices])
                if stream:
                    for sample, probs, pred in zip(samples, probabilities.tolist(), predictions.tolist()):
                        record = {k: sample[k] for k in ["id", "language", "word", "mechanism", "relatives"]}
                        record.update({
                            "gold_labels": [name for name, bit in zip(names, sample["target"]) if bit],
                            "predicted_labels": [name for name, bit in zip(names, pred) if bit],
                            "probabilities": dict(zip(names, probs)),
                        })
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        if stream:
            stream.close()
    result = metrics.result()
    result["bce_loss"] = total_loss / max(1, metrics.n)
    result["slices"] = {name: value.result() for name, value in sorted(sliced.items())}
    return result


def train(args):
    parts, manifest = load_words(args.data)
    languages = set(manifest["eligible_words_by_language"])
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
    for split, values in samples.items():
        if not values:
            raise ValueError(f"No {args.task} examples in {split}. Use more data.")
    hidden_owners = None
    if args.hide_heldout_entries:
        hidden_owners = [(row["language"], row["word"]) for split in ["validation", "test"] for row in parts[split]]
    del parts
    graph, lookup, graph_names, graph_summary = load_graph(args.graph, languages, hidden_owners)
    if graph_summary["source_sha256"] != manifest["source_sha256"]:
        raise ValueError("The graph and the prepared data come from different source files.")
    for values in samples.values():
        for sample in values:
            node = lookup.get((sample["language"], sample["word"]))
            if node is None:
                raise ValueError(f"Word missing from the graph: {sample['word']!r}")
            sample["node"] = node
            sample["relatives"] = int(graph.in_degree[node])
            sample["mechanism_id"] = FINE_LABELS.index(sample["mechanism"]) if sample["mechanism"] else 0
    del lookup
    if args.out.exists():
        raise ValueError(f"Output already exists: {args.out}. Choose a new --out directory.")
    args.out.mkdir(parents=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu.")
    device = torch.device(args.device)
    fanout = (args.fanout_citing, args.fanout_second)
    config = {
        "model": "RGCN", "task": args.task, "hidden_dim": args.hidden, "num_layers": 2,
        "num_bases": args.bases, "relation_types": 2 * graph.num_relations,
        "node_features": ["language", "is_target", "log1p(citations from other entries)"],
        "uses_spelling": False, "fanout": list(fanout),
        "target_masking": "Every edge owned by the target's own entry is hidden, in training and evaluation.",
        "hide_heldout_entries": args.hide_heldout_entries,
        "optimizer": "Adam", "learning_rate": 0.001, "loss": "BCEWithLogitsLoss",
        "batch_size": args.batch_size, "max_epochs": args.epochs, "patience": args.patience,
        "seed": args.seed, "threshold": args.threshold, "threads": args.threads, "device": args.device,
        "target_names": names, "graph_nodes": graph.num_nodes, "graph_edges": len(graph.in_source),
        "sample_counts": {split: len(rows) for split, rows in samples.items()},
        "samples_with_relatives": {split: sum(bool(s["relatives"]) for s in rows) for split, rows in samples.items()},
        "source_sha256": manifest["source_sha256"],
        "prepared_manifest_sha256": file_sha256(args.data / "preprocess_summary.json"),
        "graph_summary_sha256": file_sha256(args.graph / "graph_summary.json"),
        "split_seed": manifest["split_seed"], "torch_version": str(torch.__version__),
        "python_version": platform.python_version(), "platform": platform.platform(),
        "accuracy_definition": "Exact equality of predicted and gold label sets at the recorded threshold.",
        "comparability": manifest["comparability"],
    }
    write_json(args.out / "config.json", config)

    def batches(rows, epoch):
        for start in range(0, len(rows), args.batch_size):
            chunk = rows[start:start + args.batch_size]
            yield chunk, collate(chunk, graph, fanout, args.seed, epoch)

    fixed = {split: list(batches(samples[split], None)) for split in ["validation", "test"]}
    model = RelationalGNN(len(graph_names["languages"]), graph.num_relations, len(names),
                          args.hidden, args.bases, args.task == "parent").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    generator = torch.Generator().manual_seed(args.seed)
    best_loss, best_epoch, stale = math.inf, 0, 0
    history = []
    print(f"Task: {args.task}; device: {device}; samples: {config['sample_counts']}; "
          f"with relatives: {config['samples_with_relatives']}", flush=True)
    started = time.perf_counter()
    with (args.out / "history.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["epoch", "train_bce", "validation_bce", "validation_exact_match", "validation_micro_f1", "seconds"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            model.train()
            order = torch.randperm(len(samples["train"]), generator=generator).tolist()
            shuffled = [samples["train"][i] for i in order]
            loss_sum = count = 0
            total = math.ceil(len(shuffled) / args.batch_size)
            for batch_index, (chunk, batch) in enumerate(batches(shuffled, epoch), 1):
                batch = to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(model(batch), batch["target"])
                if not torch.isfinite(loss):
                    raise ValueError("Training loss became non-finite.")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(chunk)
                count += len(chunk)
                if batch_index % 500 == 0:
                    print(f"Epoch {epoch}, batch {batch_index}/{total}, BCE {loss_sum/count:.5f}", flush=True)
            validation = evaluate(model, fixed["validation"], names, device, args.threshold)
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
                torch.save({"model_state": model.state_dict(), "config": config}, args.out / "best_model.pt")
            else:
                stale += 1
                if stale >= args.patience:
                    print("Early stopping: validation loss has stopped improving.", flush=True)
                    break
    training_seconds = time.perf_counter() - started
    checkpoint = torch.load(args.out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    validation = evaluate(model, fixed["validation"], names, device, args.threshold)
    test = evaluate(model, fixed["test"], names, device, args.threshold, args.out / "test_predictions.jsonl")
    baseline = majority_baseline(samples["train"], samples["test"], names)
    report = {
        "status": "complete", "config": config, "best_epoch": best_epoch,
        "epochs_run": len(history), "best_validation_bce": best_loss,
        "training_and_validation_seconds": training_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "gnn_validation": validation, "gnn_test": test, "majority_test": baseline,
        "notes": ["Test examples are scored only after selecting the best checkpoint.",
                  "A target with no relatives has no edges, so its prediction is the model's prior.",
                  "Slices: relatives are citations from other entries; parent task also splits by mechanism.",
                  "No test-set threshold tuning or empty-prediction argmax fallback is applied."],
    }
    write_json(args.out / "metrics.json", report)
    print(f"Best epoch: {best_epoch}")
    print(f"Test exact-match: GNN={test['exact_match_accuracy']:.4f}; majority={baseline['exact_match_accuracy']:.4f}")
    for name, values in test["slices"].items():
        print(f"  {name}: n={values['examples']}; exact-match={values['exact_match_accuracy']:.4f}")
    print(f"Results: {args.out.resolve()}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", required=True, type=Path, help="Directory written by gnn-build.")
    parser.add_argument("--data", required=True, type=Path, help="Directory written by lstm-prepare.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--task", choices=["fine", "coarse", "parent"], default="fine")
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--bases", type=int, default=8, help="R-GCN basis matrices shared across relations.")
    parser.add_argument("--fanout-citing", type=int, default=25, help="Max sampled entries citing the target.")
    parser.add_argument("--fanout-second", type=int, default=10, help="Max sampled edges per direction at hop two.")
    parser.add_argument("--hide-heldout-entries", action="store_true",
                        help="Remove every validation/test entry from the graph, not only the target's own.")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--threads", type=int, default=1,
                        help="CPU threads; the model's many small operations run fastest on one.")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    args = parser.parse_args()
    counts = [args.hidden, args.bases, args.fanout_citing, args.fanout_second,
              args.batch_size, args.epochs, args.patience, args.threads]
    if min(counts) < 1 or not 0 < args.threshold < 1:
        parser.error("Counts must be positive and threshold must be between 0 and 1.")
    try:
        train(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(1, f"Training failed: {error}\n")


if __name__ == "__main__":
    main()
