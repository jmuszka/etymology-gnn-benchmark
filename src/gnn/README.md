# Relational GNN: run guide

A graph neural network that predicts the same three targets as the LSTM
baseline (fine mechanism, coarse mechanism, parent language) from the
relations between words instead of from spelling.

It uses the LSTM baseline's prepared data, so labels, the train/validation/test
split, the parent-language targets and every metric are identical. Prepare the
data first (see `src/lstm_baseline/README.md`).

## 1. Code layout

- `src/gnn/build_graph.py` (CLI: `uv run gnn-build`)
- `src/gnn/train_gnn.py` (CLI: `uv run gnn-train`)

## 2. Build the graph (once, about 20 seconds)

```bash
uv run gnn-build --csv Data/etymology.csv --out prepared/graph
```

Every CSV row that names a related term becomes one typed, directed edge from
the entry's word (the edge's *owner*) to the related term. Nodes are exact
`(language, term)` pairs, the same word identity the LSTM baseline uses. The
graph covers all languages: 3,686,601 words and 6,146,951 edges of 27 relation
types.

Outputs under `prepared/graph`:

| File | Contents |
| --- | --- |
| `graph.npz` | Edge arrays (`source`, `relation`, `target`) and each node's language id |
| `nodes.json` | Language names, relation names, and each node's term |
| `graph_summary.json` | Counts, dropped rows, edge policy, and source SHA-256 |

Group markers (rows with no related term), self-loops and exact duplicate
edges are dropped. Group nesting and positions are not encoded.

## 3. Train

```bash
uv run gnn-train --graph prepared/graph --data prepared/english --task parent --out runs/gnn_english_parent_seed42 --seed 42
```

Use `--task fine` or `--task coarse` for the mechanism tasks. An epoch takes
about a minute on a laptop CPU. The output directory must not already exist.

Outputs match the LSTM's (`metrics.json`, `history.csv`, `config.json`,
`best_model.pt`, `test_predictions.jsonl`), with two differences:

- The model's results are under `gnn_validation` and `gnn_test`.
- Test results include `slices`, and each prediction records `relatives`
  (how many other entries cite the word).

## What the model may see

A word's labels are read off its own Wiktionary entry: the fine label is the
type of its own relations, and the parent language is the language at the
other end. Showing the model those relations would hand it the answer, so:

> When a word is the prediction target, every edge owned by its own entry is
> hidden. Every edge owned by any other entry is visible.

This holds in training, validation and testing. What remains for a target is
its *relatives*:

- **Children**: entries that name the word as a parent or component.
- **Recorded siblings**: entries that list the word as a cognate, doublet or
  related term.

The second hop adds those relatives' own entries and the entries that cite
them. Siblings defined by a shared parent are not used, since finding them
requires knowing the parent.

Spellings never enter the model. Node features are the word's language, a flag
marking the target, and the number of citations from other entries.

`--hide-heldout-entries` is a stricter variant that removes every validation
and test entry from the graph, not only the target's own.

## Coverage

Most words have no relatives. In the English test set, 6,406 of 36,326 words
(18%) are cited by another entry. For the parent task, 25% of borrowed words
and 70% of inherited words have at least one relative.

A target with no relatives has no edges, so the model can only return its
prior (for the parent task, the most likely parent given the mechanism). Read
the `with_relatives` and `without_relatives` slices separately: the first
shows what relational structure contributes, the second cannot beat a prior.

## Model and training configuration

Two R-GCN layers (Schlichtkrull et al. 2018) from PyTorch Geometric, hidden
size 128, basis decomposition with 8 bases. Each relation has a forward
message type (related term to owner) and an inverse one (owner to related
term), 54 in total. The target's final state goes through a linear layer with
sigmoid outputs. For the parent task a mechanism embedding is added to the
target's input features, mirroring the LSTM's mechanism token.

Neighbourhoods are sampled: at most 25 entries citing the target, then at most
10 edges per direction for each of those. Evaluation uses a fixed sample per
word, so validation and test results are deterministic.

Shared with the LSTM: binary cross-entropy, Adam at 0.001, up to 50 epochs,
patience 5 on validation loss, best-validation-loss checkpoint, threshold 0.5
with no fallback, exact-match accuracy.

Different from the LSTM: batch size 256 and one CPU thread, both chosen for
speed. The model runs many small operations and is slower with more threads.

## Sources

- R-GCN: https://arxiv.org/abs/1703.06103
- `RGCNConv`: https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.RGCNConv.html
