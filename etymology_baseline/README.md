# Etymology LSTM baseline: run guide

This kit implements the baseline milestone for Wu and Yarowsky (2020),
*Computational Etymology and Word Emergence*, Section 3.

It includes preparation and training for fine mechanism, coarse mechanism,
and parent-language prediction. Begin with English fine mechanism.
The raw CSV stays on your computer.

## 1. Put the files in your existing Project folder

Extract this archive into the VS Code Project folder containing `Project.py`.
The result should include these paths:

- `Project.py` (your existing preview script)
- `Data/etymology.csv` (your full dataset)
- `etymology_baseline/prepare_etymology.py`
- `etymology_baseline/train_lstm.py`
- `etymology_baseline/test_pipeline.py`
- `etymology_baseline/requirements.txt`

Open the Project folder in VS Code. Run all commands below in its Terminal,
not inside a `.py` file or the Python interactive prompt.

## 2. Activate your environment and install PyTorch

Your existing `.venv` can be used. In Terminal:

```bash
source .venv/bin/activate
python -m pip install -r etymology_baseline/requirements.txt
python -c "import torch; print(torch.__version__); print(torch.rand(2, 2))"
```

The final command should print a version and a tensor. Record the installed
version when sharing results. The preprocessor uses only Python's standard
library; it does not require pandas or scikit-learn.

## 3. Run the focused software checks

```bash
python etymology_baseline/test_pipeline.py
```

These check hierarchy handling, duplicate labels, source-row-order invariance,
split overlap, Unicode input, padding, hand-calculated metrics, and whether
the LSTM can learn a small controlled problem. Expect the unittest summary `OK`.
If PyTorch is unavailable, neural checks are skipped; install it before training.

## 4. Prepare the FULL English dataset

```bash
python etymology_baseline/prepare_etymology.py --csv Data/etymology.csv --language English --out prepared/english --seed 42
```

Use `etymology.csv`, not `data_preview_1000.csv`, for the experiment.
The script streams the CSV into temporary SQLite storage, groups records by
exact `(language, term)`, constructs labels, and writes a stable word split.
The temporary database is removed after processing.

Outputs under `prepared/english`:

| File | Contents |
| --- | --- |
| `train.jsonl` | Training word examples |
| `validation.jsonl` | Validation word examples |
| `test.jsonl` | Held-out word examples |
| `split.csv` | Word-to-partition assignment, shared across all three tasks |
| `preprocess_summary.json` | Counts, exclusions, label policy, seed, and source SHA-256 |

Check `preprocess_summary.json` before training. All three partitions must
contain examples, and the six fine labels should have adequate training support.
The 1,000-row preview is suitable only for software verification.

The target split is 80/10/10 within each source language, using a seeded hash
ordering. Counts are rounded. Languages with fewer than three eligible words
are assigned to training; otherwise validation and test are reserved.
This is our explicit split implementation, not a recovered original split.

Existing output directories are protected against replacement. To repeat a
preparation, choose a new path such as `--out prepared/english_v2`.

## 5. Run one epoch to check the training pipeline

```bash
python etymology_baseline/train_lstm.py --data prepared/english --task fine --out runs/english_fine_smoke --epochs 1 --device cpu
```

This verifies the full pipeline on your prepared data. It is not the final
baseline result. The trainer prints batch progress and epoch losses.

## 6. Train the English fine-mechanism baseline

```bash
python etymology_baseline/train_lstm.py --data prepared/english --task fine --out runs/english_fine_seed42 --epochs 50 --patience 5 --seed 42 --device cpu
```

Training stops after five consecutive epochs without a lower validation loss,
or at the epoch limit. Test evaluation uses the best saved checkpoint.
The default is CPU, which also works on a Mac. On a machine with a compatible
NVIDIA installation, `--device cuda` can be used.

Outputs under `runs/english_fine_seed42`:

| File | Contents |
| --- | --- |
| `best_model.pt` | Best model weights, vocabulary, configuration |
| `config.json` | Hyperparameters, data signature, software versions |
| `vocabulary.json` | Training-derived character/context token IDs |
| `history.csv` | Training and validation losses by epoch |
| `metrics.json` | LSTM and majority-baseline evaluation, including per-language results |
| `test_predictions.jsonl` | Gold labels, predicted labels, probabilities for each test example |

Training time includes training and per-epoch validation. It excludes data
preparation and final test scoring. Model parameter count is also recorded.

## 7. Run the other two English tasks on the SAME word split

```bash
python etymology_baseline/train_lstm.py --data prepared/english --task coarse --out runs/english_coarse_seed42 --seed 42 --device cpu
python etymology_baseline/train_lstm.py --data prepared/english --task parent --out runs/english_parent_seed42 --seed 42 --device cpu
```

Coarse targets combine the accepted fine labels into borrowing/inheritance
and compositional. A word can carry both labels.

For the parent task, the annotated mechanism is an allowed input token, and
each word/mechanism pair becomes one example. The most frequent five parent
languages are selected from this run's training annotations only. All remaining
parent languages map to `<OTHER>`. With fewer than five training languages,
the head contains the available languages plus `<OTHER>`.

This parent-vocabulary choice is documented because the paper's exact
vocabulary construction details are incomplete.

## 8. Repeat seeds and extend the language coverage

Keep `prepared/english` fixed and change only the TRAINING seed and output:

```bash
python etymology_baseline/train_lstm.py --data prepared/english --task fine --out runs/english_fine_seed43 --seed 43 --device cpu
python etymology_baseline/train_lstm.py --data prepared/english --task fine --out runs/english_fine_seed44 --seed 44 --device cpu
```

Repeat the three tasks for each evaluation language. Example:

```bash
python etymology_baseline/prepare_etymology.py --csv Data/etymology.csv --language Japanese --out prepared/japanese --seed 42
python etymology_baseline/train_lstm.py --data prepared/japanese --task fine --out runs/japanese_fine_seed42 --seed 42 --device cpu
```

For the multilingual experiment:

```bash
python etymology_baseline/prepare_etymology.py --csv Data/etymology.csv --language all --out prepared/multilingual --seed 42
python etymology_baseline/train_lstm.py --data prepared/multilingual --task fine --out runs/multilingual_fine_seed42 --seed 42 --device cpu
```

Run the other tasks with `--task coarse` and `--task parent` and new output
directories. Multilingual inputs automatically include a source-language token.
For multilingual parent prediction, the top-five vocabulary is global within
that run's training set. Vocabulary scope is an explicit implementation choice.
The trainer loads prepared word examples into memory; begin with one language
and check its resource requirements before the full multilingual run.

## Model and training configuration

Documented paper settings: character embeddings 128; LSTM hidden size 128;
final hidden state followed by linear output and sigmoid; binary cross-entropy;
Adam learning rate 0.001; 80/10/10 data split; lowest-validation-loss selection.
Fine labels are affix, borrowing, compound, inherited, prefix, and suffix;
the noisy `derived` label is excluded.

The implementation uses `BCEWithLogitsLoss`, with sigmoid applied for inference.
This has the same objective as sigmoid followed by BCE with improved numerical
stability. A softmax is not applied because targets may contain multiple labels.

Our choices for unreported details: one unidirectional layer, no dropout, batch
64, 50-epoch limit, patience 5, seed 42, threshold 0.5, full untruncated spelling,
training-only vocabulary, and explicit accuracy/F1 definitions. These settings
must be described as our choices in the report.

## Preprocessing policy and comparability

The source resembles `droher/etymology-db`, with added IPA information.
Its exact provenance/version remains unconfirmed. The policy is deliberately
conservative and is NOT the original Yawipa preprocessing pipeline.

| CSV relation | Fine target |
| --- | --- |
| `has_affix` | `affix` |
| `borrowed_from` | `borrowing` |
| `compound_of` | `compound` |
| `inherited_from` | `inherited` |
| `has_prefix` | `prefix` |
| `has_suffix` | `suffix` |

Only explicit rows outside groups, or at position 0 through derived-chain
groups, supply targets. Later chain positions describe ancestors and are
excluded. Missing, conflicting, or cyclic group references are excluded.
Children of generic related/affix groups are not automatically converted into
formation labels. Group markers themselves are not target classes.

The parser gives both prefix-root and suffix-root relations the same string
`has_prefix_with_root`; this string supplies no prefix/suffix target here.
Generic derivations, borrowing subtypes, confixes, and other unsupported
relations are excluded from targets. This may remove useful annotations and
change label distributions; inspect the exclusions and review the policy with
the team before making quantitative comparisons with the paper.

For the spelling-only tasks, related-word text, parent language, IPA, IDs,
relationship fields, and graph structure never enter model inputs. Parent
prediction additionally receives its known mechanism, as that task specifies.

Exact-match accuracy means every predicted label must match the gold set.
The paper does not fully define its multilabel decision rule; our accuracy
should not be presented as an identical metric implementation without further
confirmation. Micro F1, macro F1, per-label support, and empty predictions are
reported alongside it. Threshold tuning on test data is not performed.

## What to share after the first full run

Share these two small files for review:

- `prepared/english/preprocess_summary.json`
- `runs/english_fine_seed42/metrics.json`

Keep model weights and full prepared datasets locally or in the team's agreed
repository/storage. Include code, split, source/version notes, software versions,
and seeds in the eventual reproducibility materials. This kit does not implement
the GNN or fusion models.

## Sources

- Paper: https://aclanthology.org/2020.lrec-1.397/
- Original parser: https://github.com/wswu/yawipa
- CSV schema and group semantics: https://github.com/droher/etymology-db
- Root-label alias in parser: https://github.com/droher/etymology-db/blob/master/templates.py
- PyTorch LSTM: https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html
- Stable BCE implementation: https://docs.pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
- Installation: https://pytorch.org/get-started/locally/
