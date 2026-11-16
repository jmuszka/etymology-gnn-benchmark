# etymology-gnn-benchmark

Benchmarking models for computational etymology (Wu and Yarowsky 2020):
a character-level LSTM baseline, with GNN and fusion methods to follow.

## Layout

```
src/                  one package per method
  lstm_baseline/      character LSTM baseline (see its README.md)
notebooks/            Jupyter notebooks for analysis
results/<method>/     small committed result files (metrics, histories)
pyproject.toml        dependencies (managed by uv); uv.lock pins versions
requirements.txt      exported from uv.lock; covers ALL methods + notebooks
```

Local data (`Data/`), prepared datasets (`prepared/`), and run outputs
(`runs/`, `*.pt`) are git-ignored.

## Setup

Install [uv](https://docs.astral.sh/uv/), then from the repository root:

```bash
uv sync            # creates .venv with all deps, installs src/ packages in editable mode
```

Without uv: `python -m venv .venv && .venv/bin/pip install -r requirements.txt -e .`

## Notebooks

```bash
uv run jupyter lab
```

In VS Code, select the `.venv` interpreter as the notebook kernel.
Notebooks import methods directly, e.g. `from lstm_baseline.train_lstm import CharacterLSTM`.

## Managing dependencies

```bash
uv add <package>           # runtime dependency for any method
uv add --dev <package>     # notebook/dev-only tooling
uv export --format requirements-txt --no-hashes --no-emit-project -o requirements.txt
```

Re-run the export after any dependency change so `requirements.txt` stays in sync.

## Adding a new method

1. Create `src/<method>/__init__.py` and its modules.
2. Add `"<method>"` to `module-name` under `[tool.uv.build-backend]` in `pyproject.toml`.
3. Run `uv sync`.
