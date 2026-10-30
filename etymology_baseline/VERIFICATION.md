# Verification performed

Validation environment: Linux CPU, PyTorch 2.5.1+cpu.

All 11 focused checks in `test_pipeline.py` passed, including:

- Immediate versus older ancestor handling; unresolved and cyclic groups.
- Excluding the shared prefix/suffix root label from prefix targets.
- Multi-label targets and deduplicated annotations.
- Fixed, disjoint splits independent of CSV row order.
- Rejection of a deliberately duplicated word across partitions.
- Unicode/context token handling and padding invariance.
- A controlled six-class problem that the model learned successfully.
- Accuracy and F1 against a hand-calculated example.

End-to-end preparation and training also completed on the uploaded 1,000-row
preview for English fine, coarse, and parent prediction (three epochs each),
and multilingual fine prediction (one epoch).

Verified that saved checkpoint selection matches minimum validation loss,
prediction counts match evaluation counts, probabilities are finite, and the
input vocabulary contains only training-derived permitted tokens.

These are SOFTWARE checks. The preview produced only 17 eligible English word
examples under the conservative policy and lacked several target classes.
It cannot establish scientific performance or reproduce the paper's results.
The full 811 MB CSV was not available here; full-data coverage, resource use,
and model performance must be checked on the user's computer.

The conservative preprocessing policy, parent vocabulary construction,
unreported hyperparameters, and metric definitions are explicit adaptations.
See `README.md` and the generated JSON summaries before interpreting results.
