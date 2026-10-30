"""Run focused checks: python etymology_baseline/test_pipeline.py."""

import csv
import json
from pathlib import Path
import random
import tempfile
import unittest
from collections import Counter

from prepare_etymology import FIELDS, example_from_rows, prepare, split_sizes

try:
    import torch
    from train_lstm import CharacterLSTM, Metrics, input_tokens, load_words
except ImportError:
    torch = None


def row(kind, parent="", parent_position="", group="", related_lang="English", term="made"):
    value = {field: "" for field in FIELDS}
    value.update({"lang": "English", "term": term, "term_id": "source-" + term,
                  "reltype": kind, "related_lang": related_lang, "related_term": "base",
                  "group_tag": group, "parent_tag": parent,
                  "parent_position": parent_position, "position": "0"})
    return value


class PreparationTests(unittest.TestCase):
    def test_immediate_ancestor_is_kept_and_older_formation_excluded(self):
        rows = [row("group_derived_root", group="chain"),
                row("inherited_from", "chain", "0", related_lang="Middle English"),
                row("has_suffix", "chain", "1", related_lang="Old English"),
                row("group_derived_root", "chain", "1", group="older"),
                row("has_prefix", "older", "0")]
        result = example_from_rows("English", "made", rows, Counter())
        self.assertEqual(result["fine_labels"], ["inherited"])
        self.assertEqual(result["parent_languages_by_mechanism"], {"inherited": ["Middle English"]})

    def test_stem_label_and_generic_derivation_do_not_create_prefix_target(self):
        rows = [row("has_prefix_with_root"), row("derived_from"), row("has_suffix")]
        result = example_from_rows("English", "made", rows, Counter())
        self.assertEqual(result["fine_labels"], ["suffix"])

    def test_multilabel_and_duplicates(self):
        rows = [row("borrowed_from", related_lang="Latin"), row("has_suffix"), row("has_suffix")]
        result = example_from_rows("English", "made", rows, Counter())
        self.assertEqual(result["fine_labels"], ["borrowing", "suffix"])
        self.assertEqual(result["coarse_labels"], ["borrowing_inheritance", "compositional"])
        self.assertEqual(result["parent_languages_by_mechanism"]["suffix"], ["English"])

    def test_unresolved_and_cyclic_groups_do_not_supply_labels(self):
        rows = [row("group_derived_root", "B", "0", group="A"),
                row("group_derived_root", "A", "0", group="B"),
                row("has_prefix", "A", "0"), row("borrowed_from", "missing", "0")]
        self.assertIsNone(example_from_rows("English", "made", rows, Counter()))

    def test_related_or_affix_group_children_do_not_automatically_supply_targets(self):
        for kind in ["group_affix_root", "group_related_root"]:
            rows = [row(kind, group="container"), row("has_prefix", "container", "0")]
            self.assertIsNone(example_from_rows("English", "made", rows, Counter()))

    def test_split_sizes(self):
        self.assertEqual(split_sizes(100), (80, 10, 10))
        self.assertEqual(split_sizes(3), (1, 1, 1))
        self.assertEqual(split_sizes(2), (2, 0, 0))

    def test_export_is_invariant_to_source_row_order_and_has_disjoint_words(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [row("has_suffix", term=f"form-{i}") for i in range(30)]
            rows += [row("has_suffix", term="form-0")]
            exported = []
            for trial in range(2):
                random.Random(trial).shuffle(rows)
                source = root / f"input-{trial}.csv"
                with source.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=FIELDS)
                    writer.writeheader()
                    writer.writerows(rows)
                destination = root / f"prepared-{trial}"
                report = prepare(source, destination, ["English"], 42, batch_size=7)
                self.assertEqual(report["diagnostics"]["unique_source_words"], 30)
                exported.append({split: (destination / f"{split}.jsonl").read_text()
                                 for split in ["train", "validation", "test"]})
            self.assertEqual(exported[0], exported[1])
            sets = [{json.loads(line)["id"] for line in exported[0][split].splitlines()}
                    for split in ["train", "validation", "test"]]
            self.assertEqual([len(x) for x in sets], [24, 3, 3])
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            if torch is not None:
                parts, _ = load_words(root / "prepared-0")
                self.assertEqual(len(parts["train"]), 24)
                path = root / "prepared-0" / "validation.jsonl"
                lines = path.read_text().splitlines()
                lines[0] = json.dumps(parts["train"][0])
                path.write_text("\n".join(lines) + "\n")
                with self.assertRaises(ValueError):
                    load_words(root / "prepared-0")


@unittest.skipIf(torch is None, "Install torch to run neural-model checks.")
class TrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)

    def test_padding_does_not_change_last_real_hidden_state(self):
        model = CharacterLSTM(12, 6).eval()
        with torch.no_grad():
            alone = model(torch.tensor([[2, 3]]), torch.tensor([2]))
            batched = model(torch.tensor([[2, 3, 0, 0], [4, 5, 6, 7]]), torch.tensor([2, 4]))
        torch.testing.assert_close(alone[0], batched[0], atol=1e-6, rtol=1e-5)

    def test_input_contains_spelling_and_only_allowed_context(self):
        example = {"word": "Å字", "language": "English", "mechanism": None}
        self.assertEqual(input_tokens(example, False), ["char:Å", "char:字"])
        self.assertEqual(input_tokens(example, True)[0], "lang:English")
        example["mechanism"] = "suffix"
        self.assertEqual(input_tokens(example, False)[-1], "mechanism:suffix")

    def test_model_can_learn_a_small_controlled_problem(self):
        model = CharacterLSTM(10, 6)
        inputs = torch.tensor([[i] * 4 for i in range(2, 8)])
        lengths, targets = torch.tensor([4] * 6), torch.eye(6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCEWithLogitsLoss()
        initial = float(criterion(model(inputs, lengths), targets).detach())
        for _ in range(120):
            optimizer.zero_grad()
            loss = criterion(model(inputs, lengths), targets)
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            logits = model(inputs, lengths)
            final = float(criterion(logits, targets))
        self.assertLess(final, initial * 0.4)
        self.assertTrue(torch.equal(logits.sigmoid() >= 0.5, targets.bool()))

    def test_metrics_agree_with_hand_calculated_example(self):
        metrics = Metrics(["A", "B"])
        metrics.update(torch.tensor([[1, 0], [1, 1]]), torch.tensor([[1, 0], [0, 1]]))
        result = metrics.result()
        self.assertAlmostEqual(result["exact_match_accuracy"], 0.5)
        self.assertAlmostEqual(result["micro_f1"], 0.8)
        self.assertAlmostEqual(result["macro_f1"], 5 / 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
