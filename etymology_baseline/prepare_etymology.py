"""Build word-level examples and a fixed split from an etymology-db-style CSV.

This is a documented adaptation to the supplied CSV, not the authors' original
preprocessor. It streams the CSV into temporary SQLite storage.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
from itertools import groupby
import json
from pathlib import Path
import sqlite3
import sys
import time


FINE_LABELS = ["affix", "borrowing", "compound", "inherited", "prefix", "suffix"]
COARSE_LABELS = ["borrowing_inheritance", "compositional"]
RELATION_MAP = {
    "has_affix": "affix", "borrowed_from": "borrowing",
    "compound_of": "compound", "inherited_from": "inherited",
    "has_prefix": "prefix", "has_suffix": "suffix",
}
FIELDS = [
    "term_id", "lang", "term", "reltype", "related_lang", "related_term",
    "group_tag", "parent_tag", "parent_position", "position",
]
POLICY = {
    "name": "conservative_immediate_explicit_v1",
    "relation_to_label": RELATION_MAP,
    "accepted_context": "Unnested rows, or position 0 through derived groups only.",
    "group_context": "Affix/related groups do not automatically provide targets.",
    "ambiguous_hierarchy": "Missing, conflicting, or cyclic groups are excluded.",
    "other_relations": "Retained in the original CSV, excluded from target labels.",
    "text": "Preserve original spelling, case, and diacritics.",
    "scope": "Adapted dataset; exact paper data and preprocessing unconfirmed.",
}


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def word_id(language, word):
    value = json.dumps([language, word], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def immediate_context(row, groups):
    """Return eligibility and a reason, using every enclosing group."""
    parent, position = row["parent_tag"], row["parent_position"]
    seen = set()
    while parent:
        if parent in seen:
            return False, "cyclic_group"
        seen.add(parent)
        definitions = groups.get(parent, set())
        if len(definitions) != 1:
            return False, "missing_or_conflicting_group"
        kind, next_parent, next_position = next(iter(definitions))
        if kind != "group_derived_root":
            return False, "non_derived_group_context"
        try:
            if int(position) != 0:
                return False, "later_ancestor_in_chain"
        except (TypeError, ValueError):
            return False, "unknown_group_position"
        parent, position = next_parent, next_position
    return True, "immediate"


def example_from_rows(language, word, rows, diagnostics):
    groups = defaultdict(set)
    for row in rows:
        if row["group_tag"]:
            groups[row["group_tag"]].add(
                (row["reltype"], row["parent_tag"], row["parent_position"])
            )
    labels = set()
    parents = defaultdict(set)
    for row in rows:
        label = RELATION_MAP.get(row["reltype"])
        if label is None:
            continue
        diagnostics["candidate_label_rows"] += 1
        eligible, reason = immediate_context(row, groups)
        diagnostics[reason] += 1
        if not eligible:
            continue
        labels.add(label)
        if row["related_lang"]:
            parents[label].add(row["related_lang"])
    if not labels:
        diagnostics["words_without_eligible_label"] += 1
        return None
    coarse = set()
    if labels.intersection({"borrowing", "inherited"}):
        coarse.add(COARSE_LABELS[0])
    if labels.intersection({"affix", "compound", "prefix", "suffix"}):
        coarse.add(COARSE_LABELS[1])
    return {
        "id": word_id(language, word), "language": language, "word": word,
        "fine_labels": [x for x in FINE_LABELS if x in labels],
        "coarse_labels": [x for x in COARSE_LABELS if x in coarse],
        "parent_languages_by_mechanism": {
            label: sorted(parents[label]) for label in FINE_LABELS if parents[label]
        },
        "source_term_ids": sorted({r["term_id"] for r in rows if r["term_id"]}),
    }


def split_sizes(n):
    if n < 3:
        return n, 0, 0
    train, validation = max(1, n * 8 // 10), max(1, n // 10)
    if train + validation >= n:
        train = n - validation - 1
    return train, validation, n - train - validation


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare(csv_path, out, languages, seed, batch_size=20000):
    if out.exists():
        raise ValueError(f"Output already exists: {out}. Choose a new --out directory.")
    out.mkdir(parents=True)
    started = time.perf_counter()
    selected = None if languages == ["all"] else set(languages)
    diagnostics = Counter()
    relations = Counter()
    db_path = out / "_working.sqlite"
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=OFF")
    db.execute("PRAGMA temp_store=FILE")
    db.execute("CREATE TABLE records (" + ",".join(f"{c} TEXT NOT NULL" for c in FIELDS) + ")")
    db.execute("CREATE TABLE examples (language TEXT, sort_key TEXT, payload TEXT)")
    insert = "INSERT INTO records VALUES (" + ",".join("?" for _ in FIELDS) + ")"
    csv.field_size_limit(sys.maxsize)
    try:
        batch = []
        with csv_path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = sorted(set(FIELDS) - set(reader.fieldnames or []))
            if missing:
                raise ValueError(f"Missing CSV columns: {missing}")
            for row in reader:
                diagnostics["scanned_csv_rows"] += 1
                if None in row or any(value is None for value in row.values()):
                    raise ValueError(f"Malformed CSV record near physical line {reader.line_num}.")
                if selected is not None and row["lang"] not in selected:
                    continue
                if not row["lang"] or not row["term"]:
                    diagnostics["empty_source_language_or_word"] += 1
                    continue
                relations[row["reltype"]] += 1
                diagnostics["selected_csv_rows"] += 1
                batch.append(tuple(row[c] for c in FIELDS))
                if len(batch) >= batch_size:
                    db.executemany(insert, batch)
                    db.commit()
                    batch.clear()
                    print(f"Selected {diagnostics['selected_csv_rows']:,} rows...", flush=True)
            if batch:
                db.executemany(insert, batch)
                db.commit()
        if not diagnostics["selected_csv_rows"]:
            raise ValueError("No matching rows. Check --language and the CSV path.")
        db.execute("CREATE INDEX records_word ON records(lang, term)")
        cursor = db.execute("SELECT * FROM records ORDER BY lang, term, rowid")
        pending = []
        for (language, word), grouped in groupby(cursor, key=lambda r: (r["lang"], r["term"])):
            rows = [dict(r) for r in grouped]
            diagnostics["unique_source_words"] += 1
            example = example_from_rows(language, word, rows, diagnostics)
            if example is not None:
                rank = hashlib.sha256(f"{seed}:{example['id']}".encode()).hexdigest()
                pending.append((language, rank, json.dumps(example, ensure_ascii=False)))
            if len(pending) >= 1000:
                db.executemany("INSERT INTO examples VALUES (?,?,?)", pending)
                db.commit()
                pending.clear()
        if pending:
            db.executemany("INSERT INTO examples VALUES (?,?,?)", pending)
            db.commit()
        counts = dict(db.execute("SELECT language, COUNT(*) FROM examples GROUP BY language"))
        if not counts:
            raise ValueError("No eligible words under the conservative label policy.")
        allocations = {lang: split_sizes(n) for lang, n in counts.items()}
        db.execute("CREATE INDEX examples_order ON examples(language, sort_key)")
        stats = {split: {"words": 0, "fine_label_counts": Counter(),
                         "coarse_label_counts": Counter(), "languages": Counter(),
                         "parent_task_examples": 0}
                 for split in ["train", "validation", "test"]}
        streams = {name: (out / f"{name}.jsonl").open("w", encoding="utf-8") for name in stats}
        try:
            with (out / "split.csv").open("w", encoding="utf-8", newline="") as split_stream:
                writer = csv.writer(split_stream)
                writer.writerow(["id", "language", "word", "split"])
                previous, index = None, 0
                for row in db.execute("SELECT language, payload FROM examples ORDER BY language, sort_key"):
                    language = row["language"]
                    if language != previous:
                        previous, index = language, 0
                    n_train, n_validation, _ = allocations[language]
                    split = "train" if index < n_train else (
                        "validation" if index < n_train + n_validation else "test")
                    index += 1
                    example = json.loads(row["payload"])
                    streams[split].write(row["payload"] + "\n")
                    writer.writerow([example["id"], language, example["word"], split])
                    stats[split]["words"] += 1
                    stats[split]["languages"][language] += 1
                    stats[split]["fine_label_counts"].update(example["fine_labels"])
                    stats[split]["coarse_label_counts"].update(example["coarse_labels"])
                    stats[split]["parent_task_examples"] += len(example["parent_languages_by_mechanism"])
        finally:
            for stream in streams.values():
                stream.close()
        for values in stats.values():
            for key, value in list(values.items()):
                if isinstance(value, Counter):
                    values[key] = dict(value)
        report = {
            "status": "complete", "format_version": 1,
            "source_file": csv_path.name, "source_size_bytes": csv_path.stat().st_size,
            "source_sha256": file_sha256(csv_path), "requested_languages": languages,
            "fine_label_order": FINE_LABELS, "coarse_label_order": COARSE_LABELS,
            "split_seed": seed, "split_method": "Seeded hash order within each language; fixed word groups.",
            "small_language_policy": "<3 eligible words: training only; otherwise reserve validation/test.",
            "label_policy": POLICY, "diagnostics": dict(diagnostics),
            "selected_relationship_row_counts": dict(relations.most_common()),
            "eligible_words_by_language": counts, "splits": stats,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "paper_reference": "https://aclanthology.org/2020.lrec-1.397/",
            "comparability": "Method reimplementation on a different, incompletely verified data source.",
        }
        write_json(out / "preprocess_summary.json", report)
        print(json.dumps({name: values["words"] for name, values in stats.items()}, indent=2))
        print(f"Prepared data: {out.resolve()}", flush=True)
        return report
    finally:
        db.close()
        db_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--language", action="append", help="CSV language name; repeat, or use all. Default English.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    languages = args.language or ["English"]
    if "all" in languages and languages != ["all"]:
        parser.error("Use --language all by itself.")
    source = args.csv.expanduser().resolve()
    if not source.is_file():
        parser.error(f"CSV not found: {source}")
    try:
        prepare(source, args.out.expanduser(), languages, args.seed)
    except (ValueError, OSError, csv.Error, sqlite3.Error) as error:
        parser.exit(1, f"Preparation failed: {error}\n")


if __name__ == "__main__":
    main()
