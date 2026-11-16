"""Build the word-relation graph from an etymology-db-style CSV.

Nodes are exact (language, term) pairs, the same word identity the LSTM
baseline uses. Every CSV row that names a related term becomes one typed,
directed edge from the entry's word (the edge's owner) to the related term.
The graph holds every entry; hiding a word's own entry when it is the
prediction target is done by the trainer, not here.
"""

import argparse
from array import array
from collections import Counter
import csv
from pathlib import Path
import sys
import time

import numpy as np

from lstm_baseline.prepare_etymology import file_sha256, write_json


FIELDS = ["lang", "term", "reltype", "related_lang", "related_term"]
UNKNOWN_LANGUAGE = "<unknown>"
POLICY = {
    "name": "all_entries_typed_edges_v1",
    "node": "Exact (language, term); case and diacritics preserved.",
    "edge": "One per distinct (owner, relation, related term); owner is the entry's word.",
    "dropped_rows": "Group markers (no related term), self-loops, and exact duplicates.",
    "ignored_fields": "Group nesting and positions are not encoded; IPA and term ids are unused.",
    "target_masking": "Applied at training time: a word's own entry is hidden when it is the target.",
}


def build(csv_path, out):
    if out.exists():
        raise ValueError(f"Output already exists: {out}. Choose a new --out directory.")
    out.mkdir(parents=True)
    started = time.perf_counter()
    nodes, languages, relations = {}, {}, {}
    terms, node_language = [], array("i")
    sources, kinds, targets = array("i"), array("i"), array("i")
    diagnostics = Counter()

    def node(language, term):
        key = (language, term)
        index = nodes.get(key)
        if index is None:
            index = nodes[key] = len(terms)
            terms.append(term)
            node_language.append(languages.setdefault(language, len(languages)))
        return index

    csv.field_size_limit(sys.maxsize)
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = sorted(set(FIELDS) - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing CSV columns: {missing}")
        for row in reader:
            diagnostics["scanned_csv_rows"] += 1
            if not row["lang"] or not row["term"]:
                diagnostics["empty_source_language_or_word"] += 1
                continue
            owner = node(row["lang"], row["term"])
            if not row["related_term"]:
                diagnostics["rows_without_related_term"] += 1
                continue
            if not row["related_lang"]:
                diagnostics["related_language_unknown"] += 1
            related = node(row["related_lang"] or UNKNOWN_LANGUAGE, row["related_term"])
            if related == owner:
                diagnostics["self_loops"] += 1
                continue
            sources.append(owner)
            kinds.append(relations.setdefault(row["reltype"], len(relations)))
            targets.append(related)
            if diagnostics["scanned_csv_rows"] % 1_000_000 == 0:
                print(f"Scanned {diagnostics['scanned_csv_rows']:,} rows...", flush=True)
    if not sources:
        raise ValueError("No relation rows found. Check the CSV path.")
    edges = np.unique(np.stack([np.frombuffer(sources, dtype=np.int32),
                                np.frombuffer(kinds, dtype=np.int32),
                                np.frombuffer(targets, dtype=np.int32)], axis=1), axis=0)
    diagnostics["duplicate_edges"] = len(sources) - len(edges)
    np.savez(out / "graph.npz", source=edges[:, 0], relation=edges[:, 1], target=edges[:, 2],
             node_language=np.frombuffer(node_language, dtype=np.int32))
    write_json(out / "nodes.json", {"languages": list(languages), "relations": list(relations), "terms": terms})
    relation_counts = np.bincount(edges[:, 1], minlength=len(relations))
    summary = {
        "status": "complete", "format_version": 1,
        "source_file": csv_path.name, "source_sha256": file_sha256(csv_path),
        "nodes": len(terms), "edges": len(edges), "languages": len(languages),
        "nodes_with_an_entry_edge": int(len(np.unique(edges[:, 0]))),
        "nodes_cited_by_another_entry": int(len(np.unique(edges[:, 2]))),
        "relation_edge_counts": {name: int(relation_counts[i]) for name, i in
                                 sorted(relations.items(), key=lambda item: -relation_counts[item[1]])},
        "diagnostics": dict(diagnostics), "policy": POLICY,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    write_json(out / "graph_summary.json", summary)
    print(f"Nodes: {len(terms):,}; edges: {len(edges):,}; relations: {len(relations)}")
    print(f"Graph: {out.resolve()}", flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    source = args.csv.expanduser().resolve()
    if not source.is_file():
        parser.error(f"CSV not found: {source}")
    try:
        build(source, args.out.expanduser())
    except (ValueError, OSError, csv.Error) as error:
        parser.exit(1, f"Graph build failed: {error}\n")


if __name__ == "__main__":
    main()
