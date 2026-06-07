"""Convert FollowTable raw data into BEIR / GenericDataLoader format.

Input layout (this directory):
  {dataset}/
    tables/<id>.txt                 raw markdown / HTML table content
    query_instruction.json          list of {query, query_positive, outs:[{instruction, ...}]}

Output layout (per dataset):
  {out_root}/{dataset}/
    corpus.jsonl                    {"_id","text","title","metadata"}
    queries.jsonl                   query+instruction pairs, _id = "{q_idx}_{ins_idx}"
                                    text = "{instruction} [SEP] {query}"  (InstructIR-style)
    only_queries.jsonl              query-only, _id = "{q_idx}"
    qrels/
      test.tsv                      instruction-aware (instruction_positive -> 1)
      for_only_query_test.tsv       topic-only (query_positive -> 1)

CLI:
  python make_dataset.py convert --out ./beir
  python make_dataset.py upload  --src ./beir --repo_id <user>/FollowTable [--private]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

DATASETS = ["new_WQT"] # "WQT", "WTR", "TArX", "IndusTR"
SEP = " [SEP] "

_MD_TITLE_RE = re.compile(r"^#\s*(.+?)\s*$", re.MULTILINE)
_HTML_CAPTION_RE = re.compile(r"<caption[^>]*>(.*?)</caption>", re.IGNORECASE | re.DOTALL)


def extract_title(text: str) -> str:
    m = _HTML_CAPTION_RE.search(text)
    if m:
        return re.sub(r"\s+", " ", m.group(1)).strip()
    m = _MD_TITLE_RE.search(text)
    if m:
        return m.group(1).strip()
    return ""


def table_id(filename: str) -> str:
    return Path(filename).stem


def write_jsonl(path: Path, rows) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def write_tsv(path: Path, rows: List[Tuple[str, str, int]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        w.writerow(["query-id", "corpus-id", "score"])
        for r in rows:
            w.writerow(r)
    return len(rows)


def build_corpus(src_dir: Path, out_dir: Path) -> int:
    tables_dir = src_dir / "tables"
    files = sorted(tables_dir.iterdir(), key=lambda p: p.name)

    def gen():
        for fp in files:
            if not fp.name.endswith(".txt"):
                continue
            text = fp.read_text(encoding="utf-8", errors="replace")
            yield {
                "_id": table_id(fp.name),
                "text": text,
                "title": extract_title(text),
                "metadata": {},
            }

    return write_jsonl(out_dir / "corpus.jsonl", gen())


def build_queries_and_qrels(src_dir: Path, out_dir: Path) -> Dict[str, int]:
    data = json.loads((src_dir / "query_instruction.json").read_text(encoding="utf-8"))

    pair_queries: List[dict] = []
    only_queries: List[dict] = []
    pair_qrels: List[Tuple[str, str, int]] = []
    only_qrels: List[Tuple[str, str, int]] = []

    for q_idx, entry in enumerate(data):
        query = entry.get("query", "") or ""
        outs = entry.get("outs") or []
        if not outs:
            continue

        qid_only = str(q_idx)
        only_queries.append({
            "_id": qid_only,
            "text": query,
            "metadata": {"origin_query": query},
        })
        for pid in entry.get("query_positive", []) or []:
            only_qrels.append((qid_only, table_id(pid), 1))

        for ins_idx, out in enumerate(outs):
            instruction = out.get("instruction", "") or ""
            qid_pair = f"{q_idx}_{ins_idx}"
            pair_queries.append({
                "_id": qid_pair,
                "text": f"{instruction}{SEP}{query}",
                "metadata": {
                    "origin_query": query,
                    "instruction": instruction,
                    "instruction_type": out.get("instruction_type", ""),
                },
            })
            for pid in out.get("instruction_positive", []) or []:
                pair_qrels.append((qid_pair, table_id(pid), 1))

    counts = {
        "queries.jsonl": write_jsonl(out_dir / "queries.jsonl", pair_queries),
        "only_queries.jsonl": write_jsonl(out_dir / "only_queries.jsonl", only_queries),
        "qrels/test.tsv": write_tsv(out_dir / "qrels" / "test.tsv", pair_qrels),
        "qrels/for_only_query_test.tsv": write_tsv(
            out_dir / "qrels" / "for_only_query_test.tsv", only_qrels
        ),
    }
    return counts


def convert_one(dataset: str, src_root: Path, out_root: Path) -> None:
    src = src_root / dataset
    out = out_root / dataset
    print(f"[{dataset}] converting  src={src}  ->  out={out}")
    n_docs = build_corpus(src, out)
    counts = build_queries_and_qrels(src, out)
    print(f"[{dataset}] corpus.jsonl={n_docs} " + " ".join(f"{k}={v}" for k, v in counts.items()))


def cmd_convert(args: argparse.Namespace) -> None:
    src_root = Path(args.src).resolve()
    out_root = Path(args.out).resolve()
    datasets = args.datasets or DATASETS
    for d in datasets:
        convert_one(d, src_root, out_root)


def cmd_upload(args: argparse.Namespace) -> None:
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError as e:
        raise SystemExit(
            "huggingface_hub is required. Install with: pip install huggingface_hub"
        ) from e

    src = Path(args.src).resolve()
    if not src.exists():
        raise SystemExit(f"--src does not exist: {src}")

    api = HfApi(token=args.token)
    create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
        token=args.token,
    )
    print(f"Uploading {src} -> {args.repo_id} (dataset)")
    api.upload_folder(
        folder_path=str(src),
        repo_id=args.repo_id,
        repo_type="dataset",
        commit_message=args.message,
        path_in_repo=args.path_in_repo,
        allow_patterns=args.allow_patterns or None,
        ignore_patterns=args.ignore_patterns or None,
    )
    print(f"Done. https://huggingface.co/datasets/{args.repo_id}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser("convert", help="convert raw data -> BEIR layout")
    pc.add_argument("--src", default=str(Path(__file__).parent), help="raw data root (default: this directory)")
    pc.add_argument("--out", default=str(Path(__file__).parent / "beir"), help="output root")
    pc.add_argument("--datasets", nargs="*", choices=DATASETS, help="subset of datasets to convert")
    pc.set_defaults(func=cmd_convert)

    pu = sub.add_parser("upload", help="upload converted folder to HuggingFace Hub")
    pu.add_argument("--src", required=True, help="converted folder root (e.g., ./beir)")
    pu.add_argument("--repo_id", required=True, help="<user-or-org>/<dataset-name>")
    pu.add_argument("--private", action="store_true")
    pu.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (default: $HF_TOKEN / cached login)")
    pu.add_argument("--message", default="Upload FollowTable BEIR-format data")
    pu.add_argument("--path_in_repo", default=None, help="optional sub-path inside the repo")
    pu.add_argument("--allow_patterns", nargs="*", default=None)
    pu.add_argument("--ignore_patterns", nargs="*", default=None)
    pu.set_defaults(func=cmd_upload)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
