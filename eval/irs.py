"""Instruction Responsiveness Score (IRS).

Per the FollowTable paper:
    w(r) = 1 / log2(r + 1)
    G+(pi) = sum_{t in T+_qi} w(r(t, pi))
    G-(pi) = sum_{t in T-_qi} w(r(t, pi))
    S(pi_qi, pi_q) = (G+(pi_qi) - G+(pi_q)) - (G-(pi_qi) - G-(pi_q))
    IRS = S / S_ideal           if S >= 0
        = S / |S_worst|         if S <  0
    IRS in [-1, 1].   IRS = 0 when T+ = T- = empty.

Inputs to the CLI:
  --instr_pickle : {qid_pair -> {doc_id -> score}} from retrieval on queries.jsonl
                   where qid_pair = "{q}_{i}"
  --base_pickle  : {qid_only -> {doc_id -> score}} from retrieval on only_queries.jsonl
                   where qid_only = "{q}"
  --raw_json     : original query_instruction.json (per dataset) so we can recover
                   T+_qi (instruction_positive) and T-_qi (query_positive \ instruction_positive)
  --corpus_size  : N. used as fallback rank for docs missing from the ranking.

Output: per-(q,i) IRS, mean IRS, plus breakdown by instruction_type.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# README numbers — used when corpus_size can't be inferred from the filesystem.
DATASET_CORPUS_SIZES: Dict[str, int] = {
    "WQT": 23784,
    "WTR": 9546,
    "TArX": 11586,
    "IndusTR": 13258,
}


def infer_corpus_size(raw_json: Path) -> Tuple[int, str]:
    """Return (corpus_size, source) inferred from raw_json's location.

    Priority:
      1. count files in <raw_json_dir>/tables/  (ground truth)
      2. count lines in <raw_json_dir>/corpus.jsonl  (if already converted)
      3. fall back to DATASET_CORPUS_SIZES[<parent dir name>]
    """
    parent = raw_json.parent
    tables = parent / "tables"
    if tables.is_dir():
        n = sum(1 for p in tables.iterdir() if p.suffix == ".txt")
        return n, f"counted {n} files in {tables}"
    corpus_jsonl = parent / "corpus.jsonl"
    if corpus_jsonl.is_file():
        n = sum(1 for _ in corpus_jsonl.open(encoding="utf-8"))
        return n, f"counted {n} lines in {corpus_jsonl}"
    name = parent.name
    if name in DATASET_CORPUS_SIZES:
        n = DATASET_CORPUS_SIZES[name]
        return n, f"DATASET_CORPUS_SIZES[{name!r}] = {n}"
    raise SystemExit(
        f"Could not infer corpus_size from {raw_json}. "
        f"Pass --corpus_size explicitly. "
        f"(Tried: {tables}, {corpus_jsonl}, name={name!r})"
    )


# --------------------------------------------------------------------------- #
# Core metric
# --------------------------------------------------------------------------- #

def _ranks_from_scores(scores: Dict[str, float]) -> Dict[str, int]:
    """Sort docs by score desc and return 1-based rank per doc."""
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {doc: r + 1 for r, (doc, _) in enumerate(ordered)}


def _w(rank: int) -> float:
    return 1.0 / math.log2(rank + 1)


def _G(rank_map: Dict[str, int], targets: Iterable[str], fallback_rank: int) -> float:
    return sum(_w(rank_map.get(t, fallback_rank)) for t in targets)


def _S(rank_instr: Dict[str, int], rank_base: Dict[str, int],
       t_plus: List[str], t_minus: List[str], fallback: int) -> float:
    gp_i = _G(rank_instr, t_plus, fallback)
    gm_i = _G(rank_instr, t_minus, fallback)
    gp_b = _G(rank_base, t_plus, fallback)
    gm_b = _G(rank_base, t_minus, fallback)
    return (gp_i - gp_b) - (gm_i - gm_b)


def _ideal_ranks(t_plus: List[str], t_minus: List[str], N: int) -> Dict[str, int]:
    """T+ at top (ranks 1..|T+|), T- at bottom (ranks N-|T-|+1..N)."""
    out: Dict[str, int] = {}
    for i, t in enumerate(t_plus):
        out[t] = i + 1
    for i, t in enumerate(t_minus):
        out[t] = N - len(t_minus) + i + 1
    return out


def _worst_ranks(t_plus: List[str], t_minus: List[str], N: int) -> Dict[str, int]:
    """T- at top, T+ at bottom (mirror of ideal)."""
    out: Dict[str, int] = {}
    for i, t in enumerate(t_minus):
        out[t] = i + 1
    for i, t in enumerate(t_plus):
        out[t] = N - len(t_plus) + i + 1
    return out


def irs(
    ranking_instr: Dict[str, float],
    ranking_base: Dict[str, float],
    t_plus: List[str],
    t_minus: List[str],
    corpus_size: int,
) -> Optional[float]:
    """Compute IRS for a single (q, i) pair.

    Returns 0.0 when both T+ and T- are empty (paper convention).
    Returns None only when the metric is degenerate (ideal == baseline AND
    worst == baseline, i.e. nothing to normalize against) — shouldn't happen
    in practice but guards against div-by-zero.
    """
    if not t_plus and not t_minus:
        return 0.0

    fallback = corpus_size + 1
    rank_i = _ranks_from_scores(ranking_instr)
    rank_b = _ranks_from_scores(ranking_base)

    s_actual = _S(rank_i, rank_b, t_plus, t_minus, fallback)

    ideal = _ideal_ranks(t_plus, t_minus, corpus_size)
    worst = _worst_ranks(t_plus, t_minus, corpus_size)
    s_ideal = _S(ideal, rank_b, t_plus, t_minus, fallback)
    s_worst = _S(worst, rank_b, t_plus, t_minus, fallback)

    if s_actual >= 0:
        return s_actual / s_ideal if s_ideal > 0 else 0.0
    denom = abs(s_worst)
    return s_actual / denom if denom > 0 else 0.0


# --------------------------------------------------------------------------- #
# Driver: load pickles + raw JSON and aggregate
# --------------------------------------------------------------------------- #

def _strip_ext(name: str) -> str:
    return name[:-4] if name.endswith(".txt") else name


def build_pairs(raw_json_path: Path) -> List[dict]:
    """Yield {qid_pair, qid_only, t_plus, t_minus, instruction_type} per (q, i)."""
    data = json.loads(raw_json_path.read_text(encoding="utf-8"))
    pairs: List[dict] = []
    for q_idx, entry in enumerate(data):
        outs = entry.get("outs") or []
        if not outs:
            continue
        qid_only = str(q_idx)
        for ins_idx, out in enumerate(outs):
            ip = {_strip_ext(x) for x in (out.get("instruction_positive") or [])}
            qp = {_strip_ext(x) for x in (out.get("query_positive") or [])}
            pairs.append({
                "qid_pair": f"{q_idx}_{ins_idx}",
                "qid_only": qid_only,
                "t_plus": sorted(ip),
                "t_minus": sorted(qp - ip),
                "instruction_type": out.get("instruction_type", ""),
            })
    return pairs


def evaluate(
    instr_results: Dict[str, Dict[str, float]],
    base_results: Dict[str, Dict[str, float]],
    pairs: List[dict],
    corpus_size: int,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, List[float]]]:
    """Returns (per_qid IRS, summary, per_type IRS list)."""
    per_qid: Dict[str, float] = {}
    per_type: Dict[str, List[float]] = defaultdict(list)
    skipped = 0
    for p in pairs:
        ri = instr_results.get(p["qid_pair"])
        rb = base_results.get(p["qid_only"])
        if ri is None or rb is None:
            skipped += 1
            continue
        v = irs(ri, rb, p["t_plus"], p["t_minus"], corpus_size)
        if v is None:
            skipped += 1
            continue
        per_qid[p["qid_pair"]] = v
        per_type[p["instruction_type"]].append(v)

    vals = list(per_qid.values())
    summary = {
        "n": float(len(vals)),
        "skipped": float(skipped),
        "IRS_mean": sum(vals) / len(vals) if vals else 0.0,
        "IRS_positive_rate": sum(1 for v in vals if v > 0) / len(vals) if vals else 0.0,
        "IRS_negative_rate": sum(1 for v in vals if v < 0) / len(vals) if vals else 0.0,
    }
    return per_qid, summary, per_type


def main() -> None:
    p = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description=__doc__)
    p.add_argument("--instr_pickle", required=True, help="pickle from retrieval on queries.jsonl")
    p.add_argument("--base_pickle",  required=True, help="pickle from retrieval on only_queries.jsonl")
    p.add_argument("--raw_json",     required=True, help="query_instruction.json for the dataset")
    p.add_argument("--corpus_size",  type=int, default=None,
                   help="N = number of tables. Auto-inferred from --raw_json if omitted.")
    p.add_argument("--out_json",     default=None, help="optional: dump per-qid IRS here")
    p.add_argument("--model_name",   default=None, help="optional: model name to include in output")
    args = p.parse_args()

    raw_json = Path(args.raw_json)
    if args.corpus_size is None:
        corpus_size, source = infer_corpus_size(raw_json)
        print(f"corpus_size inferred: {corpus_size}  ({source})")
    else:
        corpus_size = args.corpus_size

    with open(args.instr_pickle, "rb") as f:
        instr = pickle.load(f)
    with open(args.base_pickle, "rb") as f:
        base = pickle.load(f)

    pairs = build_pairs(raw_json)
    per_qid, summary, per_type = evaluate(instr, base, pairs, corpus_size)

    print(f"n={int(summary['n'])}  skipped={int(summary['skipped'])}")
    print(f"IRS mean       : {summary['IRS_mean']:.4f}")
    print(f"IRS > 0 rate   : {summary['IRS_positive_rate']:.4f}")
    print(f"IRS < 0 rate   : {summary['IRS_negative_rate']:.4f}")
    print("Per instruction_type:")
    for t in sorted(per_type):
        v = per_type[t]
        print(f"  {t:8s} n={len(v):>4}  IRS_mean={sum(v)/len(v):+.4f}")

    if args.out_json:
        output = {
            "summary": summary,
            "per_qid": per_qid,
            "per_type_mean": {t: sum(v)/len(v) for t, v in per_type.items()}
        }
        if args.model_name:
            output["model_name"] = args.model_name
        with open(args.out_json, "a", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
