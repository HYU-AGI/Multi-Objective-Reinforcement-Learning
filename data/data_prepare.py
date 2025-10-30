#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# python data_prepare.py --input ./sample/raw_samples.jsonl --outdir ./sample --train_ratio 0.9 --k_neg 2 --min_margin 0.1
import argparse, json, os, random
from collections import defaultdict

def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line:
                yield json.loads(line)

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def normalize_margin(top, cand, s_min, s_max):
    rng = (s_max - s_min)
    if rng <= 0:
        return 0.0
    return max(0.0, (top - cand) / rng)

def make_pairs(items, metric_key, k_neg=2, min_margin=0.1):
    pairs = []
    for ex in items:
        prompt = (ex.get("prompt") or "").strip()
        gens = [g for g in ex.get("generations", []) if isinstance(g, dict)]
        gens = [g for g in gens if "text" in g and metric_key in g]
        gens = [g for g in gens if isinstance(g[metric_key], (int, float)) and g["text"] and g["text"].strip()]
        if len(gens) < 2 or not prompt:
            continue
        gens.sort(key=lambda g: g[metric_key], reverse=True)
        top = gens[0]
        top_score = float(top[metric_key])
        s_min = float(min(g[metric_key] for g in gens))
        s_max = float(max(g[metric_key] for g in gens))

        negs = []
        for g in gens[1:]:
            nm = normalize_margin(top_score, float(g[metric_key]), s_min, s_max)
            if nm >= min_margin and g["text"].strip() != top["text"].strip():
                negs.append((nm, g["text"].strip()))
        negs.sort(key=lambda x: x[0], reverse=True)
        negs = negs[:k_neg]

        for _, neg_text in negs:
            pairs.append({"prompt": prompt, "chosen": top["text"].strip(), "rejected": neg_text})

    # dedup
    seen, uniq = set(), []
    for p in pairs:
        key = (p["prompt"], p["chosen"], p["rejected"])
        if key not in seen:
            seen.add(key); uniq.append(p)
    return uniq

def split_by_prompt(pairs, train_ratio=0.9, seed=42):
    by_p = defaultdict(list)
    for p in pairs:
        by_p[p["prompt"]].append(p)
    prompts = list(by_p.keys())
    random.Random(seed).shuffle(prompts)
    cut = int(len(prompts)*train_ratio)
    train_prompts = set(prompts[:cut])
    train, val = [], []
    for pr, lst in by_p.items():
        (train if pr in train_prompts else val).extend(lst)
    return train, val

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="./sample/raw samples.jsonl")
    ap.add_argument("--outdir", required=True, help="output root, e.g., ./sample")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--k_neg", type=int, default=2)
    ap.add_argument("--min_margin", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trust_key", default="trust")
    ap.add_argument("--creative_key", default="creativity")
    args = ap.parse_args()

    raw = list(read_jsonl(args.input))
    trust_pairs = make_pairs(raw, args.trust_key,   k_neg=args.k_neg, min_margin=args.min_margin)
    crea_pairs  = make_pairs(raw, args.creative_key, k_neg=args.k_neg, min_margin=args.min_margin)

    t_tr, t_va = split_by_prompt(trust_pairs, train_ratio=args.train_ratio, seed=args.seed)
    c_tr, c_va = split_by_prompt(crea_pairs,  train_ratio=args.train_ratio, seed=args.seed)

    write_jsonl(os.path.join(args.outdir, "trust",    "train.jsonl"), t_tr)
    write_jsonl(os.path.join(args.outdir, "trust",    "validation.jsonl"), t_va)
    write_jsonl(os.path.join(args.outdir, "creative", "train.jsonl"), c_tr)
    write_jsonl(os.path.join(args.outdir, "creative", "validation.jsonl"), c_va)

    print(f"[done] trust train/val: {len(t_tr)}/{len(t_va)}")
    print(f"[done] creative train/val: {len(c_tr)}/{len(c_va)}")

if __name__ == "__main__":
    main()
