#!/usr/bin/env python3

import argparse
import json

import pandas as pd
from utils.evaluation_metrics import precision_at_k, intrinsic_sts_correlation

from your_recommendation_module import recommend_for_keywords  # wherever that lives


def load_ground_truth(path):
    """Expect a JSON file: { "query1": [id1,id2,...], ... }"""
    with open(path) as f:
        raw = json.load(f)
    queries = list(raw.keys())
    gt_sets = [set(raw[q]) for q in queries]
    return queries, gt_sets


def main(args):
    # 1) Load ground truth
    queries, ground_truth = load_ground_truth(args.ground_truth)

    # 2) Generate recommendations
    recommended = []
    for q in queries:
        df_res = recommend_for_keywords(q, top_k=args.top_k,
                                        filter_platform=args.platform,
                                        filter_level=args.level)
        recommended.append(df_res['global_id'].tolist())

    # 3) Compute metrics
    prec = precision_at_k(recommended, ground_truth, k=args.top_k)
    print(f"Precision@{args.top_k}: {prec:.3f}")

    # (Optional) if you have graded scores for each query-item pair:
    if args.spearman:
        # assemble flat lists of sims and human scores here...
        # then call intrinsic_sts_correlation(embs1, embs2, gold_scores)
        pass


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Evaluate TrioLearn recommender")
    p.add_argument("--ground-truth", required=True,
                   help="Path to JSON file of query→[relevant_ids]")
    p.add_argument("--top-k", type=int, default=5,
                   help="How many recommendations per query to evaluate")
    p.add_argument("--platform", default=None,
                   help="Optional platform filter (case-insensitive)")
    p.add_argument("--level", default=None,
                   help="Optional level filter (case-insensitive)")
    p.add_argument("--spearman", action="store_true",
                   help="Also compute Spearman’s ρ if you have graded labels")
    args = p.parse_args()

    main(args)
