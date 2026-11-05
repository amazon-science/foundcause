#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FoundCause inference from a CSV of observational data.

Usage:
    python predict.py --data path/to/data.csv [options]

See README.md for details.
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch

try:
    from foundcause import ModelConfig, CausalDiscoveryTransformer, predict
except ImportError as e:
    print(f"ERROR: could not import foundcause.py: {e}", file=sys.stderr)
    sys.exit(1)


def enforce_dag(adjacency: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Greedy DAG post-processing: keep the predicted edges in descending probability
    order, skipping any that would create a cycle."""
    import networkx as nx

    n = prob.shape[0]
    candidates = [
        (prob[i, j], i, j)
        for i in range(n) for j in range(n)
        if i != j and adjacency[i, j] > 0
    ]
    candidates.sort(reverse=True)
    G = nx.DiGraph()
    G.add_nodes_from(range(n))
    out = np.zeros_like(prob)
    for _, i, j in candidates:
        if nx.has_path(G, j, i):
            continue
        G.add_edge(i, j)
        out[i, j] = 1.0
    return out


def load_model(checkpoint_path: str, device: torch.device):
    cfg = ModelConfig()
    model = CausalDiscoveryTransformer(cfg)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model


def load_csv(path: str):
    """Load a CSV with or without a header row. Returns (X, column_names)."""
    df = pd.read_csv(path)
    if not all(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns):
        df = pd.read_csv(path, header=None)
        df.columns = [f"var_{i}" for i in range(df.shape[1])]
    return df.values.astype(np.float32), [str(c) for c in df.columns]


def parse_args():
    p = argparse.ArgumentParser(description="FoundCause inference.")
    p.add_argument("--data", required=True, help="Input CSV (rows=observations, cols=variables).")
    p.add_argument("--checkpoint", default="checkpoint.pt")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output-dir", default=None, help="Default: same dir as input CSV.")
    p.add_argument("--n-runs", type=int, default=10, help="Permutation-averaged runs.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-samples", type=int, default=5000)
    p.add_argument("--threshold", type=float, default=0.5, help="Edge threshold on max(P_ij, P_ji).")
    p.add_argument("--min-agreement", type=float, default=0.5)
    p.add_argument("--enforce-dag", action="store_true", help="Post-process for acyclicity.")
    return p.parse_args()


def main():
    args = parse_args()

    for path, label in [(args.checkpoint, "checkpoint"), (args.data, "data file")]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found at {path}", file=sys.stderr)
            if label == "checkpoint":
                print(
                    "The pretrained weights are distributed as a GitHub Release "
                    "asset, not in the git repo. Download them with:\n"
                    "  curl -L -o checkpoint.pt "
                    "https://github.com/amazon-science/foundcause/releases/latest/download/checkpoint.pt",
                    file=sys.stderr,
                )
            sys.exit(1)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    print(f"Loading {args.data} ...")
    X, var_names = load_csv(args.data)
    N, D = X.shape
    print(f"  N={N} samples, D={D} variables")

    if D < 2:
        print("ERROR: need at least 2 variables.", file=sys.stderr)
        sys.exit(1)
    if D > 100:
        warnings.warn(f"FoundCause was trained on D <= 50; D={D} may be unreliable.")

    print(f"Running inference (K={args.n_runs}) ...")
    result = predict(
        model, X, device,
        temperature=args.temperature,
        n_runs=args.n_runs,
        max_samples=args.max_samples,
        threshold=args.threshold,
        min_agreement=args.min_agreement,
    )
    D_pred_bin = result["adjacency"]
    D_pred_prob = result["probabilities"]
    print(f"  Threshold={result['threshold']:.3f}, edges={int(D_pred_bin.sum())}")

    if args.enforce_dag:
        D_pred_bin = enforce_dag(D_pred_bin, D_pred_prob)
        print(f"  After DAG enforcement: {int(D_pred_bin.sum())} edges")

    C_hat = result["C"]

    base = os.path.splitext(os.path.basename(args.data))[0]
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(args.data))
    os.makedirs(out_dir, exist_ok=True)

    for matrix, suffix in [(D_pred_bin, "dag"), (D_pred_prob, "probs"), (C_hat, "confounders")]:
        path = os.path.join(out_dir, f"{base}_{suffix}.csv")
        pd.DataFrame(matrix, index=var_names, columns=var_names).to_csv(path)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
