#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FoundCause inference from a CSV of observational data.

Usage:
    python predict.py --data path/to/data.csv [options]

See README.md for details.
"""

import argparse
import csv
import itertools
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import torch
from pandas.io.common import get_handle

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


_DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")


def _first_lines(path, k=2):
    """First k non-blank lines (a quoted field spanning lines stays in one), read through
    pandas' own opener (same decompression and encoding)."""
    lines, buf, n = [], "", 0
    with get_handle(path, "r", encoding="utf-8-sig", errors="replace", compression="infer") as h:
        for line in h.handle:
            if buf or line.strip():
                buf, n = buf + line, n + 1
                if buf.count('"') % 2 == 0 or n >= 50:  # a stray quote must not swallow the file
                    lines.append(buf.rstrip("\r\n"))
                    buf, n = "", 0
                    if len(lines) == k:
                        break
    return lines or [""]


def _sniff_sep(lines):
    """Delimiter: comma, semicolon, tab or pipe (consistent over the first lines if possible); otherwise whitespace."""
    for sample in ("\n".join(lines), lines[0]):
        try:
            return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
        except csv.Error:
            pass
    return r"\s+" if len(lines[0].split()) > 1 else ","


def _first_records(path, sep, k=2):
    """First k non-empty records as raw strings (quoted fields may span lines)."""
    with get_handle(path, "r", encoding="utf-8-sig", errors="replace", compression="infer") as h:
        rows = ((next(csv.reader([line.strip().replace("\t", " ")], delimiter=" ", skipinitialspace=True), []) for line in h.handle)
                if sep == r"\s+" else csv.reader(h.handle, delimiter=sep, skipinitialspace=True))
        return list(itertools.islice((r for r in rows if any(f.strip() for f in r)), k)) or [[]]


def load_csv(path: str):
    """Load a delimited table with or without a header row.

    Returns (X float32 (n, d) with NaN for missing or non-finite values, column names).
    The delimiter is detected from the first line; the first row is the header if any of
    its cells is not a number, or if it is exactly 0, 1, 2, ... (the default header of
    pandas' DataFrame.to_csv). Raises ValueError for fewer than 2 columns and for a
    non-numeric column (naming it).
    """
    try:
        lines = _first_lines(path)
        sep = _sniff_sep(lines)
        read = dict(sep=sep, na_values=["?"], skipinitialspace=True, encoding="utf-8-sig")
        first = pd.read_csv(path, header=None, nrows=1, dtype=str, **read).iloc[0]
        cells = first.dropna().str.strip().tolist()
        is_number = pd.to_numeric(first, errors="coerce").notna() | first.str.strip().str.lower().isin(["true", "false"])
        has_header = (is_number.sum() < len(cells)
                      or (len(cells) >= 2 and cells == [str(i) for i in range(len(cells))]))  # pandas' default 0,1,2,... header
        records = _first_records(path, sep)
        head, extra, comment = records[0], False, False
        if has_header and len(records) > 1:
            n_head, n_data = len(head), len(records[1])
            if n_head == n_data + 1 and head[0].strip() in ("#", "%"):
                comment = True  # numpy.savetxt / gnuplot style "# a b c" header
            elif n_data == n_head + 1:
                extra = True  # a row index (R write.table) or a trailing delimiter on the data rows
            elif n_data != n_head:
                raise ValueError(f"the header has {n_head} fields but the first data row has {n_data}")
        df = pd.read_csv(path, header=0 if has_header else None, **read)
        if comment:
            df = df.iloc[:, :-1].set_axis(df.columns[1:], axis=1)
            head = head[1:]
        elif extra and records[1][-1] == "" and df.iloc[:, -1].isna().all():  # extra field empty everywhere: a trailing delimiter
            df = pd.read_csv(path, header=0, index_col=False, **read)
    except UnicodeDecodeError:
        raise ValueError(f"could not decode {path}: re-save the file as UTF-8") from None
    except Exception as e:
        raise ValueError(f"could not read {path} as a delimited table ({e})") from None
    if has_header:
        # an empty header cell: a written row index (first column) or a trailing delimiter
        drop = [c for i, c in enumerate(df.columns)
                if i < len(head) and not head[i].strip() and (i == 0 or df[c].isna().all())]
    else:
        df.columns = [f"X{i + 1}" for i in range(df.shape[1])]
        trailing = records[0][-1:] == [""] and df.iloc[:, -1].isna().all()
        drop = [df.columns[-1]] if df.shape[1] > 1 and trailing else []
    if drop:
        print(f"  dropping {len(drop)} column(s) without a header cell (a written row index or a trailing delimiter)")
        df = df.drop(columns=drop)
    if df.shape[1] < 2:
        raise ValueError(f"need at least 2 variables, {path} has {df.shape[1]}")
    cols = []
    for c in df.columns:
        values = pd.to_numeric(df[c], errors="coerce")
        offending = df[c][values.isna() & df[c].notna()]
        if len(offending):
            v = str(offending.iloc[0])
            hint = " (commas inside numbers are not supported; use a decimal point)" if _DECIMAL_COMMA.match(v.strip()) else ""
            raise ValueError(f"column '{c}' is not numeric (first offending value: {v!r}){hint}; "
                             "encode or drop it before running FoundCause")
        cols.append(values.to_numpy(dtype=np.float64))
    X64 = np.column_stack(cols)
    with np.errstate(over="ignore", invalid="ignore"):
        X = X64.astype(np.float32)
    non_finite = ~np.isfinite(X)
    n_new = int(non_finite.sum()) - int(np.isnan(X64).sum())
    if n_new > 0:
        print(f"  {n_new} value(s) are infinite or beyond the float32 range and are treated as missing")
    X[non_finite] = np.nan
    empty = np.isnan(X).all(axis=1)
    if empty.any():
        print(f"  dropping {int(empty.sum())} row(s) without any value")
        X = X[~empty]
    return X, [str(c) for c in df.columns]


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
    try:
        X, var_names = load_csv(args.data)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    N, D = X.shape
    print(f"  N={N} samples, D={D} variables")

    if N < 2 or D < 2:
        print(f"ERROR: need at least 2 rows and 2 variables, got N={N}, D={D}.", file=sys.stderr)
        sys.exit(1)
    if D > 100:
        warnings.warn(f"FoundCause was trained on D <= 50; D={D} may be unreliable.")

    print(f"Running inference (K={args.n_runs}) ...")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="predict: ")  # reported by name below
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
    dups = result["duplicates"]
    removed = [var_names[j] for j in result["dropped"] if j not in dups]
    if removed:
        print(f"  Removed before inference (too few observed values or constant; kept as isolated nodes): {removed}")
    if dups:
        print("  Duplicate columns (get the edges of the column they duplicate): "
              + ", ".join(f"{var_names[t]} = {var_names[r]}" for t, r in dups.items()))

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
