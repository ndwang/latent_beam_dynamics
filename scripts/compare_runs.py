#!/usr/bin/env python
"""Compare training runs from a scan.

Usage:
    python scripts/compare_runs.py runs/d_model_scan_*
    python scripts/compare_runs.py runs/d_model_scan_* --sort val_loss
    python scripts/compare_runs.py runs/d_model_scan_* --convergence
    python scripts/compare_runs.py runs/d_model_scan_* --trajectory
    python scripts/compare_runs.py runs/d_model_scan_* --overfitting
    python scripts/compare_runs.py runs/d_model_scan_* --config-diff
    python scripts/compare_runs.py runs/d_model_scan_* --all
"""

import argparse
import csv
import sys
from pathlib import Path

import yaml


# ── Data loading ──────────────────────────────────────────────────────────────

def load_history(run_dir):
    """Load *_history.csv from a run directory. Returns list of row dicts or None."""
    csvs = list(Path(run_dir).glob("*_history.csv"))
    if not csvs:
        return None
    with open(csvs[0]) as f:
        return list(csv.DictReader(f))


def load_config(run_dir):
    """Load config.yaml from a run directory. Returns dict or None."""
    path = Path(run_dir) / "config.yaml"
    if not path.exists():
        return None
    with open(path) as f:
        return yaml.safe_load(f)


def best_epoch(rows, metric="val_loss"):
    return min(rows, key=lambda r: float(r[metric]))


# ── Config diff helpers ───────────────────────────────────────────────────────

_SKIP_KEYS = {"run_name", "output_dir"}


def flatten_config(cfg, prefix=""):
    """Flatten nested dict to dot-notation keys, skipping metadata fields."""
    out = {}
    for k, v in cfg.items():
        full_key = f"{prefix}.{k}" if prefix else k
        if k in _SKIP_KEYS and not prefix:
            continue
        if isinstance(v, dict):
            out.update(flatten_config(v, full_key))
        else:
            out[full_key] = v
    return out


def find_varied_params(flat_configs):
    """Return sorted list of dot-notation keys that differ across configs."""
    all_keys = set()
    for c in flat_configs:
        all_keys |= set(c.keys())
    return sorted(k for k in all_keys
                  if len({str(c.get(k)) for c in flat_configs}) > 1)


# ── Analysis functions ────────────────────────────────────────────────────────

def print_summary(run_dirs, sort_by="val_loss"):
    results = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            print(f"  [skip] {d.name}: no history CSV", file=sys.stderr)
            continue
        best = best_epoch(rows, sort_by)
        results.append((d.name, best, rows))

    if not results:
        print("No valid runs found.")
        return

    results.sort(key=lambda r: float(r[1][sort_by]))

    header = f"{'run':<50} {'epoch':>5}  {'val_loss':>12}  {'train_loss':>12}  {'final_val':>12}"
    print(header)
    print("-" * len(header))
    for name, best, rows in results:
        print(
            f"{name:<50} {int(best['epoch']):>5}"
            f"  {float(best['val_loss']):>12.6f}"
            f"  {float(best['train_loss']):>12.6f}"
            f"  {float(rows[-1]['val_loss']):>12.6f}"
        )


def print_convergence(run_dirs, metric="val_loss"):
    all_best = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is not None:
            all_best.append(float(best_epoch(rows, metric)[metric]))

    if not all_best:
        print("No valid runs found.")
        return

    global_best = min(all_best)
    raw = [global_best * m for m in (10, 5, 2)]
    # Round thresholds to 1 significant figure
    thresholds = sorted(
        {round(t, -int(f"{t:.1e}".split("e")[1]) + 1) for t in raw},
        reverse=True,
    )

    header = f"Convergence: epochs to reach {metric} threshold\n"
    header += f"{'run':<50}"
    for t in thresholds:
        header += f"  {t:<10.2e}"
    header += f"  {'final':>12}"
    print(header)
    print("-" * (len(header.splitlines()[-1])))

    results = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        max_ep = int(rows[-1]["epoch"])
        hits = []
        for t in thresholds:
            hit = next((int(r["epoch"]) for r in rows if float(r[metric]) <= t), None)
            hits.append(hit)
        results.append((d.name, hits, float(rows[-1][metric]), max_ep))

    results.sort(key=lambda r: (r[1][-1] is None, r[1][-1] or 99999))

    for name, hits, final, max_ep in results:
        line = f"{name:<50}"
        for h in hits:
            cell = str(h) if h is not None else f">{max_ep}"
            line += f"  {cell:<10}"
        line += f"  {final:>12.6f}"
        print(line)


def print_overfitting(run_dirs):
    results = []
    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        best = best_epoch(rows)
        results.append((d.name, best))

    if not results:
        print("No valid runs found.")
        return

    results.sort(key=lambda r: float(r[1]["val_loss"]))

    header = (
        f"Overfitting check (train/val ratio at best val epoch)\n"
        f"{'run':<50} {'epoch':>5}  {'train_loss':>12}  {'val_loss':>12}  {'ratio':>7}"
    )
    print(header)
    print("-" * len(header.splitlines()[-1]))
    for name, best in results:
        tv = float(best["train_loss"])
        vv = float(best["val_loss"])
        ratio = tv / vv if vv > 0 else float("inf")
        print(f"{name:<50} {int(best['epoch']):>5}  {tv:>12.6f}  {vv:>12.6f}  {ratio:>7.3f}")


def print_trajectory(run_dirs, epochs=None):
    if epochs is None:
        epochs = [1, 5, 10, 25, 50, 100, 150, 200, 300]

    for d in run_dirs:
        rows = load_history(d)
        if rows is None:
            continue
        max_ep = int(rows[-1]["epoch"])
        by_epoch = {int(r["epoch"]): r for r in rows}

        print(f"=== {d.name} ===")
        print(f"  {'epoch':>5}  {'val_loss':>12}  {'train_loss':>12}")
        for ep in epochs:
            if ep > max_ep:
                break
            r = by_epoch.get(ep)
            if r is None:
                continue
            print(f"  {int(r['epoch']):>5}  {float(r['val_loss']):>12.6f}  {float(r['train_loss']):>12.6f}")
        print()


def print_config_diff(run_dirs):
    flat_configs = []
    names = []
    for d in run_dirs:
        cfg = load_config(d)
        if cfg is None:
            print(f"  [skip] {d.name}: no config.yaml", file=sys.stderr)
            continue
        flat_configs.append(flatten_config(cfg))
        names.append(d.name)

    if not flat_configs:
        print("No configs found.")
        return

    varied = find_varied_params(flat_configs)
    n_fixed = len(set().union(*[set(c.keys()) for c in flat_configs]) - set(varied))

    print(f"Config diff  ({len(varied)} varied, {n_fixed} fixed)\n")

    if not varied:
        print("All configs are identical.")
        return

    key_w = max(len(k) for k in varied)
    col_w = max(max(len(n) for n in names), 10)

    header = f"  {'param':<{key_w}}"
    for n in names:
        header += f"  {n:<{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for key in varied:
        line = f"  {key:<{key_w}}"
        for cfg in flat_configs:
            line += f"  {str(cfg.get(key, '—')):<{col_w}}"
        print(line)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare training runs from a scan",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_dirs", nargs="+", help="Run directories to compare")
    parser.add_argument("--sort", default="val_loss",
                        help="Metric to sort summary by (default: val_loss)")
    parser.add_argument("--convergence", action="store_true")
    parser.add_argument("--trajectory", action="store_true",
                        help="Show loss at key epochs")
    parser.add_argument("--overfitting", action="store_true",
                        help="Show train/val ratio at best epoch")
    parser.add_argument("--config-diff", action="store_true",
                        help="Show which config params vary across runs")
    parser.add_argument("--all", action="store_true", help="Run all analyses")
    args = parser.parse_args()

    run_dirs = sorted(
        [Path(d) for d in args.run_dirs if Path(d).is_dir()],
        key=lambda d: d.name,
    )
    if not run_dirs:
        print("No valid run directories found.")
        sys.exit(1)

    print(f"Comparing {len(run_dirs)} runs\n")

    explicit = args.convergence or args.trajectory or args.overfitting or args.config_diff
    show_all = args.all
    show_summary = show_all or not explicit

    if show_summary:
        print_summary(run_dirs, sort_by=args.sort)
        print()

    if args.convergence or show_all:
        print_convergence(run_dirs)
        print()

    if args.overfitting or show_all:
        print_overfitting(run_dirs)
        print()

    if args.trajectory or show_all:
        print_trajectory(run_dirs)

    if args.config_diff or show_all:
        print_config_diff(run_dirs)


if __name__ == "__main__":
    main()
