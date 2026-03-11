#!/usr/bin/env python3
"""
run.py — Single entry point to run the full autoresearch pipeline.

Steps:
  1. Check dependencies (ollama, required packages)
  2. Run prepare.py if data/tokenizer not already present
  3. Run the autotuner agent loop (train → eval → adjust → repeat)
  4. Run analyze_results.py and save a final report

Usage:
    python run.py                          # defaults: 20 runs, qwen2.5:72b
    python run.py --max-runs 10            # fewer runs
    python run.py --model llama3.3:70b     # different LLM
    python run.py --num-shards 4           # download only 4 data shards (faster prep)
    python run.py --skip-prepare           # skip data prep if already done
"""

import argparse
import json
import os
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD  = "\033[1m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED   = "\033[31m"
CYAN  = "\033[36m"
RESET = "\033[0m"

def banner(text):
    width = 64
    print(f"\n{BOLD}{CYAN}{'='*width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{BOLD}{CYAN}{'='*width}{RESET}\n")

def step(n, text):
    print(f"{BOLD}{GREEN}[Step {n}]{RESET} {text}")

def warn(text):
    print(f"{YELLOW}[WARN]{RESET} {text}")

def error(text):
    print(f"{RED}[ERROR]{RESET} {text}")

def run(cmd, desc, check=True, capture=False):
    """Run a shell command, streaming output unless capture=True."""
    print(f"  $ {' '.join(cmd)}")
    if capture:
        result = subprocess.run(cmd, capture_output=True, text=True)
    else:
        result = subprocess.run(cmd)
    if check and result.returncode != 0:
        error(f"{desc} failed (exit {result.returncode})")
        if capture:
            print(result.stderr[-2000:])
        sys.exit(1)
    return result

# ---------------------------------------------------------------------------
# Step 1: Dependency check
# ---------------------------------------------------------------------------

def check_dependencies(ollama_model):
    step(1, "Checking dependencies")

    # Python packages
    missing = []
    for pkg in ["ollama", "torch", "tiktoken", "rustbpe", "pyarrow", "requests"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        error(f"Missing Python packages: {', '.join(missing)}")
        print(f"  Install with:  pip install {' '.join(missing)}")
        sys.exit(1)
    print("  Python packages: OK")

    # Ollama daemon
    try:
        import ollama as ol
        ol.list()
        print("  Ollama daemon: OK")
    except Exception as e:
        error(f"Ollama daemon not reachable: {e}")
        print("  Start it with:  ollama serve")
        sys.exit(1)

    # Model pulled
    try:
        import ollama as ol
        pulled = {m.model for m in ol.list().models}
        # model names may have tags like "qwen2.5:72b"
        if not any(ollama_model in p for p in pulled):
            warn(f"Model '{ollama_model}' not found locally. Pulling now...")
            run(["ollama", "pull", ollama_model], f"pull {ollama_model}")
        else:
            print(f"  Ollama model '{ollama_model}': OK")
    except Exception as e:
        warn(f"Could not verify model: {e}. Proceeding anyway.")

    print()

# ---------------------------------------------------------------------------
# Step 2: Data preparation
# ---------------------------------------------------------------------------

CACHE_DIR      = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")
TOKENIZER_PKL  = os.path.join(CACHE_DIR, "tokenizer", "tokenizer.pkl")
TOKEN_BYTES_PT = os.path.join(CACHE_DIR, "tokenizer", "token_bytes.pt")

def is_prepared():
    return os.path.exists(TOKENIZER_PKL) and os.path.exists(TOKEN_BYTES_PT)

def prepare(num_shards, skip):
    step(2, "Data preparation")

    if skip:
        print("  --skip-prepare passed, skipping.")
        if not is_prepared():
            warn("Tokenizer not found. Training will likely fail.")
        print()
        return

    if is_prepared():
        print(f"  Data and tokenizer already present at {CACHE_DIR}")
        print()
        return

    print(f"  Downloading {num_shards} shard(s) and training tokenizer...")
    run(
        [sys.executable, "prepare.py", "--num-shards", str(num_shards)],
        "prepare.py",
    )
    if not is_prepared():
        error("prepare.py finished but tokenizer files not found.")
        sys.exit(1)
    print(f"  Preparation complete.\n")

# ---------------------------------------------------------------------------
# Step 3: Autotuner (train → eval → agent loop)
# ---------------------------------------------------------------------------

def run_autotuner(max_runs, ollama_model, train_script):
    step(3, f"Starting autotuner ({max_runs} runs × ~5 min each ≈ {max_runs*5} min total)")
    print(f"  LLM: {ollama_model}  |  train script: {train_script}\n")

    # Import and run directly (same process) so we get live output
    # and the history object is available afterward.
    from autotuner import AutoTuner
    tuner = AutoTuner(
        train_script=train_script,
        model=ollama_model,
        max_runs=max_runs,
    )
    tuner.run()
    print()
    return tuner  # caller can inspect tuner.history

# ---------------------------------------------------------------------------
# Step 4: Analysis and final report
# ---------------------------------------------------------------------------

def run_analysis(tuner):
    step(4, "Analyzing results")

    history_path = "autotuner_history.json"

    # Write results.tsv that analyze_results.py expects
    tsv_path = "results.tsv"
    _write_results_tsv(tuner.history, tsv_path)
    print(f"  Written: {tsv_path}")

    # Run analyze_results.py if it exists
    analysis_output = ""
    if os.path.exists("analyze_results.py"):
        result = run(
            [sys.executable, "analyze_results.py"],
            "analyze_results.py",
            check=False,
            capture=True,
        )
        analysis_output = result.stdout
        print(analysis_output)
    else:
        warn("analyze_results.py not found — skipping analysis script.")

    # Write the final report
    report_path = _write_report(tuner, analysis_output)
    print(f"\n  Final report: {report_path}")
    print()

def _write_results_tsv(history, path):
    """Write results.tsv in the format analyze_results.py expects."""
    import csv
    rows = []
    for r in history:
        rows.append({
            "commit":           f"run_{r.run_id:03d}",
            "val_bpb":          r.val_bpb if r.val_bpb is not None else "",
            "memory_gb":        round(r.peak_vram_mb / 1024, 2) if r.peak_vram_mb else "",
            "training_seconds": r.training_seconds if r.training_seconds else "",
            "total_tokens_M":   r.total_tokens_M if r.total_tokens_M else "",
            "mfu_percent":      r.mfu_percent if r.mfu_percent else "",
            "num_params_M":     r.num_params_M if r.num_params_M else "",
            "status":           "failed" if r.failed else "ok",
            "timestamp":        r.timestamp,
            # Flatten config keys
            **{f"cfg_{k}": v for k, v in (r.config or {}).items()},
        })
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

def _write_report(tuner, analysis_output):
    """Write a markdown summary report."""
    path = f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"

    successful = [r for r in tuner.history if not r.failed and r.val_bpb is not None]
    failed     = [r for r in tuner.history if r.failed]
    best       = min(successful, key=lambda r: r.val_bpb) if successful else None
    baseline   = successful[0] if successful else None

    lines = [
        "# Autoresearch Run Report",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Summary",
        f"- Total runs: {len(tuner.history)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(failed)}",
    ]

    if baseline and best:
        improvement = baseline.val_bpb - best.val_bpb
        lines += [
            f"- Baseline val_bpb: {baseline.val_bpb:.6f} (run {baseline.run_id})",
            f"- Best val_bpb:     {best.val_bpb:.6f} (run {best.run_id})",
            f"- Improvement:      {improvement:+.6f} ({improvement/baseline.val_bpb*100:+.2f}%)",
        ]

    if best:
        lines += [
            "",
            "## Best Configuration",
            "```json",
            json.dumps(best.config, indent=2),
            "```",
            "",
            f"- params:  {best.num_params_M:.1f}M" if best.num_params_M else "",
            f"- tokens:  {best.total_tokens_M:.1f}M" if best.total_tokens_M else "",
            f"- MFU:     {best.mfu_percent:.1f}%" if best.mfu_percent else "",
            f"- VRAM:    {best.peak_vram_mb:.0f} MB" if best.peak_vram_mb else "",
        ]

    lines += [
        "",
        "## All Runs (sorted by val_bpb)",
        "| run | val_bpb | params_M | tokens_M | mfu% | status |",
        "|-----|---------|----------|----------|------|--------|",
    ]
    def fmt(v, spec):
        return format(v, spec) if v is not None else "-"

    for r in sorted(tuner.history, key=lambda r: (r.val_bpb or 999, r.run_id)):
        marker = " ← best" if best and r.run_id == best.run_id else ""
        status = ("✗ " + r.error_msg[:40]) if r.failed else ("✓" + marker)
        lines.append(
            f"| {r.run_id} "
            f"| {fmt(r.val_bpb, '.6f')} "
            f"| {fmt(r.num_params_M, '.1f')} "
            f"| {fmt(r.total_tokens_M, '.1f')} "
            f"| {fmt(r.mfu_percent, '.1f')} "
            f"| {status} |"
        )

    if analysis_output.strip():
        lines += [
            "",
            "## Analysis Script Output",
            "```",
            analysis_output.strip(),
            "```",
        ]

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return path

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full autoresearch pipeline: prepare → tune → analyze",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                          # 20 runs with default settings
  python run.py --max-runs 5             # quick test with 5 runs (~25 min)
  python run.py --model llama3.3:70b     # use a different LLM
  python run.py --num-shards 4           # download only 4 data shards
  python run.py --skip-prepare           # skip data download (already done)
        """,
    )
    parser.add_argument("--max-runs",     type=int, default=20,
                        help="Number of training runs for the agent (default: 20)")
    parser.add_argument("--model",        default="qwen2.5:72b",
                        help="Ollama model name (default: qwen2.5:72b)")
    parser.add_argument("--num-shards",   type=int, default=10,
                        help="Number of data shards to download for prepare.py (default: 10)")
    parser.add_argument("--skip-prepare", action="store_true",
                        help="Skip prepare.py even if data is missing")
    parser.add_argument("--train-script", default="train.py",
                        help="Path to train.py (default: train.py)")
    args = parser.parse_args()

    banner("Autoresearch Pipeline")
    print(f"  Runs:        {args.max_runs} × ~5 min = ~{args.max_runs * 5} min")
    print(f"  LLM:         {args.model}")
    print(f"  Data shards: {args.num_shards}")
    print(f"  Train script: {args.train_script}")
    print()

    t_total = time.time()

    check_dependencies(args.model)
    prepare(args.num_shards, args.skip_prepare)
    tuner = run_autotuner(args.max_runs, args.model, args.train_script)
    run_analysis(tuner)

    elapsed = time.time() - t_total
    banner(f"Done in {elapsed/60:.1f} min")

    best = min(
        (r for r in tuner.history if not r.failed and r.val_bpb),
        key=lambda r: r.val_bpb,
        default=None,
    )
    if best:
        print(f"  Best val_bpb: {BOLD}{best.val_bpb:.6f}{RESET}  (run {best.run_id})")
    print(f"  History:      autotuner_history.json")
    print(f"  Results:      results.tsv")
    print()

if __name__ == "__main__":
    main()
