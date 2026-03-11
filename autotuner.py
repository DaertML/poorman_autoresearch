"""
autotuner.py — LLM-driven neural network hyperparameter optimization agent.

Uses Ollama to call a local LLM that iteratively:
  1. Reviews training history and previous results
  2. Calls tools to modify hyperparameters / architecture
  3. Triggers a training run (each run lasts TIME_BUDGET=300s, set in prepare.py)
  4. Evaluates results (val_bpb)
  5. Loops, trying to minimize val_bpb

The agent ONLY modifies the hyperparameter section of train.py.
prepare.py and analyze_results.py are never touched.

Requirements:
    pip install ollama

Usage:
    ollama pull qwen2.5:72b   # or any model with good tool-use
    python autotuner.py
"""

import json
import math
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict, field
from typing import Any

import ollama
import signal

# ---------------------------------------------------------------------------
# Ollama VRAM management — kill before training, restart after
# ---------------------------------------------------------------------------

def _ollama_stop():
    """Kill the ollama serve process to free VRAM before a training run."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "ollama serve"],
            capture_output=True, text=True
        )
        pids = result.stdout.strip().split()
        if pids:
            for pid in pids:
                try:
                    os.kill(int(pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
            time.sleep(2)  # give it time to release VRAM
            print("  [VRAM] Ollama stopped — VRAM freed for training.")
        else:
            print("  [VRAM] Ollama not running (already stopped).")
    except Exception as e:
        print(f"  [VRAM] Warning: could not stop ollama: {e}")


def _ollama_start():
    """Restart ollama serve in the background after training completes."""
    try:
        # Check if already running
        result = subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True, text=True)
        if result.stdout.strip():
            print("  [VRAM] Ollama already running.")
            return
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait until the API is responsive
        import urllib.request
        for _ in range(30):
            try:
                urllib.request.urlopen("http://localhost:11434/api/tags", timeout=1)
                print("  [VRAM] Ollama restarted and ready.")
                return
            except Exception:
                time.sleep(1)
        print("  [VRAM] Warning: ollama may not have started correctly.")
    except Exception as e:
        print(f"  [VRAM] Warning: could not start ollama: {e}")


# ---------------------------------------------------------------------------
# Constants mirrored from prepare.py (do not modify — owned by prepare.py)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300   # seconds per training run, fixed in prepare.py
MAX_SEQ_LEN = 2048  # context length, fixed in prepare.py

# ---------------------------------------------------------------------------
# Default hyperparameter configuration
# ---------------------------------------------------------------------------

@dataclass
class HyperparamConfig:
    # Model architecture
    depth: int = 8
    aspect_ratio: int = 64          # model_dim = depth * aspect_ratio
    head_dim: int = 128
    window_pattern: str = "SSSL"    # L=full context, S=half context

    # Training batch
    total_batch_size: int = 524288  # 2**19 tokens per optimizer step
    device_batch_size: int = 8    # conservative default for 24GB GPU without torch.compile

    # Learning rates
    embedding_lr: float = 0.6
    unembedding_lr: float = 0.004
    matrix_lr: float = 0.04
    scalar_lr: float = 0.5

    # Optimizer
    weight_decay: float = 0.2
    adam_beta1: float = 0.8
    adam_beta2: float = 0.95

    # Schedule
    warmup_ratio: float = 0.0
    warmdown_ratio: float = 0.5
    final_lr_frac: float = 0.0

    def to_env(self) -> dict[str, str]:
        """Convert to environment variables for train.py."""
        return {
            "DEPTH": str(self.depth),
            "ASPECT_RATIO": str(self.aspect_ratio),
            "HEAD_DIM": str(self.head_dim),
            "WINDOW_PATTERN": self.window_pattern,
            "TOTAL_BATCH_SIZE": str(self.total_batch_size),
            "DEVICE_BATCH_SIZE": str(self.device_batch_size),
            "EMBEDDING_LR": str(self.embedding_lr),
            "UNEMBEDDING_LR": str(self.unembedding_lr),
            "MATRIX_LR": str(self.matrix_lr),
            "SCALAR_LR": str(self.scalar_lr),
            "WEIGHT_DECAY": str(self.weight_decay),
            "ADAM_BETA1": str(self.adam_beta1),
            "ADAM_BETA2": str(self.adam_beta2),
            "WARMUP_RATIO": str(self.warmup_ratio),
            "WARMDOWN_RATIO": str(self.warmdown_ratio),
            "FINAL_LR_FRAC": str(self.final_lr_frac),
        }


# ---------------------------------------------------------------------------
# Result tracking
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    run_id: int
    config: dict
    val_bpb: float | None
    training_seconds: float | None
    total_tokens_M: float | None
    mfu_percent: float | None
    peak_vram_mb: float | None
    num_params_M: float | None
    failed: bool = False
    error_msg: str = ""
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def summary(self) -> str:
        if self.failed:
            return f"Run {self.run_id}: FAILED — {self.error_msg}"
        return (
            f"Run {self.run_id}: val_bpb={self.val_bpb:.6f} | "
            f"params={self.num_params_M:.1f}M | tokens={self.total_tokens_M:.1f}M | "
            f"mfu={self.mfu_percent:.1f}% | vram={self.peak_vram_mb:.0f}MB | "
            f"time={self.training_seconds:.0f}s"
        )


# ---------------------------------------------------------------------------
# Tool definitions (for Ollama function calling)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "set_architecture",
            "description": (
                "Set the transformer architecture hyperparameters. "
                "model_dim = depth * aspect_ratio, rounded up to nearest multiple of head_dim. "
                "Larger depth/aspect_ratio → bigger model, more compute. "
                "window_pattern controls attention window per layer: "
                "  L = full sequence length (expensive but long-range), "
                "  S = half sequence (cheaper, local). "
                "Pattern repeats cyclically. Last layer is always L."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "depth": {
                        "type": "integer",
                        "description": "Number of transformer layers. Typical range: 4–24.",
                    },
                    "aspect_ratio": {
                        "type": "integer",
                        "description": "model_dim = depth * aspect_ratio. Typical range: 32–128.",
                    },
                    "head_dim": {
                        "type": "integer",
                        "description": "Attention head dimension. Must be power of 2. Common: 64, 128.",
                    },
                    "window_pattern": {
                        "type": "string",
                        "description": (
                            "Attention window pattern string using L (full) and S (short). "
                            "e.g. 'SSSL', 'SSL', 'SSSSL', 'L', 'SLSL'."
                        ),
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_learning_rates",
            "description": (
                "Set learning rates for different parameter groups. "
                "The model uses different optimizers: "
                "  - matrix params (attention/MLP weights) → Muon optimizer, use matrix_lr "
                "  - token embeddings → AdamW, use embedding_lr "
                "  - lm_head unembedding → AdamW, use unembedding_lr "
                "  - per-layer scalars (resid_lambdas, x0_lambdas) → AdamW, use scalar_lr "
                "LRs are auto-scaled by 1/sqrt(model_dim/768). "
                "Typical ranges: matrix_lr 0.01–0.1, embedding_lr 0.1–1.0."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "matrix_lr": {"type": "number", "description": "LR for matrix params (Muon). Typical: 0.01–0.1."},
                    "embedding_lr": {"type": "number", "description": "LR for token embeddings. Typical: 0.1–1.0."},
                    "unembedding_lr": {"type": "number", "description": "LR for lm_head. Typical: 0.001–0.01."},
                    "scalar_lr": {"type": "number", "description": "LR for per-layer scalars. Typical: 0.1–1.0."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_optimizer_params",
            "description": (
                "Set optimizer hyperparameters. "
                "adam_beta1 controls momentum decay (lower = more reactive). "
                "adam_beta2 controls second moment decay. "
                "weight_decay applies only to Muon matrix params, decayed toward zero. "
                "Higher weight_decay can help generalization but too much hurts training loss."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "weight_decay": {"type": "number", "description": "Weight decay for Muon. Typical: 0.0–0.5."},
                    "adam_beta1": {"type": "number", "description": "Adam beta1. Typical: 0.8–0.95."},
                    "adam_beta2": {"type": "number", "description": "Adam beta2. Typical: 0.90–0.999."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_lr_schedule",
            "description": (
                "Set the learning rate schedule shape. "
                "The schedule has 3 phases over the total time budget: "
                "  1. Warmup: LR linearly increases from 0 to peak over warmup_ratio * time "
                "  2. Constant: LR stays at peak "
                "  3. Warmdown/cooldown: LR linearly decays to final_lr_frac * peak over warmdown_ratio * time "
                "warmdown_ratio=0.5 means last 50% of training is cooldown. "
                "Longer warmdown often improves final val_bpb."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "warmup_ratio": {"type": "number", "description": "Fraction of time for warmup. Typical: 0.0–0.1."},
                    "warmdown_ratio": {"type": "number", "description": "Fraction of time for warmdown. Typical: 0.2–0.7."},
                    "final_lr_frac": {"type": "number", "description": "Final LR as fraction of initial. Typical: 0.0–0.1."},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_batch_size",
            "description": (
                "Set batch sizes. total_batch_size is the global token count per optimizer step. "
                "device_batch_size is per-GPU microbatch (reduce if OOM). "
                "total_batch_size / device_batch_size / seq_len = grad_accum_steps. "
                "total_batch_size must be divisible by (device_batch_size * seq_len). "
                "Larger total_batch_size = fewer steps = may need higher LR."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "total_batch_size": {
                        "type": "integer",
                        "description": "Total tokens per optimizer step. Must be power of 2 times seq_len. Typical: 2^17 to 2^21.",
                    },
                    "device_batch_size": {
                        "type": "integer",
                        "description": "Per-GPU microbatch size. Reduce if OOM. Typical: 32–256.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_training",
            "description": (
                "Launch a training run with the current configuration and return results. "
                "This modifies train.py to inject the current hyperparameters, "
                "then runs it and parses the output. "
                "Returns val_bpb (bits per byte on validation set — lower is better), "
                "training throughput, MFU, VRAM usage, and parameter count. "
                "ALWAYS call this after setting hyperparameters to actually measure performance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {
                        "type": "string",
                        "description": "Optional human-readable note about what this run is testing.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_history",
            "description": (
                "Return the full history of all training runs, sorted by val_bpb. "
                "Use this to understand what has been tried and what worked best. "
                "Shows config diffs from the baseline for each run."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Show only the top N runs by val_bpb. Default: all.",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_config",
            "description": "Return the current pending hyperparameter configuration (not yet trained).",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_to_best",
            "description": (
                "Reset the current configuration to the best run seen so far. "
                "Useful to start a new search branch from the best known point."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "reset_to_default",
            "description": "Reset the current configuration to the original defaults.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------

class AutoTuner:
    def __init__(self, train_script: str = "train.py", model: str = "qwen2.5:72b", max_runs: int = 20):
        self.train_script = train_script
        self.model = model
        self.max_runs = max_runs
        self.default_config = HyperparamConfig()
        self.current_config = deepcopy(self.default_config)
        self.history: list[RunResult] = []
        self.run_counter = 0

    # ---- Tool implementations ----

    def tool_set_architecture(self, depth=None, aspect_ratio=None, head_dim=None, window_pattern=None):
        changes = []
        if depth is not None:
            self.current_config.depth = int(depth)
            changes.append(f"depth={depth}")
        if aspect_ratio is not None:
            self.current_config.aspect_ratio = int(aspect_ratio)
            changes.append(f"aspect_ratio={aspect_ratio}")
        if head_dim is not None:
            self.current_config.head_dim = int(head_dim)
            changes.append(f"head_dim={head_dim}")
        if window_pattern is not None:
            pat = str(window_pattern).upper()
            assert all(c in "SL" for c in pat), "window_pattern must only contain S and L"
            self.current_config.window_pattern = pat
            changes.append(f"window_pattern={pat}")
        # Compute derived dims for feedback
        base_dim = self.current_config.depth * self.current_config.aspect_ratio
        hd = self.current_config.head_dim
        model_dim = ((base_dim + hd - 1) // hd) * hd
        num_heads = model_dim // hd
        return {
            "status": "ok",
            "changes": changes,
            "derived": {
                "model_dim": model_dim,
                "num_heads": num_heads,
                "approx_params_M": round(12 * model_dim**2 * self.current_config.depth / 1e6, 1),
            },
        }

    def tool_set_learning_rates(self, matrix_lr=None, embedding_lr=None, unembedding_lr=None, scalar_lr=None):
        changes = []
        if matrix_lr is not None:
            self.current_config.matrix_lr = float(matrix_lr); changes.append(f"matrix_lr={matrix_lr}")
        if embedding_lr is not None:
            self.current_config.embedding_lr = float(embedding_lr); changes.append(f"embedding_lr={embedding_lr}")
        if unembedding_lr is not None:
            self.current_config.unembedding_lr = float(unembedding_lr); changes.append(f"unembedding_lr={unembedding_lr}")
        if scalar_lr is not None:
            self.current_config.scalar_lr = float(scalar_lr); changes.append(f"scalar_lr={scalar_lr}")
        return {"status": "ok", "changes": changes}

    def tool_set_optimizer_params(self, weight_decay=None, adam_beta1=None, adam_beta2=None):
        changes = []
        if weight_decay is not None:
            self.current_config.weight_decay = float(weight_decay); changes.append(f"weight_decay={weight_decay}")
        if adam_beta1 is not None:
            self.current_config.adam_beta1 = float(adam_beta1); changes.append(f"adam_beta1={adam_beta1}")
        if adam_beta2 is not None:
            self.current_config.adam_beta2 = float(adam_beta2); changes.append(f"adam_beta2={adam_beta2}")
        return {"status": "ok", "changes": changes}

    def tool_set_lr_schedule(self, warmup_ratio=None, warmdown_ratio=None, final_lr_frac=None):
        changes = []
        if warmup_ratio is not None:
            self.current_config.warmup_ratio = float(warmup_ratio); changes.append(f"warmup_ratio={warmup_ratio}")
        if warmdown_ratio is not None:
            self.current_config.warmdown_ratio = float(warmdown_ratio); changes.append(f"warmdown_ratio={warmdown_ratio}")
        if final_lr_frac is not None:
            self.current_config.final_lr_frac = float(final_lr_frac); changes.append(f"final_lr_frac={final_lr_frac}")
        return {"status": "ok", "changes": changes}

    def tool_set_batch_size(self, total_batch_size=None, device_batch_size=None):
        changes = []
        if total_batch_size is not None:
            self.current_config.total_batch_size = int(total_batch_size); changes.append(f"total_batch_size={total_batch_size}")
        if device_batch_size is not None:
            self.current_config.device_batch_size = int(device_batch_size); changes.append(f"device_batch_size={device_batch_size}")
        return {"status": "ok", "changes": changes}

    def tool_run_training(self, note=""):
        self.run_counter += 1
        run_id = self.run_counter
        config_snapshot = asdict(self.current_config)
        print(f"\n{'='*60}")
        print(f"  TRAINING RUN #{run_id}" + (f" — {note}" if note else ""))
        print(f"{'='*60}")
        print(f"  Config: {config_snapshot}")
        print(f"{'='*60}\n")

        result = self._execute_training(run_id, config_snapshot)
        self.history.append(result)
        self._save_history()

        if result.failed:
            return {"status": "failed", "run_id": run_id, "error": result.error_msg}

        best = self._best_run()
        is_best = best and best.run_id == run_id
        return {
            "status": "success",
            "run_id": run_id,
            "val_bpb": result.val_bpb,
            "training_seconds": result.training_seconds,
            "total_tokens_M": result.total_tokens_M,
            "mfu_percent": result.mfu_percent,
            "peak_vram_mb": result.peak_vram_mb,
            "num_params_M": result.num_params_M,
            "is_new_best": is_best,
            "best_val_bpb_so_far": best.val_bpb if best else None,
        }

    def tool_get_history(self, top_n=None):
        if not self.history:
            return {"runs": [], "message": "No runs yet."}
        
        default_dict = asdict(self.default_config)
        
        def format_run(r: RunResult):
            if r.failed:
                return {"run_id": r.run_id, "failed": True, "error": r.error_msg}
            diffs = {k: v for k, v in r.config.items() if v != default_dict.get(k)}
            return {
                "run_id": r.run_id,
                "val_bpb": r.val_bpb,
                "num_params_M": r.num_params_M,
                "mfu_percent": r.mfu_percent,
                "config_diffs_from_default": diffs,
                "timestamp": r.timestamp,
            }

        successful = sorted(
            [r for r in self.history if not r.failed],
            key=lambda r: r.val_bpb,
        )
        failed = [r for r in self.history if r.failed]
        
        if top_n:
            successful = successful[:top_n]
        
        return {
            "best_val_bpb": successful[0].val_bpb if successful else None,
            "total_runs": len(self.history),
            "successful_runs": len(successful),
            "failed_runs": len(failed),
            "runs_by_bpb": [format_run(r) for r in successful],
            "failed_runs_detail": [format_run(r) for r in failed],
        }

    def tool_get_current_config(self):
        return {"current_config": asdict(self.current_config)}

    def tool_reset_to_best(self):
        best = self._best_run()
        if not best:
            return {"status": "error", "message": "No successful runs yet."}
        for k, v in best.config.items():
            setattr(self.current_config, k, v)
        return {
            "status": "ok",
            "reset_to_run": best.run_id,
            "val_bpb": best.val_bpb,
            "config": best.config,
        }

    def tool_reset_to_default(self):
        self.current_config = deepcopy(self.default_config)
        return {"status": "ok", "config": asdict(self.current_config)}

    # ---- Tool dispatch ----

    def dispatch_tool(self, name: str, args: dict) -> Any:
        dispatch = {
            "set_architecture": self.tool_set_architecture,
            "set_learning_rates": self.tool_set_learning_rates,
            "set_optimizer_params": self.tool_set_optimizer_params,
            "set_lr_schedule": self.tool_set_lr_schedule,
            "set_batch_size": self.tool_set_batch_size,
            "run_training": self.tool_run_training,
            "get_history": self.tool_get_history,
            "get_current_config": self.tool_get_current_config,
            "reset_to_best": self.tool_reset_to_best,
            "reset_to_default": self.tool_reset_to_default,
        }
        fn = dispatch.get(name)
        if fn is None:
            return {"error": f"Unknown tool: {name}"}
        try:
            return fn(**args)
        except Exception as e:
            return {"error": str(e)}

    # ---- Training execution ----

    def _execute_training(self, run_id: int, config_snapshot: dict) -> RunResult:
        """
        Patch train.py with the current config values and run it.
        We inject values by rewriting the hyperparameter section.
        """
        import os
        import re

        try:
            with open(self.train_script, "r") as f:
                source = f.read()
        except FileNotFoundError:
            return RunResult(
                run_id=run_id, config=config_snapshot,
                val_bpb=None, training_seconds=None, total_tokens_M=None,
                mfu_percent=None, peak_vram_mb=None, num_params_M=None,
                failed=True, error_msg=f"train.py not found at {self.train_script}",
            )

        # Patch each hyperparameter variable individually using precise regex.
        # This only touches the assignment lines — comments, structure, and
        # everything else in train.py (including prepare.py imports) is untouched.
        cfg = self.current_config
        substitutions = [
            # (variable_name, new_value_string)
            ("ASPECT_RATIO",      str(cfg.aspect_ratio)),
            ("HEAD_DIM",          str(cfg.head_dim)),
            ("WINDOW_PATTERN",    f'"{cfg.window_pattern}"'),
            ("TOTAL_BATCH_SIZE",  str(cfg.total_batch_size)),
            ("EMBEDDING_LR",      str(cfg.embedding_lr)),
            ("UNEMBEDDING_LR",    str(cfg.unembedding_lr)),
            ("MATRIX_LR",         str(cfg.matrix_lr)),
            ("SCALAR_LR",         str(cfg.scalar_lr)),
            ("WEIGHT_DECAY",      str(cfg.weight_decay)),
            ("ADAM_BETAS",        f"({cfg.adam_beta1}, {cfg.adam_beta2})"),
            ("WARMUP_RATIO",      str(cfg.warmup_ratio)),
            ("WARMDOWN_RATIO",    str(cfg.warmdown_ratio)),
            ("FINAL_LR_FRAC",     str(cfg.final_lr_frac)),
            ("DEPTH",             str(cfg.depth)),
            ("DEVICE_BATCH_SIZE", str(cfg.device_batch_size)),
        ]
        new_source = source
        for var, val in substitutions:
            # Match: VAR_NAME = <anything> # optional comment
            # Replace only the value, preserve the comment
            new_source = re.sub(
                rf"^({var}\s*=\s*)([^\s#][^\n]*?)(\s*(#[^\n]*)?)$",
                lambda m, v=val: f"{m.group(1)}{v}{m.group(3)}",
                new_source,
                flags=re.MULTILINE,
            )

        patched_path = f"/tmp/train_run_{run_id}.py"
        project_dir = os.path.dirname(os.path.abspath(self.train_script))
        # Prepend sys.path injection so prepare.py is always importable
        # regardless of the working directory the script is launched from.
        _proj = project_dir  # captured for f-string below
        path_injection = (
            f"import sys as _sys\n_sys.path.insert(0, {_proj!r})\n"
            "import os as _os\n"
            "_os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'\n"
            "import torch._dynamo\n"
            "torch._dynamo.config.suppress_errors = True\n\n"
        )
        # Disable torch.compile — FA3 custom kernel is not traceable by dynamo.
        # FA3 returns a tuple (out, softmax_lse) in eager mode; fix the call site too.
        _compile_decorator = '@torch.compile(dynamic=False, fullgraph=True)'
        _eager_adamw_src = (
            "def adamw_step_fused(p, grad, exp_avg, exp_avg_sq, step_t, lr_t, beta1_t, beta2_t, eps_t, wd_t):\n"
            "    lr=lr_t.item(); b1=beta1_t.item(); b2=beta2_t.item()\n"
            "    eps=eps_t.item(); wd=wd_t.item(); step=int(step_t.item())\n"
            "    p.mul_(1 - lr * wd)\n"
            "    exp_avg.lerp_(grad.to(p.dtype), 1 - b1)\n"
            "    exp_avg_sq.lerp_(grad.to(p.dtype).square(), 1 - b2)\n"
            "    denom = (exp_avg_sq / (1 - b2**step)).sqrt() + eps\n"
            "    p.add_(exp_avg / (1 - b1**step) / denom, alpha=-lr)\n"
        )
        patched_source = new_source.replace(
            'model = torch.compile(model, dynamic=False)',
            'pass  # torch.compile disabled: FA3 kernel not dynamo-traceable',
        ).replace(
            'y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)',
            'y = fa3.flash_attn_func(q, k, v, causal=True, window_size=window_size)[0]',
        ).replace(
            _compile_decorator + '\ndef adamw_step_fused',
            'def adamw_step_fused',
        ).replace(
            _compile_decorator + '\ndef muon_step_fused',
            'def muon_step_fused',
        )
        # Replace adamw_step_fused and muon_step_fused with eager-safe versions.
        _eager_muon_src = (
            "def muon_step_fused(stacked_grads, stacked_params, momentum_buffer, second_momentum_buffer,\n"
            "                    momentum_t, lr_t, wd_t, beta2_t, ns_steps, red_dim):\n"
            "    momentum = momentum_t.item()\n"
            "    momentum_buffer.lerp_(stacked_grads, 1 - momentum)\n"
            "    g = stacked_grads.lerp_(momentum_buffer, momentum)\n"
            "    X = g.bfloat16()\n"
            "    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)\n"
            "    if g.size(-2) > g.size(-1):\n"
            "        for a, b, c in polar_express_coeffs[:ns_steps]:\n"
            "            A = X.mT @ X\n"
            "            B = b * A + c * (A @ A)\n"
            "            X = a * X + X @ B\n"
            "    else:\n"
            "        for a, b, c in polar_express_coeffs[:ns_steps]:\n"
            "            A = X @ X.mT\n"
            "            B = b * A + c * (A @ A)\n"
            "            X = a * X + B @ X\n"
            "    g = X\n"
            "    beta2 = beta2_t.item()\n"
            "    v_mean = g.float().square().mean(dim=red_dim, keepdim=True)\n"
            "    red_dim_size = g.size(red_dim)\n"
            "    v_norm_sq = v_mean.sum(dim=(-2, -1), keepdim=True) * red_dim_size\n"
            "    v_norm = v_norm_sq.sqrt()\n"
            "    second_momentum_buffer.lerp_(v_mean.float(), 1 - beta2)\n"
            "    step_size = second_momentum_buffer.clamp_min(1e-10).rsqrt()\n"
            "    scaled_sq_sum = (v_mean * red_dim_size) * step_size.square()\n"
            "    v_norm_new = scaled_sq_sum.sum(dim=(-2, -1), keepdim=True).sqrt()\n"
            "    final_scale = step_size * (v_norm / v_norm_new.clamp_min(1e-10))\n"
            "    g = g * final_scale.to(g.dtype)\n"
            "    lr = lr_t.item()\n"
            "    wd = wd_t.item()\n"
            "    mask = (g * stacked_params) >= 0\n"
            "    stacked_params.sub_(lr * g + lr * wd * stacked_params * mask)\n"
        )
        import re as _re
        patched_source = _re.sub(
            r'def adamw_step_fused\(.*?\n(?:    .*\n)*',
            _eager_adamw_src,
            patched_source,
        )
        patched_source = _re.sub(
            r'def muon_step_fused\(.*?\n(?:(?:    |\n).*\n)*',
            _eager_muon_src,
            patched_source,
        )
        with open(patched_path, "w") as f:
            f.write(path_injection + patched_source)

        # Stop ollama to free VRAM, run training, then restart ollama
        _ollama_stop()
        try:
            env = os.environ.copy()
            proc = subprocess.run(
                [sys.executable, patched_path],
                capture_output=True, text=True, timeout=7200, env=env,
                cwd=project_dir,
            )
            output = proc.stdout + "\n" + proc.stderr
        except subprocess.TimeoutExpired:
            _ollama_start()
            return RunResult(
                run_id=run_id, config=config_snapshot,
                val_bpb=None, training_seconds=None, total_tokens_M=None,
                mfu_percent=None, peak_vram_mb=None, num_params_M=None,
                failed=True, error_msg="Training timed out after 2 hours.",
            )
        except Exception as e:
            _ollama_start()
            return RunResult(
                run_id=run_id, config=config_snapshot,
                val_bpb=None, training_seconds=None, total_tokens_M=None,
                mfu_percent=None, peak_vram_mb=None, num_params_M=None,
                failed=True, error_msg=str(e),
            )
        else:
            _ollama_start()

        if proc.returncode != 0:
            # Print full stderr so the user can see the actual error
            print(f"\n{'!'*60}")
            print(f"  TRAINING RUN #{run_id} FAILED (exit {proc.returncode})")
            print(f"{'!'*60}")
            print(proc.stderr)
            print(f"{'!'*60}\n")
            # Also save full output to a log file for inspection
            log_path = f"/tmp/train_run_{run_id}.log"
            with open(log_path, "w") as f:
                f.write("=== STDOUT ===\n")
                f.write(proc.stdout)
                f.write("\n=== STDERR ===\n")
                f.write(proc.stderr)
            print(f"  Full log saved to: {log_path}\n")
            # Give the agent a concise but complete error (last 3000 chars of stderr)
            error_detail = proc.stderr.strip()
            # Extract just the traceback — last Exception line is most useful
            lines = error_detail.splitlines()
            short = "\n".join(lines[-30:]) if len(lines) > 30 else error_detail
            return RunResult(
                run_id=run_id, config=config_snapshot,
                val_bpb=None, training_seconds=None, total_tokens_M=None,
                mfu_percent=None, peak_vram_mb=None, num_params_M=None,
                failed=True,
                error_msg=f"Exit code {proc.returncode}. Last 30 lines of stderr:\n{short}",
            )

        # Parse structured output from the final "---" block
        metrics = self._parse_output(output)
        if "val_bpb" not in metrics:
            return RunResult(
                run_id=run_id, config=config_snapshot,
                val_bpb=None, training_seconds=None, total_tokens_M=None,
                mfu_percent=None, peak_vram_mb=None, num_params_M=None,
                failed=True,
                error_msg=f"Could not parse val_bpb from output.\nLast output:\n{output[-2000:]}",
            )

        return RunResult(
            run_id=run_id,
            config=config_snapshot,
            val_bpb=metrics.get("val_bpb"),
            training_seconds=metrics.get("training_seconds"),
            total_tokens_M=metrics.get("total_tokens_M"),
            mfu_percent=metrics.get("mfu_percent"),
            peak_vram_mb=metrics.get("peak_vram_mb"),
            num_params_M=metrics.get("num_params_M"),
        )

    def _parse_output(self, output: str) -> dict:
        metrics = {}
        for line in output.split("\n"):
            line = line.strip()
            if ":" in line:
                key, _, val = line.partition(":")
                key = key.strip()
                val = val.strip()
                if key == "val_bpb":
                    try: metrics["val_bpb"] = float(val)
                    except: pass
                elif key == "training_seconds":
                    try: metrics["training_seconds"] = float(val)
                    except: pass
                elif key == "total_tokens_M":
                    try: metrics["total_tokens_M"] = float(val)
                    except: pass
                elif key == "mfu_percent":
                    try: metrics["mfu_percent"] = float(val)
                    except: pass
                elif key == "peak_vram_mb":
                    try: metrics["peak_vram_mb"] = float(val)
                    except: pass
                elif key == "num_params_M":
                    try: metrics["num_params_M"] = float(val)
                    except: pass
        return metrics

    def _best_run(self) -> RunResult | None:
        successful = [r for r in self.history if not r.failed and r.val_bpb is not None]
        if not successful:
            return None
        return min(successful, key=lambda r: r.val_bpb)

    def _save_history(self):
        path = "autotuner_history.json"
        with open(path, "w") as f:
            json.dump(
                [
                    {
                        "run_id": r.run_id,
                        "config": r.config,
                        "val_bpb": r.val_bpb,
                        "training_seconds": r.training_seconds,
                        "total_tokens_M": r.total_tokens_M,
                        "mfu_percent": r.mfu_percent,
                        "peak_vram_mb": r.peak_vram_mb,
                        "num_params_M": r.num_params_M,
                        "failed": r.failed,
                        "error_msg": r.error_msg,
                        "timestamp": r.timestamp,
                    }
                    for r in self.history
                ],
                f,
                indent=2,
            )

    # ---- Agent loop ----

    def build_system_prompt(self) -> str:
        return f"""You are an expert ML researcher optimizing a GPT-style language model for minimum bits-per-byte (val_bpb) on a validation set.

## Your objective
Minimize val_bpb by iteratively adjusting hyperparameters and running training experiments. You have a budget of {self.max_runs} total training runs — use them wisely. Each run takes exactly {TIME_BUDGET} seconds (5 minutes), fixed by prepare.py.

## Fixed constants (do NOT try to change these — they are owned by prepare.py)
- TIME_BUDGET = {TIME_BUDGET}s per run
- MAX_SEQ_LEN = {MAX_SEQ_LEN} tokens (context length)
- vocab_size = 8192
- Validation set = fixed shard, always the same

## What you CAN tune (via tools)
- Architecture: depth, aspect_ratio, head_dim, window_pattern
- Learning rates: matrix_lr, embedding_lr, unembedding_lr, scalar_lr
- Optimizer: weight_decay, adam_beta1, adam_beta2
- LR schedule: warmup_ratio, warmdown_ratio, final_lr_frac
- Batch size: total_batch_size, device_batch_size

## The model
- GPT with RoPE, RMSNorm, squared-ReLU MLP, sliding-window attention (FlashAttention 3)
- Value residual connections (ResFormer-style): every other layer has value embeddings
- Two-optimizer setup: Muon for matrix params, AdamW for embeddings/scalars
- NorMuon: variance-normalized Muon update with Polar Express orthogonalization
- Training on a fixed TIME_BUDGET (set in prepare.py); more efficient = more tokens

## Key insight on compute
model_dim = depth * aspect_ratio (rounded to nearest head_dim multiple).
Bigger model → better per-token loss but fewer tokens trained in the time budget.
With only {TIME_BUDGET}s per run, SMALLER models often win because they see more tokens.
Rough rule: for short budgets, prefer wide-and-shallow over deep-and-narrow.

## Search strategy
1. Start with a baseline run to calibrate.
2. Explore architecture (depth, aspect_ratio) — biggest impact on compute/quality tradeoff.
3. Tune learning rates (matrix_lr is most sensitive for Muon).
4. Refine schedule (warmdown_ratio strongly affects final val_bpb — try 0.4–0.6).
5. Fine-tune optimizer params if time permits.
6. Always call get_history() to understand what has been tried.

## Important constraints
- total_batch_size must divide evenly by (device_batch_size * {MAX_SEQ_LEN}).
- window_pattern must only use characters S and L.
- head_dim must be a power of 2 (64 or 128 recommended).
- If peak_vram_mb approaches GPU limit (~80000 for H100), reduce device_batch_size or model size.

## Scoring
val_bpb lower = better. Improvements of 0.01 bpb are meaningful. 0.001 is noise.

Always reason step by step about WHY you're making a change before calling tools. Be systematic and data-driven."""

    def run(self):
        print(f"AutoTuner starting. Model: {self.model} | Max runs: {self.max_runs}")
        print(f"Default config: {asdict(self.default_config)}\n")

        messages = [
            {
                "role": "user",
                "content": (
                    "Please begin optimizing the neural network training script. "
                    f"You have a budget of {self.max_runs} training runs. "
                    "Start by running the default configuration to get a baseline, "
                    "then systematically explore improvements. "
                    "Think carefully about each change and explain your reasoning."
                ),
            }
        ]

        system = self.build_system_prompt()

        while self.run_counter < self.max_runs:
            print(f"\n[Agent thinking... runs used: {self.run_counter}/{self.max_runs}]")

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS,
                options={"temperature": 0.3, "num_ctx": 16384},
                keep_alive="30m",
            )

            msg = response.message
            messages.append({"role": "assistant", "content": msg.content or "", "tool_calls": msg.tool_calls or []})

            if msg.content:
                print(f"\n[Agent]: {msg.content}")

            if not msg.tool_calls:
                # Agent is done or wants to converse — give it a nudge if runs remain
                remaining = self.max_runs - self.run_counter
                if remaining > 0:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"You still have {remaining} training runs available. "
                            "Continue optimizing — either run more experiments or "
                            "if you believe you've found the best configuration, "
                            "call get_history() to summarize results and explain your final recommendation."
                        ),
                    })
                    continue
                else:
                    break

            # Execute tool calls
            tool_results = []
            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments if isinstance(tc.function.arguments, dict) else json.loads(tc.function.arguments)

                print(f"\n  → Tool: {fn_name}({json.dumps(fn_args, indent=None)})")
                result = self.dispatch_tool(fn_name, fn_args)
                print(f"  ← Result: {json.dumps(result, indent=None)[:300]}")

                tool_results.append({
                    "role": "tool",
                    "content": json.dumps(result),
                })

            messages.extend(tool_results)

        # Final summary
        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        best = self._best_run()
        if best:
            print(f"Best result: {best.summary()}")
            print(f"Best config: {best.config}")
        else:
            print("No successful runs.")
        self._save_history()
        print(f"\nFull history saved to autotuner_history.json")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM-driven hyperparameter optimization for train.py")
    parser.add_argument("--model", default="qwen2.5:72b",
                        help="Ollama model to use (default: qwen2.5:72b). "
                             "Other good options: llama3.3:70b, qwq:32b, deepseek-r1:32b")
    parser.add_argument("--max-runs", type=int, default=20,
                        help="Maximum number of training runs (default: 20)")
    parser.add_argument("--train-script", default="train.py",
                        help="Path to the training script (default: train.py)")
    args = parser.parse_args()

    tuner = AutoTuner(
        train_script=args.train_script,
        model=args.model,
        max_runs=args.max_runs,
    )
    tuner.run()
