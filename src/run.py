"""Episodic meta-training + evaluation driver. CPU-only, deterministic."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .common import set_seed, count_params
from .tasks import TASK_INFO, sample_episode
from .methods import build


def to_t(arr, dtype="f"):
    import torch
    if dtype == "f":
        return torch.from_numpy(np.asarray(arr, dtype=np.float32))
    return torch.from_numpy(np.asarray(arr, dtype=np.int64))


def run_config(task, method_name, k_shot, train_episodes, eval_episodes, seed, q_query=10):
    n_way = TASK_INFO[task]["n_way"]
    in_dim = TASK_INFO[task]["input_dim"]
    set_seed(seed)
    model = build(method_name, in_dim, n_way)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3) if len(list(model.parameters())) else None

    if method_name == "finetune":
        model.pretrain(task, steps=300, seed=seed)

    rng = np.random.RandomState(seed)
    losses = []
    model.train()
    for _ in range(train_episodes):
        sx, sy, qx, qy = sample_episode(task, "train", n_way, k_shot, q_query, rng)
        sx, qx = to_t(sx), to_t(qx)
        sy, qy = to_t(sy, "i"), to_t(qy, "i")
        if method_name == "finetune":
            continue
        with torch.enable_grad():
            loss = model.train_step(sx, sy, qx, qy, opt)
        if loss is not None:
            losses.append(loss)

    # eval on held-out split
    accs, times = [], []
    model.eval()
    for _ in range(eval_episodes):
        sx, sy, qx, qy = sample_episode(task, "test", n_way, k_shot, q_query, rng)
        sx, qx = to_t(sx), to_t(qx)
        sy, qy = to_t(sy, "i"), to_t(qy, "i")
        t0 = time.perf_counter()
        with torch.enable_grad():  # adaptation may need grads internally
            logits = model.predict(sx, sy, qx)
        dt = (time.perf_counter() - t0) * 1000.0
        pred = logits.argmax(-1)
        accs.append((pred == qy).float().mean().item())
        times.append(dt)
    return {
        "acc_mean": float(np.mean(accs)),
        "acc_std": float(np.std(accs)),
        "adapt_ms_mean": float(np.mean(times)),
        "params": count_params(model),
        "train_loss_mean": float(np.mean(losses)) if losses else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["vision", "tabular", "chess", "coding"])
    ap.add_argument("--methods", nargs="+",
                    default=["finetune", "protonet", "matching", "fomaml", "reptile", "anil", "paho"])
    ap.add_argument("--train-episodes", type=int, default=300)
    ap.add_argument("--eval-episodes", type=int, default=200)
    ap.add_argument("--shots", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        for shot in args.shots:
            eff_shot = shot
            if task == "coding" and shot > 2:  # keep coding 1-/2-shot per spec
                eff_shot = 2 if shot > 2 else shot
            key = f"{task}_{eff_shot}shot"
            print(f"[{datetime.now(timezone.utc).isoformat()}] {key}", flush=True)
            res = {"task": task, "n_way": TASK_INFO[task]["n_way"], "shot": eff_shot,
                   "q_query": 10, "train_episodes": args.train_episodes,
                   "eval_episodes": args.eval_episodes, "seed": args.seed,
                   "results": {}}
            for m in args.methods:
                print(f"  method={m} ...", flush=True)
                r = run_config(task, m, eff_shot, args.train_episodes,
                               args.eval_episodes, args.seed)
                res["results"][m] = r
                print(f"    acc={r['acc_mean']:.3f}±{r['acc_std']:.3f} "
                      f"adapt={r['adapt_ms_mean']:.1f}ms params={r['params']}", flush=True)
            (outdir / f"{key}.json").write_text(json.dumps(res, indent=2))

    # summary fragment for the report
    lines = ["# Benchmark summary (auto-generated)", ""]
    for f in sorted(outdir.glob("*.json")):
        if f.name == "SUMMARY.md":
            continue
        d = json.loads(f.read_text())
        lines.append(f"## {d['task']} {d['n_way']}-way {d['shot']}-shot")
        lines.append("| method | acc | adapt_ms | params |")
        lines.append("|---|---|---|---|")
        for m, r in d["results"].items():
            lines.append(f"| {m} | {r['acc_mean']:.3f}±{r['acc_std']:.3f} "
                         f"| {r['adapt_ms_mean']:.1f} | {r['params']} |")
        lines.append("")
    (outdir / "SUMMARY.md").write_text("\n".join(lines))
    print("wrote", outdir)


if __name__ == "__main__":
    main()
