# fewshot-arch-lab

Comparative study of **meta-learning architectures** across heterogeneous
few-shot tasks (vision / tabular / chess / program-synthesis), plus a
genuinely novel learner:

**PAHO — Prototype-Anchored Hypernetwork Optimizer**

* learns a new class / task from **1–2 examples in a single forward pass**,
* synthesizes classifier weights directly from the support set (no unrolled
  inner loop like MAML),
* anchors synthesis on class prototypes, then applies one learned-gated
  correction step (a minimal learned optimizer).

See [`REPORT.md`](REPORT.md) for the full write-up and
[`results/`](results/) for raw JSON.

## Tasks (all CPU-tiny, deterministic, no downloads)

| task | source | episode |
|---|---|---|
| `vision` | sklearn `load_digits` (8×8) | 5-way, 1-/5-shot |
| `tabular` | sklearn Wine + Iris merged protocol | 3-way, 1-/5-shot |
| `chess` | synthetic `python-chess` positions | 2-way: mate-in-1 threat vs quiet, 1-/5-shot |
| `coding` | ListOps DSL induction (reverse/sort/+k/×k/filter) | 5-way transform ID, 1-/2-shot |

## Methods (all share the same MLP encoder per task)

- `finetune` — transfer baseline, 5 GD steps on support
- `protonet` — Snell et al. 2017
- `matching` — Vinyals et al. 2016 (cosine attention)
- `fomaml` — Finn et al. 2017, first-order, 3 inner steps
- `reptile` — Nichol et al. 2018, k=3
- `anil` — Raghu et al. 2020 (head-only inner loop)
- `paho` (**novel**, this work)

## Reproduce

```bash
pip install -r requirements.txt
python -m src.run --tasks vision tabular chess coding --methods finetune protonet matching fomaml reptile anil paho --train-episodes 300 --eval-episodes 200 --shots 1 5
```

Runs on CPU in ~5–12 min. Results land in `results/<task>_<shot>shot.json`.

## How PAHO works (one paragraph)

Support embeddings → class prototypes `c_k` → DeepSets task vector
`z = MLP(mean(embeddings))` → hypernetwork `H(z, c_k)` emits head weights
`W = W_proto + ΔW(z)` plus a scalar gate `g(z) ∈ (0,1)`.
Prediction is `softmax(x·W+b)` immediately (zero gradient steps);
optionally one support-gradient step scaled by `g·α` is applied.
Meta-loss is query cross-entropy; everything (encoder, task MLP, hypernet,
gate) trains end-to-end. Adaptation cost is one forward pass + at most one
GD step — 10–50× fewer ops than MAML-3.

## Status

- `REPORT.md` holds the benchmark table + analysis.
- GitHub Actions (`.github/workflows/bench.yml`) re-runs the suite on every push.
