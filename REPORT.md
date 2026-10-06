# Few-shot learning architectures: a 4-task meta-learning bake-off + PAHO

Research lab: systematic comparison of **meta-learning** learners on
heterogeneous few-shot tasks, followed by a novel architecture designed for
**1–2 example learning with single-forward-pass adaptation**.

> Status: design + code landed; benchmark numbers below are filled in from
> `results/` after the GitHub Actions run. Where cells say *pending*, the
> workflow has not reported yet — no numbers are invented.

## 1. Why this slice

The request was deliberately broad: any learning architecture (backprop,
Hebbian, meta-learned), any task family (chess, coding, tabular, vision),
ending in something genuinely new that learns from one or two strong
examples. The feasible free-tier slice:

- **Meta-learning only** (per scoping answer), plus a transfer fine-tune
  baseline for calibration. No Hebbian/STDP branch: on these symbolic +
  pixel tasks with 1–5 shots, Hebbian correlational updates underperform
  prototype methods and would double the compute budget for little insight.
  Documented as a limitation, not an oversight.
- **Four tiny CPU tasks**, each with a fixed encoder so methods — not
  backbones — are compared:
  - `vision`: sklearn digits 8×8 + 90°-rotation pseudo-classes → **5-way**,
    12 train pseudo-classes / 8 held-out.
  - `tabular`: Wine + Iris, standardized, **3-way** exemplar-disjoint.
  - `chess`: synthetic `python-chess` random-play positions, 18-d
    handcrafted features, **2-way** mate-in-1-threat vs quiet,
    exemplar-disjoint, classes balanced by rejection sampling.
  - `coding`: ListOps DSL (10 transforms), pair encoding
    `concat(enc(in), enc(out))`, **4-way**, class-disjoint
    (train ids [0,1,3,4,5,6], test ids [2,7,8,9]).
- **Shared MLP encoder** (in→128→64) per task; ~300 meta-train episodes,
  200 eval episodes, seed 0, CPU-only, deterministic.

## 2. Methods

| method | idea | adaptation cost |
|---|---|---|
| `finetune` | supervised pretrain on train pools, fresh N-way head + 20 SGD steps | 20 GD steps |
| `protonet` | class means in embedding space, euclidean | 0 steps |
| `matching` | cosine attention over support | 0 steps |
| `fomaml` | first-order MAML, 3 inner SGD steps (create_graph=False ⇒ 1st order) | 3 steps |
| `reptile` | 3 inner SGD steps, outer move toward adapted weights (eps=0.1) | 3 steps |
| `anil` | MAML with head-only inner loop | 3 steps (head) |
| `paho` (**novel**) | prototypes anchor hypernetwork-synthesized head + 1 learned-gated correction | ≤1 step |

## 3. PAHO — Prototype-Anchored Hypernetwork Optimizer

Goal: learn from 1–2 examples in **one forward pass**, generalizing across
task families without per-task tuning.

1. Encode support → embeddings `e_i`, prototypes `c_k = mean(e_{y=k})`.
2. Task vector `z = MLP(mean(e_i))` (DeepSets).
3. Hypernetwork `H([c_k; z]) → (Δw_k, Δb_k)`; head
   `W = normalize(C)^T + ΔW`, `b = Δb`.
4. Gate `g = σ(MLP(z))`; optional single support-gradient correction
   scaled by `g·α` with learned `α`.
5. Meta-loss: mean of zero-step and one-step query cross-entropy —
   trains both the direct synthesizer and the corrector.

Novelty claim (narrow, verifiable): the *combination* of prototype-anchoring
with task-conditioned hypernetwork synthesis plus a single learned gate, in
one end-to-end episodic objective, with no inner-loop unrolling. Components
exist separately (ProtoNet; HyperNetworks; learned optimizers); this exact
assembly, to our knowledge, does not. If a reviewer finds prior art, the
fallback contribution is the 4-task bake-off itself.

## 4. Results

Auto-generated tables live in `results/SUMMARY.md` (committed by Actions).
Snapshot after the latest run:

*pending — workflow has not reported yet.*

Readout protocol: within each task×shot cell, compare `acc_mean±std`;
`adapt_ms_mean` measures one `predict()` call (forward + adaptation steps);
`params` counts trainable parameters of the meta-model.

## 5. Analysis (to be completed from real numbers)

Hypotheses registered *before* seeing results:

1. `protonet` ≥ `matching` on vision/tabular (euclidean beats cosine at K=1).
2. `fomaml`/`anil` best on coding (relational rule needs head flexibility),
   worst adapt-time everywhere.
3. `paho` matches `protonet` at 1-shot with lower variance, and closes the
   gap to gradient methods at 5-shot while adapting 5–20× faster.
4. `finetune` trails everywhere at 1-shot (20 steps overfit 1 example).

Whatever the numbers say — including a null — will be reported as-is.

## 6. Limitations & threats

- Tiny synthetic tasks; no CIFAR/ImageNet, no real chess ELO, no HumanEval.
- No hyperparameter search (single seed 0, single lr); error bars are
  across episodes only.
- Class-disjoint only for vision/coding; tabular/chess test new exemplars.
- PAHO evaluated in-lab only; no cross-lab replication.
- Free-tier CPU: torch CPU build, ~5–12 min runs; timings are relative.

## 7. Reproduce

```bash
pip install -r requirements.txt
python -m src.run --train-episodes 300 --eval-episodes 200 --shots 1 5
```

## 8. Files

- `src/tasks.py` — episode generators
- `src/methods.py` — all 7 learners incl. PAHO
- `src/run.py` — training/eval driver
- `results/*.json`, `results/SUMMARY.md` — raw numbers (Actions-committed)
