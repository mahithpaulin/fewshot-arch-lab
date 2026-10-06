# Few-shot learning architectures: a 4-task meta-learning bake-off + PAHO

Research lab: systematic comparison of **meta-learning** learners on
heterogeneous few-shot tasks, followed by a novel architecture designed for
**1–2 example learning with single-forward-pass adaptation**.

> Status: benchmark complete (Actions run passed, `results/` committed).
> Section 4 quotes the real numbers; Section 5 scores four pre-registered
> hypotheses against them — three were falsified, one partial.

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
Snapshot from the passing run (`300` train / `200` eval episodes, seed 0,
`results/*.json`): accuracy = mean±std over eval episodes, adapt = one
`predict()` call.

**vision 5-way** — finetune **0.802±0.108** (1-shot) / **0.946±0.047**
(5-shot); matching 0.724/0.848; protonet 0.647/0.770; paho 0.633/0.749;
fomaml 0.494/0.514; anil 0.473/0.465; reptile 0.377/0.400.

**coding 4-way** — finetune **0.535±0.104** (1-shot) / **0.613±0.083**
(2-shot); paho 0.398/0.452; protonet 0.404/0.432; matching 0.383/0.426;
fomaml 0.312/0.323; reptile 0.292/0.291; anil 0.294/0.298.

**tabular 3-way** — ceiling: everything 0.93–1.00 (protonet 0.991/0.998,
finetune 0.979/0.998, paho 0.975/0.975). Task too easy; discriminates nothing.

**chess 2-way** — null: all methods 0.48–0.54 ≈ chance at both 1- and 5-shot.
The 18-d features + random-play positions carry no learnable threat signal at
this budget. Reported as a null, not tuned until positive.

Adaptation cost (representative, vision 5-shot): matching/protonet
~0.3 ms (0 steps) < paho ~0.9 ms (≤1 step) < anil ~1.4 ms < fomaml/reptile
~1.8 ms (3 steps) < finetune ~8.8 ms (20 steps). PAHO is ~2× faster than
gradient meta-learners, ~10× faster than fine-tuning — not the 5–20×
hypothesized.

## 5. Analysis — hypotheses vs numbers

1. `protonet` ≥ `matching` on vision/tabular — **falsified** on vision
   (matching wins 0.724 vs 0.647 at 1-shot, 0.848 vs 0.770 at 5-shot);
   tabular is a ceiling tie.
2. `fomaml`/`anil` best on coding — **strongly falsified**; they are the
   worst on coding (≤0.32) and weak everywhere. Three inner steps over 300
   episodes are not enough to meta-learn an initialization here.
3. `paho` matches `protonet` at 1-shot, closes the gap at 5-shot while
   adapting much faster — **partial**: PAHO ≈ ProtoNet on vision 1-shot
   (0.633 vs 0.647, within noise), beats it on coding 2-shot (0.452 vs
   0.432), and adapts ~2× faster than MAML-family. But it never beats the
   best baseline in any cell.
4. `finetune` trails at 1-shot — **falsified**: supervised pretraining +
   20 SGD steps is the *best* 1-shot method on vision (0.802) and coding
   (0.535). With tiny meta-training budgets, transfer beats episodic
   meta-learning outright.

Takeaway, stated plainly: **no method here learns from 1–2 examples at
anything close to generalization** except on the trivial tabular task.
PAHO is a legitimate fast adapter (single forward pass + ≤1 gated step,
2× quicker than MAML, competitive with ProtoNet), but it is an incremental
hybrid, not a breakthrough. The honest contributions are (a) the 4-task
bake-off with two falsified expectations and one clean null (chess), and
(b) a working, documented starting point for hypernetwork-based fast
adaptation. Next steps that could change the picture: 5–10× more episodes,
a convolutional encoder for vision, harder tabular splits, and a chess
signal worth learning (e.g. tactical-motif labels instead of mate-in-1
from random play).

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
