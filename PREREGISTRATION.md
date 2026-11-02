# Preregistration — 2026-11-02

Written before any benchmark training. No ImageNet-R/CIFAR training, validation
selection, or GPU experiment has run in this workspace.

## Scope

Stage one only: official MambaCL, GL, MambaCL+GL; ImageNet-R 10-split; seed 0.
All 10 tasks, 50 epochs/task, effective batch 200, Adam/official ModAdam,
backbone LR 2e-4, classifier LR 1e-2, weight decay 5e-5, milestones 25/40,
eta .90. Main runs use the complete official training split.

Tune each GL arm independently over lambda {0.1, 1, 10}, mu=1e-4, gamma=.95.
Hold out 10% of each training class by a fixed filename hash. First three tasks
only; maximize validation AIA, tie goes to the smaller lambda. Freeze selection
before main runs. Main runs start again from pretrained weights, not tuning
checkpoints. Never select on the official evaluation split.

MambaCL's main run is also the reproduction measurement: report its measured
AA/forgetting beside paper 81.67/4.07. Those numbers are not a stopping gate.

## Registered claims (unchanged)

Paired delta by seed on final AA:

- Main: MambaCL+GL > MambaCL AND GL > L2 in at least 2/3 seeds.
- Secondary: GL > InfSSM AND GL > EWC in at least 2/3 seeds.
- SSM-specific: GL > KFAC by at least 0.5 percentage points in at least
  2/3 seeds; report GL−A and GL−Delta without changing the criterion.
- Statistics floats and GPU-hours: report only, no advance superiority claim.

Stage one, with only seed 0 and no L2/etc., cannot establish these claims.
Stage two, other seeds/arms and CIFAR-100 require later user approval.

## Test amendment authorized by the user

The original section-six exactness assertions for factored C and time-diagonal
Delta were replaced by the user's explicit response, before any benchmark run:

1a. Unfactored C operator equals the actual output change squared (w=1),
    absolute error <=1e-8.
1b. Trace-corrected factored C versus exact output-change energy, 100 random
    perturbations: initially Spearman >=.9 (superseded below); also report median relative error.
3a. Full cross-time Delta reference response equals an autograd JVP, relative
    error <=1e-6.
3b. Time-diagonal H_up versus exact JVP energy, 100 random perturbations:
    Spearman >=.85; also report median relative error.

These references never replace the registered training operators. If a
threshold fails, report it and stop; do not alter formulas or search fixtures.
Tiny fixture fixed at D=6, n=4, rank=2, T=20, float64 CPU, fixture seed7;
perturbation seed19 for C and seed23 for Delta. Delta test uses row d=2.
No full pytest or end-to-end smoke test.

## Supervisor correction before any training

The user reported the .9 threshold had been calibrated on the wrong fixture,
and explicitly authorized this replacement before any training result:

- 1b: same fixture, same100 perturbations, seed19. Spearman of trace-corrected
  Sigma_x tensor P must be strictly greater than that of Sigma_x tensor I,
  and must be >=.8. Log both Spearman values and median GL relative error.
- 3b: same fixture and100 perturbations, retaining seed23 and row2. Spearman
  of H_up must be strictly greater than that of unweighted Sigma_u,
  and must remain >=.85. Log both values and median GL relative error.
- No training formula, exact reference, fixture, perturbation or claim changed.
- Test5 remains DEFERRED until phase two.

Both relative checks now pass (.852433>.725509; .968065>.950543). See
reports/section6.json. Stage one has NOT started locally; the user elected to
run01 then02 on Kaggle. No benchmark result informed this correction.

## User-authorized protocol amendment after the failed fp32 pilot

New protocol amp_fp16_resize256_tune10_v2 applies fp16 train/eval AMP and a shared
short-side256 image cache to every arm; GL curvature/penalty remain fp32. Tuning
uses10 epochs with LR milestones5/8 for each candidate; main retains50 epochs and
25/40. Microbatch is selected50/25/10 by an isolated baseline fit/throughput probe,
with checkpointing disabled if feasible. All arms share that execution profile.
Old fp32/raw-resolution/50-epoch tuning state and selections are not mixed with
this amended protocol. This amendment follows a pilot with task1 epoch1/2 train
logs and storage failure; it is not described as registered before any training.
Details, including numerical/data changes and unmeasured GPU limitations, are
recorded in DECISIONS.md.
