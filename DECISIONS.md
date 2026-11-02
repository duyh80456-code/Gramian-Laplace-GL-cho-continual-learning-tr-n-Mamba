# Implementation decisions — 2026-11-02

## Source and scope

- Official source: https://github.com/zugexiaodui/mamba-cl at
  `0ebac10c600ca7179174e2fa9bf2036218d48dba`, vendored unchanged under
  `upstream/mamba-cl`. It already includes the De-focus model. A second clone
  of De-focus is unnecessary and is not used.
- Only stage-one arms are implemented. No L2/EWC/KFAC/InfSSM training arms or
  ablations are implemented before phase-two approval. EWC/KFAC in diagnostics
  are measurement-only quadratic proxies. Test5 is explicitly DEFERRED, not PASS.
- Kaggle source setup now follows the attached H4 notebook: clone this project's
  GitHub repo with Kaggle Secret github_token, fetch the full immutable commit,
  detached checkout, verify HEAD, log code_commit.json. REPO_COMMIT/RUN_TAG/JOBS
  are explicit in the first configuration cell. No source ZIP is required.
  The earlier ZIP bootstrap was replaced at the user's request. Existing source
  digests remain as additional resume-identity checks.
- The requested full benchmark is not executed locally: no CUDA T4, dataset, or
  pretrained weights were supplied here. The amended CPU gate now passes; the
  user will launch notebooks on Kaggle themselves.

## Authorized test amendments and their history

The user explicitly chose to keep section-four training formulas and replace
original tests1/3 with exact-reference and approximation-ranking checks:
C exact <=1e-8; 100-noise factored C Spearman >=.9; Delta exact JVP <=1e-6;
100-noise factored Delta Spearman >=.85. Report median relative errors. References
are used only in tests/diagnostics, never training. Do not change formulas if fail.

The initial fixture (seed7) was retained. B,C,u are linear projections of x.
Noise seeds19/23 are fixed before the amended checks. No fixture/seed search.
The exact references pass, but C's Spearman=.8524332433243322 fails .9.
Delta's Spearman=.968064806480648 passes .85. Median errors: C=.1642084,
Delta=1.7598363. A high rank correlation does not imply calibrated magnitudes.
That was the result under the first amendment; no benchmark was run.

### Supervisor correction before training

“ngưỡng 1b hiệu chỉnh lại vì sai fixture, trước khi có bất kỳ kết quả train nào”.

The user explicitly replaced the absolute .9 threshold for 1b with a relative
comparison on the SAME fixture and SAME 100 perturbations (seed19):
trace-corrected Sigma_x tensor P must rank above Sigma_x tensor I, and its
Spearman must be >=.8. Test3b similarly must exceed the unweighted Sigma_u
proxy while retaining the >=.85 floor. No training formula, fixture, or noise
seed changed. Test5 remains DEFERRED until phase two.

Recomputed float64 CPU results:

- C: GL .8524332433243322 > input-only proxy .7255085508550856, and >=.8: PASS.
- Delta: GL .968064806480648 > Sigma_u proxy .9505430543054305, and >=.85: PASS.
- Exact references, Q recursion and lambda0 checks remain PASS.

The stage-one CPU gate is now open. No benchmark has started in this workspace.
The user chose to run Kaggle notebooks themselves. This host has no NVIDIA GPU,
no configured Kaggle API credentials, and browser control was unavailable.

## Official-code differences that affect interpretation

- Official `get_args()` uses 10 epochs and milestones5/8. The supplied request
  explicitly asks 50 and25/40; all stage-one runs override those values together.
- Official training_string is `head,x_proj,out_proj.weight,A_log`, which freezes
  dt_proj (both W_up and bias). The literal baseline retains that set. GL and
  hybrid add dt_proj, as required to evaluate section4.4. This is an experimental
  confound: hybrid-versus-baseline changes the trainable parameter set. Report it;
  an additional matched-parameter baseline would need a later scope decision.
  The user explicitly confirmed this choice: keep dt_proj frozen for MambaCL
  to retain the official baseline parameter set for comparison to paper81.67.
  This does not guarantee that the observed AA will equal81.67.
  comparison.csv and comparison.md disclose each run's training patterns and
  the frozen/trainable status of dt_proj.weight and bias, read from that run's
  recorded identity.json configuration. Both reports state the parameter-set
  confound explicitly; missing recorded configurations are labeled UNKNOWN.
- Pretrained YAML actually has RoPE, group norm64, and checkpointing enabled;
  time_step_rank=48, not D/16. None is disabled. Raw scan-output gradients are
  captured BEFORE normalization, skip, gating and out_proj, so w includes their
  real downstream Jacobians. B/C sufficient factors are pulled back through each
  token's actual RoPE linear map. delta_scale_factor is included in derivatives.
  This is an adaptation of section4 to the official backbone, not evidence that
  the paper's approximation/error bound holds with those operations.
- The original MambaCL projector and ModAdam remain unmodified. The runner adds
  loss/checkpoint insertion points in memory and wraps only the epoch's memory
  scheduling. The original source is kept intact for inspection.
- PIL replaces unavailable accimage decoding; upstream loads images via PIL
  already. sample_type='path' avoids storing all images in memory; transforms,
  image size, expansion factor5 and labels remain upstream's.

## Curvature and online update

- mu=1e-4, gamma=.95, lambda grid {.1,1,10}; independently selected per GL arm.
- Statistics: up to2000 deterministic random current-task TRAIN samples,
  evaluated with upstream eval transform, one sampled predictive label/image.
  No images or replay buffer are persisted. The training holdout is excluded
  from tuning statistics. Main uses full official train data after selection.
- Per-sample factors use token sums as written. Divide by sample count, not
  token count, except denominators explicitly present in section4. Trace scales
  are computed AFTER averaging factors and trace numerators. A zero denominator
  gives coefficient0, without an arbitrary ridge.
- Q is the backward output-diagonal Gramian. It omits cross-time injection terms;
  neither trace correction nor references silently add those terms to training.
- Online precision retains each task's Kronecker pair separately with decayed
  weight gamma. Summing left/right factors independently would introduce false
  cross-task terms, so it is not done. Each historical A factor retains the
  A=-exp(A_log) used at its own estimation point. Anchor is latest task endpoint.
- Kroneckers are applied in trace form. Operators spill to per-task/block files;
  a custom autograd function streams one at a time to GPU and stores only the
  aggregate derivative. Full operators are never materialized in training.
- Heads are task-specific in upstream; GL protects SSM/out-proj blocks, not
  unrelated class-head coordinates. The isotropic prior uses the same protected
  coordinates; no GL penalty exists before first-task consolidation.
- lambda=0 bypasses GL completely (including statistics/hooks), leaving underlying
  optimization untouched. The 20-step bitwise test uses the official ModAdam
  on a tiny CPU model for both projected/unprojected modes; it is not an E2E test.

## Diagnostics

Task1/main GL arms:64 ephemeral train images,50 isotropic unit directions per
parameter block; true logits GGN is computed as exact reverse-Jacobian-row Jv,
then Jv^T(diag(p)-pp^T)Jv. This avoids unsupported scan double backward and does
not use finite differences. This is exact but can be expensive; no runtime
claim is made. Standard KFAC and sampled-label diagonal Fisher are collected
only as diagnostics. Report per block and shallow/deep-half means.

The original A projector acts in token space; an input-only n-state quadratic
proxy is not specified. Its diagnostic Spearman is null with a reason, not an
invented number. dt_proj input-only covariance is labeled as a proxy because
the official projector does not protect dt_proj. These are reporting limitations.

## Kaggle sessions and limits

- Three notebooks:00 CPU checks;01 six validation candidates on two lanes;
  02 full three-arm comparison, GPU0 baseline then hybrid, GPU1 GL. Repeat01/02
  with the previous Output until complete; the number of 12-hour sessions is
  unknown until measured. No horizon is shortened.
- Effective batch200, microbatch5; weighted CE gradients, one GL penalty and one
  optimizer step per effective batch. Checkpointing is upstream's. AMP remains
  OFF, matching the official default. Accumulation changes floating-point order
  relative to a literal batch200, so do not claim bitwise identity to that run.
- Single process/model per GPU, one BLAS thread, two train workers and one eval
  worker per lane. No DDP communication. These are initial memory-conservative
  choices, not a measured throughput optimum. VRAM and CUDA builds are untested.
- Isolated Python3.11, official torch2.2.2/timm1.0.9 and pinned supplemental
  dependencies. T4 sm75 kernels built explicitly. No host environment replacement.
- Atomic latest checkpoint every epoch and after each evaluated task; includes
  model, optimizer, scheduler, RNG, heads, class maps, metrics, projector matrices,
  GL anchor/operator paths. Parameter-id maps are remapped by name on restore.
- Nominal deadline11h from notebook start (including setup). At next checkpoint
  exit75. At+5min, launcher terminates a lingering process and retains the previous
  checkpoint; interrupted task statistics may be recomputed. No mid-epoch resume.
- Source/config/arm/stage/lambda, pretrained-file SHA256 and dataset relative-path
  + size manifest must match on resume. Dataset content itself is not hashed.
- Heartbeat every30s and archive every300s, matching H4. Two explicit notebook
  GPU queues are validated against the registered stage-one schedule. Each stage
  records its source commit, queues and worker settings in a protocol manifest.
- Periodic/final compressed archive; attach exactly one prior Output archive.
  Checkpoints use atomic replacement; prior archives cannot overwrite local state.
- Disk and transfer cost grow with tasks. Approximately tens of GB of exact
  online factor summands per GL arm are possible. This has not been benchmarked;
  spilling limits RAM/VRAM but does not make disk I/O or archive cost negligible.

## Raw ImageNet-R input preparation

The user selected Kaggle dataset my1nonly/imagenet-r and requested automatic
handling. Both01 and02 discover raw class trees below /kaggle/input (or an
explicit DATA_ROOT), validate all filenames required by the pinned official
train/val lists, and stage symlinks at /kaggle/working/glcl/input_data/imagenet_r.
The split is exactly24000/6000 images and200 classes, with no overlap and no
random resplit. Kaggle Inputs are never modified; no images are copied. An
existing split must match the official file membership exactly. Missing files,
extra split files, or ambiguous source roots stop before training. Staging is
idempotent, and stale staging symlinks can be reattached to the current mount.
The preparation report, source root and official split hashes are archived with
results, but staged images/symlinks stay outside glcl_outputs and its archive.
This preparation does not decode images or run the training model.

### Input discovery fix after Kaggle Found:none

The first scanner used os.walk's default followlinks=False and identified a raw
root only if ALL200 class directories plus one particular probe image existed.
That could skip a directory-symlink mount entirely, or hide an incomplete or
mismatched dataset behind the same unhelpful Found:none message. The received
traceback alone does not establish which condition occurred on the user's mount.

Discovery now follows directory aliases with canonical-path deduplication and
cycle protection. It recognizes class-root layout independently of a single
probe filename; the existing all-file/class/split validation then reports missing
or mismatched files explicitly. No split or training formula changed. Empty or
ambiguous discovery prints a bounded mount listing and filesystem errors. The
checkpoint finder uses the same symlink-aware walk instead of Path.rglob.
A temporary filesystem reproduction covers directory aliases, duplicate aliases,
cycles, an incomplete source and checkpoint discovery; no model was executed.

### Safe reruns of the clone cell

The kernel ends the clone cell inside the source checkout. Previously, rerunning
that cell removed its own current working directory, so git clone inherited a
deleted cwd and failed before contacting GitHub. The bootstrap now changes to
/kaggle/working BEFORE deleting REPO, including recovery from an already deleted
cwd, and supplies an explicit existing cwd to every Git subprocess. Source-only
refresh still preserves outputs, staged inputs and the isolated environment.
The runtime source pin is unchanged by this bootstrap-only fix. A local Git
reproduction checks recovery from a deleted cwd and two consecutive executions,
with a preserved output marker and no model or network execution.
