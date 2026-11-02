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
  dependencies. Official binary wheels are selected for the runtime CUDA/Torch/
  Python/C++ ABI; source builds are disabled. No host environment replacement.
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

### Full Input search and renamed Kaggle files

The user reported all30000 official names missing and requested full automatic
Input search. Rechecking the public Kaggle listing showed names such as
imagenet-r/n01443537/goldfish_0.jpg, whereas the pinned official lists contain
n01443537/sketch_3.jpg, art_7.jpg, etc. The first199 listed images have no official
filename matches. Thus the earlier assertion that this Kaggle copy could be used
directly with the exact original split was insufficiently verified. A root scan
cannot recover the missing filename correspondence; no index/sort renaming or
random resplit is introduced.

Preparation now scans every file below /kaggle/input, follows directory aliases,
and indexes exact class/filename keys regardless of wrapper/split depth. Identical
aliases are deduplicated; conflicting image copies stop with an explicit error.
If a complete canonical image set is unavailable, original ImageNet-R tar/ZIP
Inputs are inspected and only matching regular image members are extracted. No
archive paths, symlinks or unrelated members determine extraction destinations.
If still incomplete, the notebook automatically downloads the author archive at
https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar, linked from the official
hendrycks/imagenet-r repo. HEAD returned2191079936 bytes; a bounded real archive
read verified five actual image names against the pinned repo lists. Full2.19GB
was not downloaded in the local check. The download is atomic, retries failures,
and records URL/size/computed SHA256; this computed hash is provenance, not a
publisher-supplied checksum. The exact30000-name coverage and24000/6000 split
checks remain enforced before training.

Original archives/extracted images are cached under working input_data, excluded
from checkpoint archives. Reruns reuse a complete cache; a fresh Kaggle session
can reuse an attached original archive or download again. Input preparation
reports whether attached files, an attached archive, a local cache or the official
download supplied the images. Training formulas and benchmark protocol are unchanged.

Verification for this Input-handling revision used a temporary tar containing all
30000 official class/filename keys with one-byte placeholder contents, a renamed
Kaggle-style Input, and a local HTTP-response substitute. Download streaming/hash,
canonical extraction, exact24000/6000 staging, cache reuse and whole-Input lookup
through nested symlinks passed. These are filesystem/preparation checks, not an
image-decoding or model-training run. The real author's archive was separately
checked by HTTP HEAD and a bounded read of five actual image headers.

### CUDA extension installation after the reported build failures

The user's complete pasted log confirms Python3.11, Torch2.2.2, CUDA12.1 and
Triton2.2.0. It still contains no specific compiler diagnostic: both setup.py
wheel builds failed. The original notebook unnecessarily set MAMBA_FORCE_BUILD
and CAUSAL_CONV1D_FORCE_BUILD to TRUE, bypassing the upstream cached-wheel route.
That forced-build path is removed.

The installer now selects direct official GitHub release .whl URLs for pinned
causal-conv1d1.4.0 and mamba-ssm2.2.2. The upstream setup.py files map CUDA11.x
Torch builds to cu118 and CUDA12.x to cu122; selection also uses Python version,
Torch minor version, Linux architecture and torch._C._GLIBCXX_USE_CXX11_ABI.
For the reported standard Torch2.2.2/Python3.11/CUDA12 environment, the matching
cp311 Linux x86_64 ABI-FALSE assets were confirmed in both official release APIs
and via HTTP HEAD200 (104653602 and323776477 bytes). No400MB wheel download or
Linux CUDA import was attempted on this local macOS host.

Pip receives only those .whl URLs, --only-binary=:all: and --no-deps; there is no
source-build fallback or Torch upgrade. Forced-build env vars are removed from
the child environment. Existing matching installations are reused only after a
fresh-process import/API check succeeds. Otherwise the wheels are reinstalled.
The full pip/import log and runtime/URLs/versions/status JSON are written under
Output. Importing the native extensions and the official fast-path APIs checks
loading/ABI availability; no model forward or E2E smoke test is added.

Official source build flags include sm70 binaries (not an explicit sm75 target).
NVIDIA's Turing guide documents Volta-to-Turing binary forward compatibility,
with limitations such as shared-memory capacity. This is compatibility evidence,
not a measured guarantee for the full model; actual T4 execution remains pending.
Sources: https://github.com/Dao-AILab/causal-conv1d/blob/v1.4.0/setup.py ;
https://github.com/state-spaces/mamba/blob/v2.2.2/setup.py ;
https://docs.nvidia.com/cuda/turing-compatibility-guide/index.html .

### Live performance display and the active01 run

The user requested visible performance, validation accuracy, and confirmed that
01 is already running. Only02's generated notebook is updated;01 and all training
modules/configs are left unchanged. A stdlib observer is embedded in02 around the
existing launcher command, retaining the exact runtime source pin and preserving
checkpoint/selection identity. It incrementally reads the same console logs,
labels GPU/job/metric source, prints actual epoch loss/train_acc, per-task and
aggregate evaluation accuracy/forgetting, timing and final AA/AIA. Existing values
on resume are labeled previous log; incomplete log lines are buffered and
truncated/replaced logs are handled. Observer failures only emit warnings and
never terminate or mutate training. A latest-metrics JSON snapshot is saved.

Validation is evaluated only at the existing tuning task boundaries on the10%
training holdout. Main02 uses the full training set, so its official evaluation
accuracy is labeled test_acc, not a new validation metric. No new evaluation
calls, per-epoch held-out checks, early stopping or tuning criterion are added.
The currently active01 cannot gain a new display without replacing its launcher;
it should continue uninterrupted. Both stages retain existing epoch/task resume.


### Storage repair after both epoch2 checkpoint writes failed

The user reported ENOSPC during archive creation, then PyTorch file-write and
unexpected-position errors for both lanes. Full frozen backbones, repeated
static caches, periodic tar copies and failed temporary files competed for disk.
No training horizon, GL formula, precision, baseline parameter set, task split
or lambda-selection criterion is changed by this repair.

Epoch checkpoints now contain the configured training parameter patterns and all
buffers; frozen weights are reconstructed from the SHA256-checked original
pretrained checkpoint. Immutable task contexts hold cached heads/projectors and
GL state once per task. Large CPU square matrices use upper-triangle bits plus
lower-XOR residual and lossless zlib byte-plane compression; all asymmetric bits
are retained. Restoring optimizer/model on a tiny CPU model gives bitwise equal
next-update weights. Completed-task RNG is restored before next-head creation.

Periodic tar copies are disabled by default. Environment and data cache move to
/kaggle/temp/glcl outside saved Output, without assuming a separate physical
disk. The original downloaded tar is removed only after all30000 extracted
required images pass coverage checks. Valid checkpoints and archives are never
removed by the bootstrap scratch cleanup; a newer complete .tmp checkpoint is
CRC checked and promoted. Cleanup refuses active GL worker/launcher processes.

Cross-process serialized writes check actual disk plus a16GiB Output ceiling
and1GiB reserve. Failed writes remove their partial temporary files and preserve
the previous checkpoint; capacity exhaustion exits75. Restored immutable blobs
are read from prior Input shards with SHA verification when recorded. Users must
retain all shards listed in resume_dependencies.json. The exact old source
SHA256 ddf74bae8905a19b6d4a83ac0fba297407a8815fbfa197814ff5498923afa07a
is accepted solely for this audited storage migration; all other identity fields
remain exact. Renamed standalone old checkpoint downloads are recognized too.
Both notebooks now embed the read-only metrics observer, since the old run failed.

Validation: registered section-six CPU checks PASS, InfSSM still DEFERRED;
CPU-only storage tests verify compact and legacy model loading, exact next Adam
update, immutable context reuse, Input-only factor resolution, renamed checkpoint
discovery, RNG roundtrip, lossless asymmetric/NaN/signed-zero bit packing, disk and
Output-budget pause, and failed checkpoint/archive cleanup retaining old files.
Bootstrap checks validate complete temporary-checkpoint recovery and actual small
venv relocation. Local test subprocesses lowered the free-space reserve to16MiB
because this machine itself has little free disk; production remains1GiB. No
CUDA/full Mamba/end-to-end benchmark run was executed locally. The downloaded
old startup archive contained epoch0, not the later epoch1 state; recovery of the
user's epoch1 requires their actual valid latest.pt or newer saved Output.
