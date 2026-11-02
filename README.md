# Gramian–Laplace on official Mamba-CL

Current protocol: `amp_fp16_resize256_tune10_v2`. This is a new experiment relative
to the earlier fp32/raw-image/50-epoch tuning pilot. Old Output is preserved but
its checkpoints and validation selections are not reused in this protocol.
Registered tiny CPU math checks pass; actual T4 speed and microbatch fit must be
measured by the notebook. InfSSM remains DEFERRED until phase two.

## Kaggle notebooks

Import01, enable Internet and Kaggle Secret `github_token`, select T4 x2 and
attach the original defocus_mamba_large_cls_21k.pth weights. Source is cloned from
GitHub at the exact notebook pin; no source ZIP is required. Input ImageNet-R is
searched recursively. Renamed Inputs such as my1nonly/imagenet-r cannot identify
the official split; the original archive is downloaded automatically when needed.

- **01 tuning:** GL and hybrid, independent lambda grids0.1/1/10, three tasks,
  **10 epochs/task, LR milestones5/8**, stratified10% train holdout. GPU0 queues
  GL; GPU1 queues hybrid. Selection maximizes validation AIA, ties favor lower lambda.
- **02 main:** MambaCL, GL, hybrid, ten tasks, **50 epochs/task, milestones25/40**.
  GPU0 queues baseline then hybrid; GPU1 runs GL. Attach completed01 Output,
  including selection.json and execution_profile.json, plus required resume shards.

Changes shared by every arm: train/eval AMP fp16; one shared ImageNet-R cache
resized to short side256 with PIL bicubic/JPEG quality95; persistent training
loader workers. Main02 uses full train data. Evaluation is at task boundaries:
validation accuracy in01, official test accuracy in02. The upstream task log can
show1/10 in01 even though tuning stops after task3.

Effective batch stays200 with one ModAdam update per effective batch. The upstream
`expand_times=5` is retained and now explicit in effective_config.json: an epoch
processes five augmented repeats, so throughput must count processed images,
not unique original filenames. Original baseline freezes dt_proj; GL/hybrid
include it, and comparison reports disclose this parameter-set difference.
The altered AMP/resize protocol and shorter tuning cannot be claimed to reproduce
the paper's81.67 merely because the baseline parameter set matches.

## Required microbatch/throughput probe

Before either01 lane starts, fresh MambaCL subprocesses on GPU0 try microbatch
**50 →25 →10** under AMP. They try gradient checkpointing OFF first. If no candidate
fits, they retry with it ON. The YAML USE_CHECKPOINT=True really enables
checkpoint.checkpoint in each Mamba block; the runtime override is recorded with
the number of checkpointed blocks in effective_config.json. Vendor source stays
unchanged.

Each successful attempt measures20 actual optimizer updates, including any AMP
scale-adjustment attempts in timing; skipped updates are recorded. A candidate
must retain2GiB relative to peak reserved VRAM. The largest passing microbatch is
saved in execution_profile.json and shared by all arms and both notebooks.
On resume that profile stays fixed. This baseline probe cannot guarantee memory
for all future GL histories. All-OOM/insufficient-headroom results pause for review.
Below10 images/second also pauses before tuning. Other errors are surfaced and
are never mistaken for an OOM. The report is throughput_probe/benchmark.json;
per-attempt logs/results and attempts.json are retained. Probe weights/optimizer
are discarded. There is no local GPU measurement in this repository.

## Precision, progress and resume

Microbatch CE uses CUDA fp16 autocast and GradScaler. GL penalty and sampled-label
statistics/Gramian/diagnostics run outside autocast in fp32. Model master weights
remain fp32. The original evaluator receives --use_amp=True. Scaler state is
checkpointed with optimizer/scheduler/RNG and restored before further updates.

Every5 optimizer attempts the notebook prints loss (running mean CE), seconds/step,
images/second, exposed dataloader wait, peak allocated/reserved VRAM and sampled
GPU utilization (null when unavailable). progress.json updates after every batch;
this is live progress, not a step checkpoint. The actual resume point remains the
last committed epoch. Config/args/AMP/persistent-workers/checkpoint setting are
recorded in each run's effective_config.json.

Each arm retains one atomic latest.pt containing trainable parameters, all buffers,
heads, optimizer/scheduler/scaler, RNG, labels and metrics. Frozen backbone weights
are reconstructed from SHA256-checked pretrained weights. Static null-space/GL
cache is stored in an immutable sidecar referenced by latest.pt, reused across
epochs. Lossless square-matrix packing retains asymmetric IEEE bits. Old contexts
are pruned only after latest.pt is committed. Historical GL precision summands
are required by the method and retained. On fresh sessions they can be read from
attached Input shards; keep every dependency listed by resume_dependencies.json.

## Disk policy

Venv, uv-managed Python, uv/pip cache, tar, decoded/resized images and scratch live
under /tmp/glcl, outside /kaggle/working. Generated tar is deleted after successful
extraction. The256 cache is reused across arms/sessions with matching local cache
manifest. Attached images and checkpoints are never deleted. /tmp may share a
physical filesystem; moving files alone does not guarantee free physical space.

There is no periodic tar backup. Optional archive errors are nonfatal. Every30s
heartbeat reports working-tree bytes/GiB and physical free space. Below2GiB requests
PAUSED at an optimizer boundary, preserving the last epoch checkpoint. Atomic writes
also reserve2GiB and enforce a16GiB Output payload ceiling. Save Output, attach it
and all earlier dependency shards, and rerun the same notebook. Non-capacity save I/O failures warn, retain the prior checkpoint and let training
continue; the next epoch retries. Capacity exhaustion produces controlled PAUSED. Bootstrap cleans
failed generated temporaries and preserves valid checkpoint/archive files.

## CPU verification

```bash
python3 -m tests.section6
python3 -m tests.amp_protocol
python3 -m tests.step_telemetry
python3 -m tests.storage_resume
python3 -m tests.storage_bootstrap
```

Local storage tests used GL_MIN_FREE_BYTES=16777216 only because the local disk
itself is nearly full; production remains2GiB. No full pytest suite or local
Mamba/CUDA benchmark is run. C relative rho=.852433>.725509; Delta=.968065>.950543;
exact-reference and lambda0 tiny tests pass. See reports/section6.json,
reports/amp_protocol.json and DECISIONS.md for scope and remaining limitations.
