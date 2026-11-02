# Gramian–Laplace on official Mamba-CL

Stage-one implementation and Kaggle notebooks. **All active CPU checks pass**;
InfSSM test5 remains DEFERRED until phase two. No benchmark training or validation
selection has run here; the user will launch01 then02 on Kaggle.

The supervisor corrected test1b's threshold before any training, replacing it
with a relative comparison on the unchanged fixture and100 perturbations:

| Approximation check | GL Spearman | Input-only proxy | Floor | Result |
| --- | ---: | ---: | ---: | --- |
| C, seed19 | .852433 | .725509 (Sigma_x tensor I) | .8 | PASS |
| Delta, seed23 | .968065 | .950543 (Sigma_u) | .85 | PASS |

Exact C/reference error1.33e-15; Q error1.67e-16; exact Delta/JVP relative
error7.28e-16; lambda0 is bitwise identical in both tiny20-step official-optimizer
checks. Details: `reports/section6.json` and `DECISIONS.md`.

## Local checks

```bash
python3 -m tests.section6
```

Expected current exit code:0. Both relative comparisons and their floors are enforced. No full pytest
or end-to-end smoke tests are part of this request.

## Kaggle sessions — same source setup as H4

The notebooks clone this GitHub repository using Kaggle Secret `github_token`,
fetch the exact40-character REPO_COMMIT and verify the detached checkout.
The official Mamba-CL source is already vendored at its pinned upstream commit.
No source ZIP or code overlay is attached to Kaggle.

1. Import `notebooks/01_tune_s0.ipynb` into Kaggle. Enable Internet and grant
   this notebook access to your existing `github_token` secret. Select T4 x2.
2. Attach the official `defocus_mamba_large_cls_21k.pth` weights. Dataset input
   is optional: the notebook searches ALL files under /kaggle/input for the exact
   official class/filenames, including nested and symlinked mounts. It can combine
   official images across attached folders, or extract an attached original
   `imagenet-r.tar`/ZIP. Missing/renamed filenames trigger automatic download of
   the author's original archive (about2.19GB) with Internet ON. It then stages
   exactly24000/6000 train/val symlinks and checks200 classes and zero overlap.
   The Kaggle copy `my1nonly/imagenet-r` uses renamed files such as goldfish_0.jpg;
   those cannot be matched to the repo's rendition-based split without a verified
   mapping. The notebook never guesses by file order or makes a random split.
   Downloaded archives and extracted original images are cached outside Output;
   a fresh Kaggle session may download them again unless the original archive or
   unchanged original images are attached. Renamed Input images are left untouched.
   Leave DATA_ROOT=None for a full Input scan; set it only to restrict the search.
   PRETRAINED is also found automatically through directory symlinks.
3. Run All. Source clone, isolated dependencies, registered CPU checks and
   import/GPU checks precede training. The first configuration cell exposes
   REPO_COMMIT, RUN_TAG, STAGE and both GPU JOBS queues, following H4.
   The clone cell may be rerun safely: it leaves the source directory before
   refreshing that checkout and keeps outputs and the environment intact.
4. In01, GPU0 runs GL lambda0.1/1/10 and GPU1 runs hybrid lambda0.1/1/10.
   Selection uses only the10% training holdout on the first three tasks.
5. If PAUSED, save the Output and attach it to the SAME notebook, keeping its
   commit, data and weights. Resume until COMPLETE. Checkpoints preserve the
   registered horizon. Existing local checkpoints take precedence over an
   older attached archive.
6. After01 is COMPLETE, import `notebooks/02_main_s0.ipynb` and attach01's
   Output containing `glcl_outputs.tar.gz` and selection.json, plus the same
   dataset/weights. Enable the same secret and T4 x2. GPU0 runs MambaCL then
   hybrid; GPU1 runs GL. Stop after these seed0 stage-one runs.

Both01 and02 use the SAME source commit and run tag. Only their stage and job
queues differ. Previous Output carries results/checkpoints and validation
selection; source always comes from GitHub. Atomic code_commit.json and stage
protocol manifests record the actual checkout and queues. Heartbeat every30s,
backup every5min,11h session deadline with checkpoint resume, as in H4.
Outputs are under `/kaggle/working/glcl/glcl_outputs`; the resume archive is
`/kaggle/working/glcl/glcl_outputs.tar.gz`.

To regenerate notebooks for a committed source revision already pushed to origin:

```bash
python3 scripts/build_notebooks.py --commit <full-source-commit>
```

The optional00 notebook only runs the registered CPU checks. Full-model CUDA
execution has not been performed locally.

## Code map

- `glcl/math.py`: registered curvature factors and trace-form quadratics.
- `glcl/references.py`: exact tiny references, tests/diagnostics only.
- `glcl/statistics.py`: sampled-label metric and end-task hook collection.
- `glcl/regularizer.py`: online anchors, decayed operator sum, disk streaming.
- `glcl/runner.py`: pinned official pipeline, loss insertion, accumulation/resume.
- `glcl/diagnostics.py`: measurement-only GL/KFAC/EWC/input-proxy comparisons.
- `glcl/select.py`: validation-only lambda selection with identity checks.
- `glcl/kaggle.py`: two-lane launcher, checkpoint archives and deadline handling.
- `scripts/prepare_imagenet_r.py`: full Input filename search, safe archive
  extraction, original-dataset download fallback and exact official split staging;
  preparation provenance is included in the Output archive.
- `PREREGISTRATION.md`: claims and selection rules registered before training.

## Interpretation and remaining validation

Official source is pinned at
`0ebac10c600ca7179174e2fa9bf2036218d48dba`. Each generated notebook pins the
project source commit in its first configuration cell and logs the resolved
commit to code_commit.json; source/config digests also protect resume identity.

The official baseline freezes dt_proj; GL/hybrid include it to implement the
requested Delta factors. The user confirmed keeping this original baseline
parameter set. comparison.csv and comparison.md report both dt_proj weight/bias
status and each run's recorded training patterns, and explicitly disclose this
parameter-set difference. Matching the parameter set does not guarantee AA81.67.
RoPE/group normalization are retained and handled in statistics. No phase-two
training methods, extra seeds or CIFAR runs are queued.

CUDA installation, T4 VRAM, actual throughput, full-model hooks/resume and diagnostic
cost have not been exercised. Exact online operator summands can consume tens of GB
per arm and substantial disk I/O. The schedule is prepared for2×T4; it is not a
measured optimum. Do not interpret these notebooks as successful experiment results.
