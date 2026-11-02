"""Generate Kaggle notebooks using H4's authenticated, commit-pinned clone flow."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import textwrap

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stage', action='append', choices=['tests', 'tune', 'main'], help='Generate only selected notebook stages; default: all')
parser.add_argument('--commit', help='Immutable source commit available on origin; defaults to local HEAD')
args = parser.parse_args()
pin = args.commit or subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
if not re.fullmatch('[0-9a-f]{40}', pin):
    parser.error('--commit must be a full 40-character Git commit')
subprocess.run(['git', '-C', str(ROOT), 'cat-file', '-e', pin + ':glcl/runner.py'], check=True)


def cell(kind, source):
    d = dict(id=hashlib.sha256(source.encode()).hexdigest()[:12], cell_type=kind,
             metadata={}, source=textwrap.dedent(source).strip().splitlines(keepends=True))
    if kind == 'code':
        d.update(execution_count=None, outputs=[])
    return d


setup = r'''
import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path
from datetime import datetime, timezone
from kaggle_secrets import UserSecretsClient

os.environ.update(TQDM_DISABLE='1', PYTHONUNBUFFERED='1',
                  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION='python')
MAIN_URL = 'https://github.com/duyh80456-code/Gramian-Laplace-GL-cho-continual-learning-tr-n-Mamba.git'
WORKING = Path('/kaggle/working')
# A previous execution leaves the kernel inside REPO. Always leave that tree
# before removing it, and also recover if a prior run already deleted the cwd.
WORKING.mkdir(parents=True, exist_ok=True)
os.chdir(WORKING)
REPO = Path('/kaggle/working/gramian-laplace')
RUN_ROOT = Path('/kaggle/working/glcl')
EPHEMERAL = Path('/kaggle/temp/glcl')
os.environ.update(GL_EPHEMERAL_ROOT=str(EPHEMERAL), UV_CACHE_DIR=str(EPHEMERAL/'cache/uv'),
                  PIP_CACHE_DIR=str(EPHEMERAL/'cache/pip'))
OUTPUT = RUN_ROOT / 'glcl_outputs'
ROOT_OUT = OUTPUT
subprocess.run([sys.executable, '-c', STORAGE_PREFLIGHT, '--working', str(RUN_ROOT),
                '--ephemeral', str(EPHEMERAL), '--repo', str(REPO)], cwd=WORKING, check=True)

def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)

def private_clone(url, destination, commit):
    token = UserSecretsClient().get_secret('github_token').strip()
    if not token:
        raise RuntimeError('Enable Kaggle Secret github_token for this notebook')
    askpass = Path('/kaggle/working/.glcl_git_askpass.py')
    askpass.write_text("#!/usr/bin/env python3\nimport os,sys\np=sys.argv[1] if len(sys.argv)>1 else ''\nprint('x-access-token' if 'Username' in p else os.environ['GITHUB_TOKEN_RUNTIME'])\n")
    askpass.chmod(0o700)
    env = dict(os.environ, GITHUB_TOKEN_RUNTIME=token, GIT_ASKPASS=str(askpass), GIT_TERMINAL_PROMPT='0')
    try:
        subprocess.run(['git', 'clone', '--filter=blob:none', '--no-checkout', url, str(destination)], cwd=destination.parent, env=env, check=True)
        subprocess.run(['git', '-C', str(destination), 'fetch', 'origin', commit], cwd=destination.parent, env=env, check=True)
        subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', commit], cwd=destination.parent, env=env, check=True)
    finally:
        askpass.unlink(missing_ok=True)

# Only refresh the source checkout. Outputs and environment live elsewhere.
if REPO.exists():
    shutil.rmtree(REPO)
OUTPUT.mkdir(parents=True, exist_ok=True)
private_clone(MAIN_URL, REPO, REPO_COMMIT)
CODE_COMMIT = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], cwd=WORKING, text=True).strip()
if CODE_COMMIT != REPO_COMMIT:
    raise RuntimeError('Pinned repository revision mismatch')
os.chdir(REPO)
print('Resolved code commit:', CODE_COMMIT, 'pin:', REPO_COMMIT)
print('Official MambaCL:', json.loads((REPO/'upstream/LOCK.json').read_text()))
'''
setup = 'STORAGE_PREFLIGHT = ' + repr((ROOT/'scripts/prepare_storage.py').read_text()) + '\n' + setup

install = '''
# Isolated Python 3.11 + official torch 2.2.2; no Kaggle host torch replacement.
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'uv==0.5.31'], check=True)
VENV = EPHEMERAL/'env'
if not (VENV/'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'uv', 'venv', '--python', '3.11', str(VENV)], check=True)
PYTHON = str(VENV/'bin/python')
subprocess.run([sys.executable, '-m', 'uv', 'pip', 'install', '--python', PYTHON,
                '-r', str(REPO/'requirements-kaggle.txt'), 'pip==24.2'], check=True)
# Install official CUDA wheels for the isolated Torch/Python/ABI combination.
# No source compilation and no dependency resolution that could upgrade Torch.
subprocess.run([PYTHON, 'scripts/install_cuda_extensions.py',
                '--log', str(OUTPUT/'environment_install.log'),
                '--report', str(OUTPUT/'environment_install.json')], env=ENV, check=True)
ENV = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
           PYTHONUNBUFFERED='1', TQDM_DISABLE='1')
subprocess.run([PYTHON, '-c', "import torch; print(torch.__version__); print([(i,torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())]); assert torch.cuda.device_count()==2"], env=ENV, check=True)
# Syntax/import only; no image forward or end-to-end smoke test.
subprocess.run([PYTHON, '-c', "import sys; sys.path.insert(0,'upstream/mamba-cl'); import train_eval; from utils.mamba_block import is_fast_path_available; assert is_fast_path_available"], env=ENV, check=True)
'''
tests = '''
# Recompute the supervisor-corrected section-six checks, same fixture/noises:
# C .852433 > proxy .725509 and >=.8; Delta .968065 > proxy .950543 and >=.85.
# Test5 remains DEFERRED until phase two.
# A failure intentionally stops before model initialization/training.
subprocess.run([PYTHON, '-m', 'tests.section6'], env=ENV, check=True)
print((REPO/'reports/section6.json').read_text())
'''
restore = '''
OUTPUT.mkdir(parents=True, exist_ok=True)
subprocess.run([PYTHON, '-c', "from glcl.kaggle import restore; restore('/kaggle/input', '/kaggle/working/glcl/glcl_outputs')"], env=ENV, check=True)
commit_path = OUTPUT/'code_commit.json'
sessions = json.loads(commit_path.read_text()) if commit_path.exists() else []
sessions.append(dict(timestamp=datetime.now(timezone.utc).isoformat(), pin=REPO_COMMIT, commit=CODE_COMMIT, stage=STAGE))
atomic_json(sessions, commit_path)
DATA_REPORT = OUTPUT/'data_preparation.json'
prepare = [PYTHON, 'scripts/prepare_imagenet_r.py', '--input', str(DATA_ROOT or '/kaggle/input'),
           '--stage', str(EPHEMERAL/'input_data'/'imagenet_r'), '--report', str(DATA_REPORT), '--download-official']
subprocess.run(prepare, env=ENV, check=True)
prepared = json.loads(DATA_REPORT.read_text())
if prepared['status'] != 'PASSED':
    raise RuntimeError('ImageNet-R preparation failed; training blocked')
DATA_ROOT = prepared['data_dir']
if PRETRAINED is None:
    PRETRAINED = subprocess.check_output(
        [PYTHON, 'scripts/prepare_imagenet_r.py', '--find-checkpoint', '/kaggle/input'],
        env=ENV, text=True).strip()
print('Data:', DATA_ROOT, 'Weights:', PRETRAINED)
'''
launch = '''
DEADLINE = SESSION_START + 11*3600  # includes environment setup; 5-min hard grace
jobs_path = RUN_ROOT/'effective_jobs'/f'{STAGE}.json'
atomic_json(JOBS, jobs_path)
atomic_json(dict(run_tag=RUN_TAG, stage=STAGE, seed=0, source_commit=CODE_COMMIT,
                 jobs=JOBS, workers=2, OMP_NUM_THREADS=1, MKL_NUM_THREADS=1,
                 performance_observer_sha256=globals().get('PERFORMANCE_OBSERVER_SHA256'),
                 protocol='Full registered horizon; exit75 resumes without truncation'),
            OUTPUT/f'notebook_protocol_{STAGE}.json')
command = [PYTHON, '-m', 'glcl.kaggle', '--stage', STAGE, '--root', str(OUTPUT),
           '--data', DATA_ROOT, '--pretrained', PRETRAINED, '--deadline', str(DEADLINE), '--jobs', str(jobs_path)]
result = subprocess.run(command, env=ENV)
if result.returncode not in (0, 75):
    raise RuntimeError('Run failed. Inspect glcl_outputs/*/*/console.log')
print('PAUSED: attach this Output and rerun same notebook' if result.returncode == 75 else 'Stage complete')
'''
report = '''
for path in (OUTPUT/'selection.json', OUTPUT/'session.json', OUTPUT/'comparison.csv', OUTPUT/'comparison.md'):
    if path.exists():
        print(path.name, path.read_text())
print('Save this notebook Output. Attach its glcl_outputs tree to resume; no tar backup is required.')
print('If resume_dependencies.json names earlier Output shards, keep those versions attached too.')
'''
core_install, kernel_install = install.split('# Install official CUDA wheels', 1)
core_install += "\nENV = dict(os.environ, PYTHONPATH=str(REPO), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1', TQDM_DISABLE='1')\n"
kernel_install = '# Install official CUDA wheels' + kernel_install


for name, stage, title, description in [
    ('00_section6', 'tests', 'Section 6 — CPU float64 gate',
     'Only the registered CPU checks. Exact references and relative comparisons run again. InfSSM remains DEFERRED until phase two.'),
    ('01_tune_s0', 'tune', 'Validation tuning — seed 0',
     'GPU0: GL lambda0.1,1,10. GPU1: MambaCL+GL lambda0.1,1,10. Each candidate uses the first three tasks, 50 epochs, stratified10% training holdout. Max validation AIA, ties use smaller lambda. Repeat with previous Output until COMPLETE.'),
    ('02_main_s0', 'main', 'Full ImageNet-R 10-split — seed 0',
     'Attach completed01 Output containing selection.json. If01 is PAUSED, resume01 first. GPU0: MambaCL then MambaCL+GL; GPU1: GL. Full10 tasks x50 epochs, effective batch200, microbatch5. Main runs start from pretrained weights and use full training data. Stop after seed0 stage one.')]:
    if args.stage and stage not in args.stage:
        continue
    if stage == 'main':
        description += ' Live output includes epoch loss/train_acc and task-end test_acc/forgetting; final AA/AIA are printed when each arm completes. The observer only reads logs; training and resume are unchanged.'
    if stage == 'tune':
        jobs = [[dict(name=f'{arm}_lambda{strength:g}', arm=arm, strength=strength, stage=stage)
                 for strength in (.1, 1., 10.)] for arm in ('GL', 'MambaCL+GL')]
    elif stage == 'main':
        jobs = [[dict(name=arm, arm=arm, stage=stage) for arm in ('MambaCL', 'MambaCL+GL')],
                [dict(name='GL', arm='GL', stage=stage)]]
    else:
        jobs = [[], []]
    config = f"""BRANCH = 'GL'
REPO_COMMIT = {pin!r}
RUN_TAG = {'glcl_imagenetr_s0_' + pin[:12]!r}
STAGE = {stage!r}
DATA_ROOT = None  # optional raw OR official train/val root; auto-discovered under /kaggle/input
PRETRAINED = None  # optional explicit path to defocus_mamba_large_cls_21k.pth
JOBS = {jobs!r}
import time
SESSION_START = time.time()
"""
    stage_launch = launch
    if stage in ('tune', 'main'):
        observer = (ROOT/'scripts/live_performance.py').read_text()
        # Embed the same read-only observer in both stage notebooks.
        stage_launch = ('PERFORMANCE_OBSERVER = ' + repr(observer) + '\n'
                        + 'PERFORMANCE_OBSERVER_SHA256 = ' + repr(hashlib.sha256(observer.encode()).hexdigest()) + '\n' + launch)
        stage_launch = stage_launch.replace(
            'result = subprocess.run(command, env=ENV)',
            "observer = [PYTHON, '-u', '-c', PERFORMANCE_OBSERVER, '--root', str(OUTPUT), '--jobs', str(jobs_path), '--', *command]\nresult = subprocess.run(observer, env=ENV)")
    cells = [cell('markdown', f'# {title}\n\n{description}\n\nSame setup flow as H4: Internet ON, Kaggle Secret `github_token`, T4 x2. Source is cloned from GitHub and checked out at a fixed commit. Attach official pretrained weights. Original ImageNet-R images or imagenet-r.tar are optional Inputs: all Input files are searched automatically. If filenames have been changed (as in my1nonly/imagenet-r), the notebook automatically downloads the author archive (about2.19GB) with Internet ON. It preserves the exact Mamba-CL24000/6000 split and200 classes; it never guesses a mapping from renamed files. Previous Output is for resume and for carrying01 selection into02. Checkpoints are compact; large immutable contexts/curvature can remain on earlier Output Inputs. Keep the shards listed in resume_dependencies.json attached. Archives are optional; the default saves the Output tree directly. No source ZIP is required.\n\nActive CPU checks pass and are recomputed. CUDA compatibility, VRAM and throughput remain unmeasured.'),
             cell('code', config), cell('code', setup), cell('code', core_install), cell('code', tests)]
    if stage != 'tests':
        cells += [cell('code', restore), cell('code', kernel_install), cell('code', stage_launch), cell('code', report)]
    notebook = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                    language_info=dict(name='python', version='3.11'), kaggle=dict(accelerator='nvidiaTeslaT4', isInternetEnabled=True, language='python', sourceType='notebook')), nbformat=4, nbformat_minor=5)
    path = ROOT/'notebooks'/f'{name}.ipynb'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
print('Pinned GitHub source commit:', pin)
print('Generated stages:', args.stage or ['tests', 'tune', 'main'], '; clone via Kaggle github_token; no source ZIP')
