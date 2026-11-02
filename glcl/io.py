import hashlib
import json
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_torch(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, tmp)
    os.replace(tmp, path)


def digest_sources():
    digest = hashlib.sha256()
    files = []
    for folder in ('glcl', 'upstream/mamba-cl', 'configs', 'tests'):
        files += [p for p in (ROOT / folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    for p in sorted(files):
        digest.update(str(p.relative_to(ROOT)).encode())
        digest.update(p.read_bytes())
    return digest.hexdigest()


def require_tests():
    import subprocess
    import sys
    # Always recompute, never trust an attached PASS report.
    subprocess.run([sys.executable, '-m', 'tests.section6'], cwd=ROOT, check=True)
