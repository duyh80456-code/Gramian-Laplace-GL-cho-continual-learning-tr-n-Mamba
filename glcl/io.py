import hashlib
import json
import os
import errno
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    try:
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
        os.replace(tmp, path)
    except OSError as error:
        if error.errno in (errno.ENOSPC,errno.EIO,getattr(errno,'EDQUOT',122)):
            from .storage import StoragePause
            raise StoragePause('Metadata write failed; previous committed state retained: '+str(path)) from error
        raise
    finally:
        tmp.unlink(missing_ok=True)


def atomic_torch(value, path):
    from .storage import StoragePause, disk_writer, require_space, tensor_bytes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with disk_writer(path):
        # A previous failed attempt is never a resume checkpoint.
        tmp.unlink(missing_ok=True)
        require_space(path, tensor_bytes(value))
        try:
            with tmp.open('wb') as stream:
                torch.save(value, stream)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
        except (OSError, RuntimeError) as error:
            if (isinstance(error, OSError)
                    or any(s in str(error).lower() for s in ('file write failed', 'failed writing', 'unexpected pos'))):
                raise StoragePause('Checkpoint write failed; previous committed file retained: '
                                   + str(path) + '. ' + str(error)) from error
            raise
        finally:
            tmp.unlink(missing_ok=True)


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
