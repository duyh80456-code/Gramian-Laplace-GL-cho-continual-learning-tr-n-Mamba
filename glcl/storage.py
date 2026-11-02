"""Bounded disk writes and immutable resume artifacts; no training math changes."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickletools
import shutil
import zipfile

# Audited pre-storage-fix source, including the user's 9367998 Kaggle run.
LEGACY_SOURCES = set()  # AMP/resize/tune10 is a new protocol, not a storage-only migration.


class StoragePause(RuntimeError):
    """Keep the last committed checkpoint and request another Output session."""


def compatible_source(value):
    from .io import digest_sources
    return value == digest_sources() or value in LEGACY_SOURCES


def compatible_identity(saved, expected):
    if not compatible_source(saved.get('source_sha256')):
        return False
    a, b = dict(saved), dict(expected)
    a.pop('source_sha256', None)
    b.pop('source_sha256', None)
    return a == b


def output_root(path):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        if candidate.name == 'glcl_outputs':
            return candidate
    return path.parent


@contextmanager
def disk_writer(path):
    root = output_root(path)
    root.mkdir(parents=True, exist_ok=True)
    with (root / '.storage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def tensor_bytes(value):
    import torch
    seen = set()
    def size(x):
        if torch.is_tensor(x):
            storage = x.untyped_storage()
            key = (str(x.device), storage.data_ptr(), storage.nbytes())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(x, dict):
            return sum(size(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return sum(size(v) for v in x)
        if isinstance(x, (bytes, bytearray, memoryview)):
            return len(x)
        return 0
    return size(value)


def require_space(path, incoming):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(path.parent).free
    reserve = int(os.environ.get('GL_MIN_FREE_BYTES', str(2 * 1024**3)))
    # Tensor storages dominate torch serialization; allow metadata/alignment too.
    needed = int(incoming * 1.03) + 4 * 1024**2 + reserve
    # Kaggle's saved-Output quota is independent of filesystem free space on
    # some images. Keep a conservative logical payload ceiling as well.
    root = output_root(path)
    budget = int(os.environ.get('GL_OUTPUT_BUDGET_BYTES', str(16 * 1024**3)))
    used = 0
    for file in root.rglob('*'):
        try:
            if file.is_file():
                used += file.stat().st_size
        except FileNotFoundError:  # another lane just committed/pruned a file
            pass
    if used + needed - reserve > budget:
        raise StoragePause(f'Output payload budget reached: used={used/1024**3:.2f}GiB, '
                           f'new write={(needed-reserve)/1024**3:.2f}GiB, budget={budget/1024**3:.2f}GiB. '
                           'Keep this Output shard and resume in a fresh session; '
                           'old immutable artifacts will be read from attached Inputs.')
    if free < needed:
        raise StoragePause(f'Insufficient disk for {path.name}: free={free/1024**3:.2f}GiB, '
                           f'new write+reserve={needed/1024**3:.2f}GiB. '
                           'The previous checkpoint is retained. Save this Output, then resume '
                           'with it and all required earlier Output shards attached.')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_metadata(path):
    """Read only primitive header fields; never execute the checkpoint pickle."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        with archive.open(name) as stream:
            prefix = stream.read(256 * 1024)
    fields, pending = {}, None
    unknown = object()
    memo, last = {}, unknown
    keys = ('task', 'epoch', 'task_complete', 'arm', 'stage', 'strength', 'source_sha256')
    for op, arg, _ in pickletools.genops(prefix):
        if op.name in ('BINPUT', 'LONG_BINPUT', 'PUT'):
            memo[int(arg)] = last
            continue
        if op.name == 'MEMOIZE':
            memo[len(memo)] = last
            continue
        if op.name in ('BINGET', 'LONG_BINGET', 'GET'):
            atom = memo.get(int(arg), unknown)
        elif op.name in ('BININT', 'BININT1', 'BININT2', 'INT', 'BINFLOAT', 'FLOAT',
                         'BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE'):
            atom = arg
        elif op.name in ('NEWTRUE', 'NEWFALSE'):
            atom = op.name == 'NEWTRUE'
        else:
            last = unknown
            continue
        last = atom
        if atom is unknown:
            continue
        if pending:
            fields[pending] = atom
            pending = None
            if all(k in fields for k in keys):
                return fields
        elif isinstance(atom, str) and atom in keys:
            pending = atom
        # Position fields come immediately after identity in our checkpoint.
        # Generic/old fixtures may not have the optional identity fields.
        if all(k in fields for k in ('task', 'epoch', 'task_complete')):
            return fields
    raise ValueError('Checkpoint task/epoch metadata is missing or truncated: ' + str(path))


def checkpoint_position(path):
    fields = checkpoint_metadata(path)
    return (int(fields['task']), int(fields['epoch']), int(fields['task_complete']))


def artifact_path(out, relative, expected_hash=None):
    """Use local files or immutable prior Output Inputs, never copy their blobs."""
    out, relative = Path(out), Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe resume artifact path')
    root = output_root(out)
    key = str(out.relative_to(root) / relative)
    local = out / relative
    choices = [local] if local.is_file() else []
    catalog = root / '.external_artifacts.json'
    if catalog.is_file():
        choices += [Path(p) for p in json.loads(catalog.read_text()).get(key, [])]
    seen = set()
    for candidate in choices:
        if not candidate.is_file() or candidate.resolve() in seen:
            continue
        seen.add(candidate.resolve())
        if expected_hash is None or sha256_file(candidate) == expected_hash:
            return candidate
    raise FileNotFoundError('Missing resume artifact ' + key + '. Attach the earlier Output '
                            'shard containing this file; do not restart the experiment from scratch.')


def walk_roots(search):
    seen = set()
    for directory, dirs, files in os.walk(search, followlinks=True):
        path = Path(directory)
        try:
            real = path.resolve(strict=True)
        except (OSError, RuntimeError):
            dirs[:] = []
            continue
        if real in seen:
            dirs[:] = []
            continue
        seen.add(real)
        dirs.sort()
        if path.name == 'glcl_outputs':
            yield path
            dirs[:] = []


def restore_direct(inputs, working):
    """Restore latest model state; reference large immutable blobs on Inputs."""
    from .io import atomic_json
    working = Path(working)
    roots = [p for p in walk_roots(inputs) if p.resolve() != working.resolve()]
    catalog_file = working / '.external_artifacts.json'
    previous = json.loads(catalog_file.read_text()) if catalog_file.is_file() else {}
    catalog = {key: [p for p in paths if Path(p).is_file()] for key, paths in previous.items()}
    candidates = {}
    for root in roots:
        for path in root.rglob('*.pt'):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if 'contexts' in relative.parts or 'curvature' in relative.parts:
                if str(path) not in catalog.setdefault(str(relative), []):
                    catalog[str(relative)].append(str(path))
            elif path.name == 'latest.pt' and len(relative.parts) == 3:
                try:
                    if not compatible_source(checkpoint_metadata(path).get('source_sha256')):
                        continue
                    rank = checkpoint_position(path)
                except (ValueError, OSError, zipfile.BadZipFile, StopIteration):
                    continue
                candidates.setdefault(str(relative.parent), []).append((rank, path.stat().st_mtime, path))
    # Also accept individual downloaded latest.pt files, even if renamed by
    # the browser. The recorded arm/stage identify the destination unambiguously.
    visited = set()
    for directory, dirs, files in os.walk(inputs, followlinks=True):
        directory = Path(directory)
        real = directory.resolve()
        if real in visited or directory.name == 'glcl_outputs':
            dirs[:] = []
            continue
        visited.add(real)
        for name in files:
            if not name.endswith('.pt'):
                continue
            path = directory / name
            try:
                info = checkpoint_metadata(path)
                if (info.get('arm') not in ('MambaCL', 'GL', 'MambaCL+GL')
                        or info.get('stage') not in ('main', 'tune')
                        or not compatible_source(info.get('source_sha256'))):
                    continue
                job = info['arm'] if info['stage'] == 'main' else f"{info['arm']}_lambda{info['strength']:g}"
                relative = str(Path(info['stage']) / job)
                rank = (int(info['task']), int(info['epoch']), int(info['task_complete']))
                candidates.setdefault(relative, []).append((rank, path.stat().st_mtime, path))
            except (ValueError, OSError, zipfile.BadZipFile, StopIteration, KeyError, TypeError):
                continue
    working.mkdir(parents=True, exist_ok=True)
    # Refresh Input mount paths every session; don't persist stale absolute mounts.
    atomic_json(catalog, working / '.external_artifacts.json')
    chosen = []
    for relative, options in candidates.items():
        target = working / relative
        current = target / 'latest.pt'
        best = max(options, key=lambda item: (item[0], item[1]))
        if current.is_file():
            # A same-session valid checkpoint always takes precedence over Inputs.
            checkpoint_position(current)
            continue
        target.mkdir(parents=True, exist_ok=True)
        require_space(current, best[2].stat().st_size)
        temporary = current.with_suffix('.pt.tmp')
        try:
            try:
                os.link(best[2], temporary)
            except OSError:
                shutil.copyfile(best[2], temporary)
            os.replace(temporary, current)
        finally:
            temporary.unlink(missing_ok=True)
        if best[2].parent.name == Path(relative).name:
            for file in best[2].parent.iterdir():
                if file.is_file() and file.suffix in ('.json', '.log', '.csv', '.md'):
                    shutil.copyfile(file, target / file.name)
        chosen.append(dict(run=relative, source=str(best[2]), task=best[0][0]+1, epoch=best[0][1],
                           task_complete=bool(best[0][2])))
        print('RESTORED', relative, 'task', best[0][0]+1, 'epoch', best[0][1],
              '(large contexts/curvature remain on Input)', flush=True)
    # Results/selection metadata are small; completed runs stay skippable.
    for root in sorted(roots, key=lambda p: p.stat().st_mtime):
        identities=list(root.glob('*/*/identity.json'))
        if identities and not any(compatible_source(json.loads(p.read_text()).get('source_sha256')) for p in identities):
            continue
        for file in root.iterdir():
            if file.is_file() and not file.name.startswith('.') and file.suffix in ('.json', '.csv', '.md', '.log'):
                destination = working / file.name
                if not destination.exists():
                    shutil.copyfile(file, destination)
    atomic_json(dict(checkpoints=chosen, input_outputs=[str(p) for p in roots],
                     policy='checkpoint copied; immutable sidecars referenced on Input'), working / 'restore_sources.json')
    return bool(candidates) or any(working.rglob('latest.pt'))
