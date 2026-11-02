"""Recover generated scratch space without deleting valid experiment checkpoints."""
import argparse
import json
import os
import pickletools
from pathlib import Path
import shutil
import zipfile


def checkpoint_position(path):
    # Bootstrap is deliberately stdlib-only, before the pinned repo/venv exists.
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        with archive.open(name) as stream:
            prefix = stream.read(256*1024)
    fields, pending = {}, None
    for op, arg, _ in pickletools.genops(prefix):
        if op.name in ('BINUNICODE', 'SHORT_BINUNICODE', 'UNICODE') and arg in ('task', 'epoch', 'task_complete'):
            pending = arg
        elif pending and op.name in ('BININT', 'BININT1', 'BININT2', 'INT', 'NEWTRUE', 'NEWFALSE'):
            fields[pending] = True if op.name == 'NEWTRUE' else False if op.name == 'NEWFALSE' else arg
            pending = None
            if len(fields) == 3:
                return (int(fields['task']), int(fields['epoch']), int(fields['task_complete']))
    raise ValueError('Missing checkpoint position')


def active_gl_processes():
    found = []
    proc = Path('/proc')
    if proc.is_dir():
        for path in proc.iterdir():
            if not path.name.isdigit():
                continue
            try:
                words = (path / 'cmdline').read_bytes().split(b'\0')
            except OSError:
                continue
            if any(words[i:i+2] in ([b'-m', b'glcl.runner'], [b'-m', b'glcl.kaggle']) for i in range(len(words)-1)):
                found.append(int(path.name))
    return found


def migrate_directory(source, destination, events):
    if not source.is_dir() or source.resolve() == destination.resolve():
        return
    if destination.exists():
        # Never overwrite another environment/cache. Retain the old one under
        # ephemeral storage so the new canonical cache can be validated first.
        alternate = destination.with_name(destination.name + '-previous-working')
        if alternate.exists():
            raise RuntimeError('Both migration targets exist; inspect storage manually: ' + str(alternate))
        destination = alternate
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    events.append(dict(moved_from=str(source), moved_to=str(destination)))
    # Pip is invoked as python -m pip. Repair console-script shebangs as well.
    if (destination / 'pyvenv.cfg').is_file():
        for file in (destination / 'bin').iterdir():
            if file.is_file() and not file.is_symlink():
                data = file.read_bytes()
                if data.startswith(('#!' + str(source)).encode()):
                    file.write_bytes(data.replace(str(source).encode(), str(destination).encode(), 1))


def prepare(working, ephemeral, repo=None):
    working, ephemeral = Path(working).absolute(), Path(ephemeral).absolute()
    running = active_gl_processes()
    if running:
        raise RuntimeError('A previous GL launcher/worker is still active (PIDs ' + str(running)
                           + '). Stop the failed run before repairing its scratch files.')
    output = working / 'glcl_outputs'
    output.mkdir(parents=True, exist_ok=True)
    ephemeral.mkdir(parents=True, exist_ok=True)
    before = shutil.disk_usage(working).free
    removed, events, recovered = [], [], []
    paths = [p for p in output.rglob('*.tmp')
             if p.name.endswith(('.pt.tmp', '.json.tmp'))]
    paths += list(output.glob('*/*/contexts/context-pending.pt'))
    paths.append(working / 'glcl_outputs.tar.tmp')
    for path in paths:
        if path.is_file() or path.is_symlink():
            if path.name == 'latest.pt.tmp' and zipfile.is_zipfile(path):
                try:
                    rank = checkpoint_position(path)
                    with zipfile.ZipFile(path) as archive:
                        if archive.testzip() is not None:
                            raise ValueError('Temporary checkpoint CRC failed')
                    current = path.with_suffix('')
                    try:
                        old_rank = checkpoint_position(current)
                    except (OSError, ValueError, zipfile.BadZipFile, StopIteration):
                        old_rank = (-1, -1, -1)
                    if rank > old_rank:
                        os.replace(path, current)
                        recovered.append(dict(path=str(current), task=rank[0]+1, epoch=rank[1]))
                        continue
                except (OSError, ValueError, zipfile.BadZipFile, StopIteration):
                    pass
            size = path.stat().st_size if path.exists() else 0
            path.unlink()
            removed.append(dict(path=str(path), bytes=size))
    # The downloaded tar is a reproducible cache, not a checkpoint. Remove it
    # only if every official image has already been extracted and is present.
    tools = Path(repo)/'upstream/mamba-cl/tools' if repo else None
    if tools and all((tools/f'imagenet_r_{mode}.txt').is_file() for mode in ('train', 'val')):
        names = [line for mode in ('train', 'val') for line in (tools/f'imagenet_r_{mode}.txt').read_text().splitlines()]
        for base in (working/'input_data/original_imagenet_r', ephemeral/'input_data/original_imagenet_r'):
            archive = base/'imagenet-r.tar'
            if archive.is_file() and len(names) == 30000 and all((base/'images'/name).is_file() for name in names):
                size = archive.stat().st_size
                archive.unlink()
                removed.append(dict(path=str(archive), bytes=size, reason='Verified extracted original images; redundant download cache'))
    migrate_directory(working.parent / 'glcl_env', ephemeral / 'env', events)
    migrate_directory(working / 'input_data', ephemeral / 'input_data', events)
    result = dict(removed_failed_temporary_files=removed, recovered_complete_temporary_checkpoints=recovered, migrations=events,
                  preserved='All latest.pt files, immutable contexts/curvature and valid archives',
                  working_free_bytes_before=before, working_free_bytes_after=shutil.disk_usage(working).free,
                  ephemeral_free_bytes=shutil.disk_usage(ephemeral).free)
    target = output / 'storage_preflight.json'
    target.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--working', required=True)
    p.add_argument('--ephemeral', required=True)
    p.add_argument('--repo')
    a = p.parse_args()
    prepare(a.working, a.ephemeral, a.repo)


if __name__ == '__main__':
    main()
