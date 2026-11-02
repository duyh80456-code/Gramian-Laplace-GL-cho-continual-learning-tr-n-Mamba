"""Stage raw ImageNet-R using the pinned official Mamba-CL train/val lists.

Only symlinks are created under the writable working directory. Kaggle Inputs
are never modified or copied into the resume archive. No images are decoded.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'upstream/mamba-cl/tools'


def official_lists():
    splits, hashes = {}, {}
    for mode, expected_count in (('train', 24000), ('val', 6000)):
        file = TOOLS / f'imagenet_r_{mode}.txt'
        paths = [Path(line) for line in file.read_text().splitlines() if line.strip()]
        if any(p.is_absolute() or len(p.parts) != 2 or '..' in p.parts for p in paths):
            raise ValueError('Invalid official split path')
        if len(paths) != expected_count or len(set(paths)) != len(paths):
            raise ValueError('Official split size/uniqueness mismatch: ' + mode)
        splits[mode] = paths
        hashes[mode] = hashlib.sha256(file.read_bytes()).hexdigest()
    if set(splits['train']) & set(splits['val']):
        raise ValueError('Official train/val lists overlap')
    classes = {p.parts[0] for p in splits['train']}
    if len(classes) != 200 or classes != {p.parts[0] for p in splits['val']}:
        raise ValueError('Official split must contain the same200 classes')
    return splits, classes, hashes


def walk_inputs(search_root, errors):
    """Follow Kaggle directory aliases once, including symlinks, without cycles."""
    visited = set()
    def onerror(error):
        if len(errors) < 10:
            errors.append(str(error))
    for directory, dirs, files in os.walk(search_root, followlinks=True, onerror=onerror):
        path = Path(directory)
        try:
            resolved = path.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            onerror(error)
            dirs[:] = []
            continue
        if resolved in visited:
            dirs[:] = []
            continue
        visited.add(resolved)
        dirs.sort()
        yield path, dirs, files


def input_inventory(search_root):
    """Small mount listing for an actionable error, not a dump of image names."""
    entries = []
    for path in sorted(Path(search_root).iterdir())[:12]:
        entry = dict(path=str(path), directory=path.is_dir(), symlink=path.is_symlink())
        if path.is_dir():
            try:
                entry['children'] = sorted(p.name for p in path.iterdir())[:12]
            except OSError as error:
                entry['error'] = str(error)
        entries.append(entry)
    return entries


def discover(search_root, splits, classes):
    """Identify layout first; validate ALL required files separately afterward."""
    search_root = Path(search_root).absolute()
    if not search_root.is_dir():
        raise FileNotFoundError('Dataset search root not found: ' + str(search_root))
    candidates = {}
    errors = []
    for path, dirs, files in walk_inputs(search_root, errors):
        directories = set(dirs)
        split_classes = set()
        if {'train', 'val'} <= directories:
            for mode in ('train', 'val'):
                split_classes.update(p.name for p in (path / mode).iterdir() if p.is_dir())
        if split_classes & classes:
            candidates[path.resolve()] = (path, 'split')
            dirs[:] = []
        elif classes & directories:
            # Do not treat train/val children of a split dataset as raw roots.
            if path.name in ('train', 'val') and (path.parent / 'train').is_dir() and (path.parent / 'val').is_dir():
                candidates[path.parent.resolve()] = (path.parent, 'split')
            else:
                candidates[path.resolve()] = (path, 'raw')
            dirs[:] = []
    if len(candidates) != 1:
        details = dict(search_root=str(search_root),
                       candidates=[str(source) for source, kind in candidates.values()],
                       mounts=input_inventory(search_root), walk_errors=errors)
        raise ValueError('Expected one raw ImageNet-R class root or official train/val root. '
                         'Attach the ImageNet-R dataset, or set DATA_ROOT to its exact directory. '
                         'Input diagnostics: ' + json.dumps(details))
    return next(iter(candidates.values()))


def find_checkpoint(search_root):
    errors, choices = [], {}
    for path, dirs, files in walk_inputs(search_root, errors):
        if 'defocus_mamba_large_cls_21k.pth' in files:
            checkpoint = path / 'defocus_mamba_large_cls_21k.pth'
            choices[checkpoint.resolve()] = checkpoint
    if len(choices) != 1:
        raise ValueError('Expected one defocus_mamba_large_cls_21k.pth; set PRETRAINED explicitly. '
                         + json.dumps(dict(candidates=[str(p) for p in choices.values()],
                                           mounts=input_inventory(search_root), walk_errors=errors)))
    return next(iter(choices.values()))


def validate_split(root, splits, classes):
    for mode, paths in splits.items():
        part = root / mode
        if not part.is_dir():
            raise ValueError('Missing split directory: ' + str(part))
        entries = list(part.iterdir())
        actual_classes = {p.name for p in entries}
        if actual_classes != classes or any(not p.is_dir() for p in entries):
            raise ValueError(f'{mode}: class directories do not match the official200 classes')
        expected = set(paths)
        actual = set()
        invalid = []
        for cls in sorted(classes):
            for file in (part / cls).iterdir():
                relative = Path(cls) / file.name
                if not file.is_file():
                    invalid.append(str(relative))
                actual.add(relative)
        missing, extra = expected - actual, actual - expected
        if missing or extra or invalid:
            raise ValueError(f'{mode}: split mismatch; missing={len(missing)}, extra={len(extra)}, '
                             f'invalid={len(invalid)}; examples=' + json.dumps(dict(
                                 missing=[str(p) for p in sorted(missing)[:5]],
                                 extra=[str(p) for p in sorted(extra)[:5]], invalid=invalid[:5])))


def stage(source, destination, splits, classes):
    # Verify every required source file BEFORE creating any output link.
    missing = [str(p) for paths in splits.values() for p in paths if not (source / p).is_file()]
    if missing:
        raise FileNotFoundError(f'ImageNet-R is incomplete: {len(missing)} required files missing; '
                                + json.dumps(missing[:10]))
    for mode, paths in splits.items():
        for cls in classes:
            (destination / mode / cls).mkdir(parents=True, exist_ok=True)
        for relative in paths:
            original = (source / relative).resolve()
            link = destination / mode / relative
            if link.is_symlink():
                if link.resolve() == original:
                    continue
                # Reattach a changed Kaggle mount without touching Input files.
                link.unlink()
            elif link.exists():
                raise FileExistsError('Refusing to replace a non-symlink staged file: ' + str(link))
            link.symlink_to(original)
    validate_split(destination, splits, classes)


def prepare(search_root, destination, report):
    splits, classes, hashes = official_lists()
    source, kind = discover(search_root, splits, classes)
    if kind == 'split':
        validate_split(source, splits, classes)
        resolved = source
    else:
        resolved = Path(destination).absolute()
        if resolved == source or source in resolved.parents:
            raise ValueError('Staged output must be separate from the input dataset')
        stage(source, resolved, splits, classes)
    result = dict(status='PASSED', source=str(source), input_layout=kind,
                  data_dir=str(resolved), classes=200, train_samples=24000,
                  val_samples=6000, split_sha256=hashes,
                  split_policy='pinned official Mamba-CL lists; no random resplit',
                  storage='source reused' if kind == 'split' else 'symlinks only; no image copies')
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(report)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='/kaggle/input', help='Raw or split root; auto-discovery searches beneath it')
    parser.add_argument('--stage', help='Writable destination for official split symlinks')
    parser.add_argument('--report')
    parser.add_argument('--find-checkpoint', metavar='SEARCH_ROOT', help='Locate checkpoint through Kaggle directory symlinks')
    args = parser.parse_args()
    if args.find_checkpoint:
        print(find_checkpoint(args.find_checkpoint))
        return
    if not args.stage or not args.report:
        parser.error('--stage and --report are required for data preparation')
    prepare(args.input, args.stage, args.report)


if __name__ == '__main__':
    main()
