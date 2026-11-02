"""Find original ImageNet-R and stage the pinned Mamba-CL train/val lists.

Search all Input files, then archives, then optionally download the author copy.
The split uses symlinks; any downloaded archive/images are cached separately.
Kaggle Inputs are never modified. No images are decoded or included in Output.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / 'upstream/mamba-cl/tools'
OFFICIAL_URL = 'https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar'


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


def canonical_name(path, classes):
    """Identify a class/filename independently of wrapper and split directories."""
    path = Path(path)
    for parent in reversed(path.parts[:-1]):
        if parent in classes:
            return Path(parent) / path.name
    return None


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def scan_all_inputs(search_root, splits, classes):
    """Search every file, not just a guessed raw root or a single probe image."""
    required = {p for paths in splits.values() for p in paths}
    found, archives, errors, examples = {}, [], [], []
    for path, dirs, files in walk_inputs(search_root, errors):
        for filename in sorted(files):
            file = path / filename
            key = canonical_name(file, classes)
            if key in required and file.is_file():
                if key in found and found[key].resolve() != file.resolve():
                    if file_sha256(found[key]) != file_sha256(file):
                        raise ValueError('Conflicting copies of ' + str(key) + '; set DATA_ROOT explicitly')
                else:
                    found[key] = file
            elif key is not None and len(examples) < 6:
                examples.append(str(file))
            normalized = filename.lower().replace('_', '-')
            if 'imagenet-r' in normalized and normalized.endswith(('.tar', '.tar.gz', '.tgz', '.zip')):
                archives.append(file)
    print(f'Full Input scan: {len(found)}/{len(required)} official filenames matched; '
          f'{len(archives)} possible ImageNet-R archives', flush=True)
    if examples:
        print('Examples with different names:', json.dumps(examples), flush=True)
    if errors:
        print('Input walk errors:', json.dumps(errors), flush=True)
    return found, list(dict.fromkeys(p.resolve() for p in archives))


def unpack_official_names(archive, destination, splits, classes):
    """Extract only regular images with exact official names; never extractall."""
    required = {p for paths in splits.values() for p in paths}
    destination = Path(destination)
    is_zip = zipfile.is_zipfile(archive)
    opener = zipfile.ZipFile if is_zip else tarfile.open
    with opener(archive) as container:
        members = container.infolist() if is_zip else container.getmembers()
        selected = {}
        for member in members:
            name = member.filename if is_zip else member.name
            if is_zip:
                regular = not member.is_dir() and (member.external_attr >> 16) & 0o170000 != 0o120000
            else:
                regular = member.isfile()
            key = canonical_name(name, classes)
            if regular and key in required:
                if key in selected:
                    raise ValueError('Duplicate official image inside archive: ' + str(key))
                selected[key] = member
        if set(selected) != required:
            raise ValueError(f'{archive}: only {len(selected)}/30000 official filenames; '
                             'cannot reconstruct the official split from renamed images')
        # Validated class/filename keys determine destination; archive paths do
        # not. Symlinks, traversal paths and unrelated members are not extracted.
        for index, (key, member) in enumerate(selected.items(), 1):
            target = destination / key
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(target.suffix + '.tmp')
            stream = container.open(member) if is_zip else container.extractfile(member)
            with stream, temporary.open('wb') as output:
                shutil.copyfileobj(stream, output, 1024 * 1024)
            temporary.replace(target)
            if index % 5000 == 0:
                print(f'Extracted official images: {index}/30000', flush=True)
    return {key: destination / key for key in required}


def download_official(cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / 'imagenet-r.tar'
    if archive.is_file():
        return archive
    temporary = archive.with_suffix('.tar.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(OFFICIAL_URL, headers={'User-Agent': 'GLCL-ImageNetR-preparation'})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open('wb') as output:
                expected = response.headers.get('Content-Length')
                expected = int(expected) if expected is not None else None
                print('Downloading original ImageNet-R:', OFFICIAL_URL,
                      'bytes:', expected, '(the public archive is about2.19GB)', flush=True)
                received, next_log = 0, 128 * 1024 * 1024
                digest = hashlib.sha256()
                for chunk in iter(lambda: response.read(8 * 1024 * 1024), b''):
                    output.write(chunk)
                    digest.update(chunk)
                    received += len(chunk)
                    if received >= next_log:
                        print(f'ImageNet-R download: {received / 1e9:.2f}GB', flush=True)
                        next_log += 128 * 1024 * 1024
                if expected is not None and received != expected:
                    raise OSError(f'Incomplete download: {received}/{expected} bytes')
            temporary.replace(archive)
            (cache / 'download.json').write_text(json.dumps(dict(
                url=OFFICIAL_URL, bytes=received, sha256=digest.hexdigest()), indent=2) + '\n')
            return archive
        except (OSError, ValueError) as error:
            if attempt == 2:
                raise RuntimeError('Official ImageNet-R download failed. Enable Internet, or attach '
                                   'the original imagenet-r.tar with unchanged filenames. ' + str(error)) from error
            print('Retrying official download:', str(error), flush=True)
            time.sleep(attempt + 1)


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


def stage(source, destination, splits, classes, sources=None):
    # Verify every required source file BEFORE creating any output link.
    sources = sources if sources is not None else {p: source / p for paths in splits.values() for p in paths}
    missing = [str(p) for paths in splits.values() for p in paths
               if p not in sources or not sources[p].is_file()]
    if missing:
        raise FileNotFoundError(f'ImageNet-R is incomplete: {len(missing)} required files missing; '
                                + json.dumps(missing[:10]))
    for mode, paths in splits.items():
        for cls in classes:
            (destination / mode / cls).mkdir(parents=True, exist_ok=True)
        for relative in paths:
            original = sources[relative].resolve()
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


def prepare(search_root, destination, report, allow_download=False):
    splits, classes, hashes = official_lists()
    search_root = Path(search_root).absolute()
    if not search_root.is_dir():
        raise FileNotFoundError('Input directory not found: ' + str(search_root))
    resolved = Path(destination).absolute()
    found, archives = scan_all_inputs(search_root, splits, classes)
    kind, origin = 'input_file_index', None
    required = {p for paths in splits.values() for p in paths}
    cache = resolved.parent / 'original_imagenet_r'
    images = cache / 'images'
    if set(found) != required and images.is_dir():
        cached, _ = scan_all_inputs(images, splits, classes)
        if set(cached) == required:
            provenance = cache / 'origin.json'
            origin = json.loads(provenance.read_text())['origin'] if provenance.is_file() else 'existing local original-image cache'
            found, kind = cached, 'cached_original_images'
    if set(found) != required:
        for archive in archives:
            try:
                found = unpack_official_names(archive, images, splits, classes)
                kind, origin = 'attached_original_archive', str(archive)
                break
            except (OSError, ValueError, tarfile.TarError, zipfile.BadZipFile) as error:
                print('Archive cannot supply official filenames:', str(error), flush=True)
    if set(found) != required and allow_download:
        print('Attached Input cannot be mapped to all official split filenames. '
              'Renamed class-index images have no verified mapping to the original split. '
              'Fetching the original dataset; no random resplit will be used.', flush=True)
        archive = download_official(cache)
        try:
            found = unpack_official_names(archive, images, splits, classes)
        except (OSError, ValueError, tarfile.TarError) as error:
            print('Original archive cache is invalid; downloading it once more:', str(error), flush=True)
            archive.unlink()
            archive = download_official(cache)
            found = unpack_official_names(archive, images, splits, classes)
        kind, origin = 'official_download', OFFICIAL_URL
    if set(found) != required:
        missing = sorted(required - set(found))
        raise FileNotFoundError(f'Full Input scan found {len(found)}/30000 official filenames; '
                                f'{len(missing)} missing. Enable --download-official with Internet ON, '
                                'or attach the original imagenet-r.tar. Examples: '
                                + json.dumps([str(p) for p in missing[:10]]))
    if kind in ('attached_original_archive', 'official_download'):
        (cache / 'origin.json').write_text(json.dumps(dict(origin=origin), indent=2) + '\n')
    source = Path(os.path.commonpath([str(p.parent) for p in found.values()]))
    if resolved == source or source in resolved.parents:
        raise ValueError('Staged output must be separate from source image directories')
    print('Preparing exact official train/val symlinks from:', str(source), flush=True)
    stage(source, resolved, splits, classes, sources=found)
    archive_cache = cache / 'imagenet-r.tar'
    if archive_cache.is_file():
        archive_cache.unlink()
        print('Removed redundant downloaded tar after exact split verification; original images retained.', flush=True)
    result = dict(status='PASSED', source=str(source), input_layout=kind,
                  archive_origin=origin, search_root=str(search_root),
                  download_metadata=json.loads((cache / 'download.json').read_text())
                      if origin == OFFICIAL_URL and (cache / 'download.json').is_file() else None,
                  data_dir=str(resolved), classes=200, train_samples=24000,
                  val_samples=6000, split_sha256=hashes,
                  split_policy='pinned official Mamba-CL lists; no random resplit',
                  storage='symlink split; original archive/images cached outside Output if needed')
    report = Path(report)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + '.tmp')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(report)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='/kaggle/input', help='Search every file beneath this root')
    parser.add_argument('--stage', help='Writable destination for official split symlinks')
    parser.add_argument('--report')
    parser.add_argument('--find-checkpoint', metavar='SEARCH_ROOT', help='Locate checkpoint through Kaggle directory symlinks')
    parser.add_argument('--download-official', action=argparse.BooleanOptionalAction, default=True,
                        help='Automatically fetch the author archive if attached filenames do not match (default: enabled)')
    args = parser.parse_args()
    if args.find_checkpoint:
        print(find_checkpoint(args.find_checkpoint))
        return
    if not args.stage or not args.report:
        parser.error('--stage and --report are required for data preparation')
    prepare(args.input, args.stage, args.report, allow_download=args.download_official)


if __name__ == '__main__':
    main()
