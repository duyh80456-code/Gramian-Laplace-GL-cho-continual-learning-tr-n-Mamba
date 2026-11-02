"""Two independent GPU lanes with bounded sessions and atomic output archives."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
import shutil
from .telemetry import disk_status, check_free
from .io import atomic_json
from .storage import (StoragePause, compatible_source, disk_writer, require_space,
                      restore_direct)


def archive(root, destination):
    root, destination = Path(root), Path(destination)
    tmp = destination.with_suffix('.tmp')
    with disk_writer(root / 'archive-reservation'):
        tmp.unlink(missing_ok=True)
        files = [p for p in sorted(root.rglob('*')) if p.is_file() and not p.name.endswith('.tmp')
                 and not p.name.startswith('.')]
        require_space(destination, sum(p.stat().st_size for p in files))
        try:
            with tarfile.open(tmp, 'w:gz', compresslevel=1) as tar:
                for p in files:
                    tar.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
            os.replace(tmp, destination)
        finally:
            tmp.unlink(missing_ok=True)


def restore(inputs, working):
    """One explicit archive avoids accidentally merging incompatible histories."""
    inputs, working = Path(inputs), Path(working)
    if restore_direct(inputs, working):
        return
    choices, visited = [], set()
    for directory, dirs, files in os.walk(inputs, followlinks=True):
        path = Path(directory)
        real = path.resolve()
        if real in visited:
            dirs[:] = []
            continue
        visited.add(real)
        if 'glcl_outputs.tar.gz' in files:
            choices.append(path / 'glcl_outputs.tar.gz')
    if not choices:
        return
    if len(choices) != 1:
        raise RuntimeError('Attach exactly one previous Output archive')
    # Reject old fp32 backups before decompressing their multi-GB checkpoints.
    accepted=False
    with tarfile.open(choices[0],mode='r|gz') as preview:
        for member in preview:
            if Path(member.name).name.startswith('notebook_protocol_') and member.size<1024**2:
                metadata=json.loads(preview.extractfile(member).read())
                accepted=metadata.get('experiment_protocol')==__import__('glcl.configuration',fromlist=['load_config']).load_config('tune')['protocol']
                break
            if member.name.endswith('.pt'):
                break
    if not accepted:
        print('Old fp32 backup preserved on Input; new AMP/resize protocol starts fresh.',flush=True)
        return
    with tarfile.open(choices[0]) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                raise ValueError('Unsafe archive member')
            if path.parts[0] != 'glcl_outputs':
                raise ValueError('Unexpected archive root')
        # Expand legacy archives outside Output. Their static blobs are indexed
        # there and reused; only latest model/optimizer state is restored below.
        import hashlib
        tag = hashlib.sha256((str(choices[0]) + str(choices[0].stat().st_size)).encode()).hexdigest()[:16]
        cache = Path(os.environ.get('GL_EPHEMERAL_ROOT', '/tmp/glcl')) / 'legacy_resume' / tag
        cache.mkdir(parents=True, exist_ok=True)
        marker = cache / '.complete'
        incoming = sum(m.size for m in tar.getmembers() if m.isfile() and not m.name.endswith('.tmp')
                       and not ((cache/m.name).is_file() and (cache/m.name).stat().st_size == m.size))
        require_space(cache / 'unpack', incoming)
        for member in tar.getmembers():
            if not member.isfile() or member.name.endswith('.tmp'):
                continue
            target = cache / member.name
            if marker.is_file() and target.is_file():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination, 1024**2)
        marker.touch()
    restore_direct(cache, working)


def run_lanes(jobs, root, data, pretrained, deadline, selection=None, make_archive=False):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    archive_errors = []
    lanes = {}
    pause_marker = root / '.pause_requested'
    pause_marker.unlink(missing_ok=True)
    archive_path = root.parent / 'glcl_outputs.tar.gz'
    def backup():
        if not make_archive:
            return
        with lock:
            try:
                archive(root, archive_path)
            except Exception as exc:
                archive_errors.append(repr(exc))
                print('OPTIONAL ARCHIVE SKIPPED:', repr(exc), flush=True)
    def heartbeat():
        while not stop.wait(30):
            working = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else root
            try:
                status = disk_status(working)
                atomic_json(dict(path=str(working),lanes=dict(lanes),**status),root/'heartbeat.json')
                if os.environ.get('GL_METRICS_DETAIL','full')=='full':
                    print('DISK working_used_GiB='+str(round(status['working_used_bytes']/1024**3,2))+' free_GiB='+str(round(status['free_bytes']/1024**3,2)),flush=True)
                    print('GPU lanes:',json.dumps(lanes),flush=True)
                check_free(working)
            except StoragePause as error:
                atomic_json(dict(reason=str(error)), pause_marker)
                print('STORAGE PAUSED:', str(error), flush=True)
    def periodic():
        while not stop.wait(300):
            try:
                backup()
            except Exception as exc:
                archive_errors.append(repr(exc))
                print('ARCHIVE ERROR', repr(exc), flush=True)
    def lane(index):
        for job in jobs[index]:
            target = root / job['stage'] / job['name']
            if (target / 'summary.json').exists():
                identity = json.loads((target / 'identity.json').read_text())
                if not compatible_source(identity['source_sha256']):
                    raise ValueError('Completed output uses another source revision')
                continue
            if time.time() >= deadline or pause_marker.exists():
                return 75
            target.mkdir(parents=True, exist_ok=True)
            cmd = [sys.executable, '-m', 'glcl.runner', '--arm', job['arm'], '--stage', job['stage'],
                   '--strength', str(job.get('strength', 1.)), '--data-root', str(data),
                   '--pretrained', str(pretrained), '--out', str(target)]
            if selection and job['arm'] != 'MambaCL' and job['stage'] == 'main':
                cmd += ['--selection', str(selection)]
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(index), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                       PYTHONUNBUFFERED='1', GL_DEADLINE=str(deadline))
            print('START GPU', index, job['name'], flush=True)
            log_path = target / 'console.log'
            with log_path.open('a') as log:
                process = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                lanes[index] = dict(job=job['name'], pid=process.pid, log=str(log_path))
                while process.poll() is None:
                    if time.time() > deadline + 300:
                        # An epoch/statistics pass exceeded grace: preserve last atomic checkpoint.
                        import signal
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            process.wait()
                        backup()
                        return 75
                    stop.wait(10)
            backup()
            lanes[index] = dict(job=job['name'], returncode=process.returncode)
            if process.returncode == 75:
                atomic_json(dict(job=job['name'], reason='One lane paused; finish current epoch and keep Output'), pause_marker)
                return 75
            if process.returncode != 0:
                raise RuntimeError(f'{job["name"]} failed ({process.returncode}); read {log_path}')
        return 0
    threads = [threading.Thread(target=periodic, daemon=True),
               threading.Thread(target=heartbeat, daemon=True)]
    for thread in threads:
        thread.start()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(lane, i) for i in range(2)]
            statuses = [f.result() for f in futures]
        atomic_json(dict(status='PAUSED' if 75 in statuses else 'COMPLETE', lanes=statuses,
                         archive_errors=archive_errors, backup_mode='optional_archive' if make_archive else 'direct_output'), root / 'session.json')
        return statuses
    except BaseException as error:
        try:
            atomic_json(dict(status='FAILED', lanes=lanes, error=str(error),
                             backup_mode='optional_archive' if make_archive else 'direct_output'),
                        root / 'session.json')
        except OSError as status_error:
            print('Could not save failure status:', str(status_error), flush=True)
        raise
    finally:
        stop.set()
        for thread in threads:
            thread.join(timeout=5)
        backup()


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=['tune', 'main'], required=True)
    p.add_argument('--root', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--deadline', type=float, required=True)
    p.add_argument('--jobs', help='Two explicit GPU queues exported by the notebook JOBS cell')
    p.add_argument('--archive', action='store_true', help='Optional full archive only if enough spare disk; direct Output is the default')
    args = p.parse_args()
    from .io import ROOT, require_tests
    require_tests()
    from .configuration import load_config
    config = load_config(args.stage)
    selection = Path(args.root) / 'selection.json'
    if args.stage == 'tune':
        from scripts.benchmark_throughput import verify
        verify(Path(args.root)/'throughput_probe/benchmark.json', args.data, args.pretrained)
        jobs = [[dict(name=f'{arm}_lambda{strength:g}', arm=arm, strength=strength, stage='tune')
                 for strength in config['lambda_grid']] for arm in ('GL', 'MambaCL+GL')]
    else:
        if not selection.exists():
            raise RuntimeError('Attach completed tuning Output containing selection.json')
        jobs = [[dict(name=arm, arm=arm, stage='main') for arm in ('MambaCL', 'MambaCL+GL')],
                [dict(name='GL', arm='GL', stage='main')]]
    if args.jobs:
        requested = json.loads(Path(args.jobs).read_text())
        if not isinstance(requested, list) or len(requested) != 2:
            p.error('--jobs must contain exactly two GPU queues')
        for queue in requested:
            if not isinstance(queue, list):
                p.error('Each GPU queue must be a list')
            for job in queue:
                if (job.get('stage') != args.stage or job.get('arm') not in config['arms']
                        or not isinstance(job.get('name'), str)
                        or Path(job['name']).name != job['name'] or job['name'] in ('.', '..')):
                    p.error('Invalid stage, arm, or name in notebook JOBS')
        # A notebook may expose the schedule but must not silently change the
        # registered arms, lambda grid, or lane assignment.
        if requested != jobs:
            p.error('Notebook JOBS differ from the registered phase-one schedule')
        jobs = requested
    statuses = run_lanes(jobs, args.root, args.data, args.pretrained, args.deadline, selection, args.archive)
    if args.stage == 'tune' and statuses == [0, 0]:
        subprocess.run([sys.executable, '-m', 'glcl.select', '--root', str(Path(args.root) / 'tune'), '--out', str(selection)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/collect_results.py'), args.root], check=True)
    if args.archive:
        try:
            archive(Path(args.root), Path(args.root).parent / 'glcl_outputs.tar.gz')
        except Exception as error:
            print('OPTIONAL ARCHIVE SKIPPED:', str(error), flush=True)
    raise SystemExit(75 if 75 in statuses else 0)

if __name__ == '__main__':
    main()
