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
from .io import atomic_json


def archive(root, destination):
    root, destination = Path(root), Path(destination)
    tmp = destination.with_suffix('.tmp')
    with tarfile.open(tmp, 'w:gz', compresslevel=1) as tar:
        for p in sorted(root.rglob('*')):
            if p.is_file() and not p.name.endswith('.tmp'):
                tar.add(p, arcname=str(Path(root.name) / p.relative_to(root)), recursive=False)
    os.replace(tmp, destination)


def restore(inputs, working):
    """One explicit archive avoids accidentally merging incompatible histories."""
    inputs, working = Path(inputs), Path(working)
    choices = list(inputs.rglob('glcl_outputs.tar.gz'))
    if not choices:
        return
    if len(choices) != 1:
        raise RuntimeError('Attach exactly one previous Output archive')
    if any(working.rglob('latest.pt')):
        print('Existing local checkpoints retained; restore skipped', flush=True)
        return
    with tarfile.open(choices[0]) as tar:
        for member in tar.getmembers():
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or member.issym() or member.islnk() or (not member.isfile() and not member.isdir()):
                raise ValueError('Unsafe archive member')
            if path.parts[0] != 'glcl_outputs':
                raise ValueError('Unexpected archive root')
        tar.extractall(working.parent)


def run_lanes(jobs, root, data, pretrained, deadline, selection=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    lock = threading.Lock()
    archive_errors = []
    lanes = {}
    archive_path = root.parent / 'glcl_outputs.tar.gz'
    def backup():
        with lock:
            archive(root, archive_path)
    def heartbeat():
        while not stop.wait(30):
            print('GPU lanes:', json.dumps(lanes), flush=True)
            for p in root.rglob('progress.json'):
                print(str(p.relative_to(root)), p.read_text().strip(), flush=True)
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
                from .io import digest_sources
                if identity['source_sha256'] != digest_sources():
                    raise ValueError('Completed output uses another source revision')
                continue
            if time.time() >= deadline:
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
                         archive_errors=archive_errors), root / 'session.json')
        return statuses
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
    args = p.parse_args()
    from .io import ROOT, require_tests
    require_tests()
    config = json.loads((ROOT / 'configs/phase1.json').read_text())
    selection = Path(args.root) / 'selection.json'
    if args.stage == 'tune':
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
    statuses = run_lanes(jobs, args.root, args.data, args.pretrained, args.deadline, selection)
    if args.stage == 'tune' and statuses == [0, 0]:
        subprocess.run([sys.executable, '-m', 'glcl.select', '--root', str(Path(args.root) / 'tune'), '--out', str(selection)], check=True)
    subprocess.run([sys.executable, str(ROOT / 'scripts/collect_results.py'), args.root], check=True)
    archive(Path(args.root), Path(args.root).parent / 'glcl_outputs.tar.gz')
    raise SystemExit(75 if 75 in statuses else 0)

if __name__ == '__main__':
    main()
