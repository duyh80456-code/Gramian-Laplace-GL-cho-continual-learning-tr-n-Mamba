"""Run the required isolated20-step MambaCL gate before starting tuning lanes."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def verify(report, data, pretrained):
    from glcl.io import ROOT, digest_sources
    from glcl.runner import Runtime
    result = json.loads(Path(report).read_text())
    config = json.loads((ROOT/'configs/phase1.json').read_text())
    identity = result['identity']
    if (result['status'] != 'PASS' or result['optimizer_steps'] != 20
            or result['gpu_count'] != 1 or result['images_per_second'] < 10
            or identity['arm'] != 'MambaCL' or identity['stage'] != 'tune'
            or identity['config'] != config or identity['source_sha256'] != digest_sources()
            or identity['dataset_manifest'] != Runtime.data_manifest(data)
            or identity['pretrained_sha256'] != Runtime.file_hash(pretrained)):
        raise RuntimeError('Throughput gate is missing, slow, or belongs to different source/data/weights; tuning blocked.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--verify-only', action='store_true')
    a = p.parse_args()
    out = Path(a.out)
    report = out/'benchmark.json'
    if not a.verify_only:
        # A fresh child initializes pretrained weights/optimizer independently.
        report.unlink(missing_ok=True)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', OMP_NUM_THREADS='1',
                   MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1')
        print('BENCHMARK:20 optimizer steps, MambaCL, GPU0 only,2 loader workers; no tuning lanes yet.', flush=True)
        result = subprocess.run([sys.executable, '-m', 'glcl.runner', '--arm', 'MambaCL',
            '--stage', 'tune', '--strength', '0', '--benchmark-steps', '20',
            '--data-root', a.data, '--pretrained', a.pretrained, '--out', str(out)], env=env)
        if result.returncode:
            raise SystemExit(result.returncode)
        measured = json.loads(report.read_text())
        print('Throughput:', json.dumps({k:measured[k] for k in (
            'optimizer_steps','seconds_per_step','images_per_second',
            'dataloader_wait_fraction','loader_diagnosis','limitation','status')}), flush=True)
        if measured['status'] != 'PASS':
            print('PAUSED: below10 images/second. Inspect benchmark.json and STEP_METRICS before starting01.', flush=True)
            raise SystemExit(75)
    verify(report, a.data, a.pretrained)
    print('20-step throughput gate PASS; this child exits before tuning starts.', flush=True)


if __name__ == '__main__':
    main()
