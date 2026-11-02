"""Select50/25/10 microbatch under AMP, then gate tuning on20 baseline updates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def verify(report,data,pretrained):
    from glcl.io import digest_sources
    from glcl.configuration import load_config
    from glcl.runner import Runtime
    result=json.loads(Path(report).read_text())
    identity=result['identity']
    cfg=load_config('tune',result['microbatch'],result['use_checkpoint'])
    if (result['status']!='PASS' or result['optimizer_steps']!=20 or result['gpu_count']!=1
        or result['images_per_second']<10 or identity['arm']!='MambaCL'
        or identity['stage']!='tune' or identity['config']!=cfg
        or identity['source_sha256']!=digest_sources()
        or identity['dataset_manifest']!=Runtime.data_manifest(data)
        or identity['pretrained_sha256']!=Runtime.file_hash(pretrained)):
        raise RuntimeError('Throughput gate mismatch/slow; tuning blocked')
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',required=True);p.add_argument('--pretrained',required=True)
    p.add_argument('--out',required=True);p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);report=out/'benchmark.json'
    if a.verify_only:
        verify(report,a.data,a.pretrained);return
    from glcl.io import atomic_json
    from glcl.configuration import base_config_hash
    report.unlink(missing_ok=True)
    attempts=[]
    profile_path=out.parent/'execution_profile.json'
    locked=json.loads(profile_path.read_text()) if profile_path.exists() else None
    if locked and locked['base_config_sha256']!=base_config_hash():
        raise RuntimeError('Saved execution profile belongs to a different protocol')
    # First avoid checkpoint recomputation; fall back to it only if AMP cannot fit.
    for checkpointing in ((locked['use_checkpoint'],) if locked else (False,True)):
        for microbatch in ((locked['microbatch'],) if locked else (50,25,10)):
            attempt=out/f'mb{microbatch}_checkpoint{int(checkpointing)}'
            attempt.mkdir(exist_ok=True)
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1')
            env.pop('GL_EXECUTION_PROFILE',None)
            command=[sys.executable,'-m','glcl.runner','--arm','MambaCL','--stage','tune',
                     '--strength','0','--benchmark-steps','20','--microbatch',str(microbatch),
                     '--checkpointing',str(int(checkpointing)),'--data-root',a.data,
                     '--pretrained',a.pretrained,'--out',str(attempt)]
            print(f'PROBE AMP fp16: microbatch={microbatch}, gradient_checkpointing={checkpointing}',flush=True)
            # Tee directly to notebook output while retaining the OOM diagnostic.
            with (attempt/'console.log').open('w') as log:
                child=subprocess.Popen(command,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
                for line in child.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=child.wait()
            text=(attempt/'console.log').read_text()
            attempts.append(dict(microbatch=microbatch,use_checkpoint=checkpointing,returncode=code))
            atomic_json(attempts,out/'attempts.json')
            if code:
                if 'out of memory' in text.lower():continue
                raise SystemExit(code)
            measured=json.loads((attempt/'benchmark.json').read_text())
            headroom=measured['gpu_total_bytes']-measured['vram_peak_reserved_bytes']
            if headroom<2*1024**3:
                print('Less than2GiB reserved-memory headroom; trying smaller microbatch.',flush=True);continue
            measured.update(microbatch=microbatch,use_checkpoint=checkpointing,headroom_bytes=headroom)
            atomic_json(measured,report)
            print('BENCHMARK SUMMARY:',json.dumps(measured),flush=True)
            if measured['status']!='PASS':
                print('PAUSED: below10 images/s; inspect loader wait before continuing.',flush=True)
                raise SystemExit(75)
            atomic_json(dict(base_config_sha256=base_config_hash(),microbatch=microbatch,
                             use_checkpoint=checkpointing,amp='fp16',headroom_bytes=headroom,
                             scope='One-GPU MambaCL probe, not a guarantee for future GL/history VRAM'),
                        out.parent/'execution_profile.json')
            verify(report,a.data,a.pretrained)
            return
    print('PAUSED: no candidate fits with2GiB headroom; review attempts.json',flush=True)
    raise SystemExit(75)

if __name__=='__main__':main()
