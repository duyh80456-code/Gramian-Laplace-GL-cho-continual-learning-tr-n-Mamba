"""Select50/25/10 microbatch under AMP, then gate tuning on20 baseline updates."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys


def verify(report,data,pretrained):
    from glcl.io import digest_sources
    from glcl.configuration import load_config
    from glcl.runner import Runtime
    result=json.loads(Path(report).read_text())
    if result.get('status')!='PASS':
        raise RuntimeError('Probe blocked: '+str(result.get('status'))+'. '+str(result.get('reason','Inspect benchmark.json')))
    identity=result['identity']
    cfg=load_config('tune',result['microbatch'],result['use_checkpoint'])
    if (result['status']!='PASS' or result['optimizer_steps']!=20 or result['gpu_count']!=1
        or not math.isfinite(result['images_per_second']) or result['images_per_second']<10
        or result.get('numerical_status')!='FINITE'
        or not all(result.get(key) is True for key in ('finite_logits','finite_loss','finite_parameters','finite_optimizer'))
        or identity['arm']!='MambaCL'
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
                for line in child.stdout:
                    log.write(line);log.flush()
                    if not line.startswith('BENCHMARK_RESULT '):print(line,end='',flush=True)
                code=child.wait()
            text=(attempt/'console.log').read_text()
            attempts.append(dict(microbatch=microbatch,use_checkpoint=checkpointing,returncode=code))
            atomic_json(attempts,out/'attempts.json')
            if code:
                failure_path=attempt/'benchmark.json'
                if code==76 and failure_path.exists():
                    failure=json.loads(failure_path.read_text())
                    atomic_json(failure,report)
                    print('NUMERICAL FAILURE: '+failure.get('reason','nonfinite model computation')+'; tuning blocked.',flush=True)
                    raise SystemExit(76)
                if 'out of memory' in text.lower():continue
                atomic_json(dict(status='ERROR',reason='Probe child failed; inspect '+str(attempt/'console.log'),returncode=code),report)
                raise SystemExit(code)
            measured=json.loads((attempt/'benchmark.json').read_text())
            if measured.get('numerical_status')!='FINITE' or not all(measured.get(key) is True for key in ('finite_logits','finite_loss','finite_parameters','finite_optimizer')):
                atomic_json(dict(status='NUMERICAL_FAILURE',reason='Probe lacks finite-state validation'),report)
                raise SystemExit(76)
            headroom=measured['gpu_total_bytes']-measured['vram_peak_reserved_bytes']
            if headroom<2*1024**3:
                print('Less than2GiB reserved-memory headroom; trying smaller microbatch.',flush=True);continue
            measured.update(microbatch=microbatch,use_checkpoint=checkpointing,headroom_bytes=headroom)
            atomic_json(measured,report)
            summary={key:measured.get(key) for key in ('status','optimizer_steps','optimizer_attempts','amp_skipped_updates',
                'seconds_per_step','images_per_second','dataloader_wait_seconds','dataloader_wait_fraction','numerical_status')}
            summary.update(backbone=measured.get('identity',{}).get('config',{}).get('backbone','large'),microbatch=microbatch,
                effective_batch=measured.get('effective_batch'),amp='fp16',ssm_dtype='fp32',
                use_checkpoint=checkpointing,vram_peak_allocated_GiB=measured.get('vram_peak_allocated_bytes',0)/1024**3,
                vram_peak_reserved_GiB=measured['vram_peak_reserved_bytes']/1024**3,headroom_GiB=headroom/1024**3)
            print('BENCHMARK SUMMARY:',json.dumps(summary),flush=True)
            if measured['status']!='PASS':
                print('PAUSED: below10 images/s; inspect loader wait before continuing.',flush=True)
                raise SystemExit(75)
            verify(report,a.data,a.pretrained)
            atomic_json(dict(base_config_sha256=base_config_hash(),microbatch=microbatch,
                             use_checkpoint=checkpointing,amp='fp16',ssm_dtype='fp32',headroom_bytes=headroom,
                             scope='One-GPU MambaCL probe, not a guarantee for future GL/history VRAM'),
                        out.parent/'execution_profile.json')
            return
    print('PAUSED: no candidate fits with2GiB headroom; review attempts.json',flush=True)
    raise SystemExit(75)

if __name__=='__main__':main()
