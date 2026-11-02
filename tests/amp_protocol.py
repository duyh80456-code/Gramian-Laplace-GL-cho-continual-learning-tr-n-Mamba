"""CPU-only protocol checks: schedules, resized cache and mismatched resume."""
import json
from pathlib import Path
import tempfile
import os
from unittest.mock import patch
from PIL import Image
from glcl.configuration import load_config,base_config_hash
from glcl.storage import compatible_identity
from glcl.io import digest_sources
from scripts.resize_imagenet_r import prepare

with patch.dict(os.environ):
    os.environ.pop('GL_EXECUTION_PROFILE',None)
    tune,main=load_config('tune'),load_config('main')
    assert tune['epochs']==10 and tune['decay_milestones']==[5,8]
    assert main['epochs']==50 and main['decay_milestones']==[25,40]
    assert tune['use_amp'] and main['use_amp'] and tune['persistent_workers']
    assert tune['microbatch_candidates']==[50,25,10] and tune['batch_size']==200
    assert tune['expand_times']==5
    with tempfile.TemporaryDirectory() as temporary:
        root=Path(temporary);src=root/'source';dest=root/'resize'
        for part,shape in [('train',(800,400)),('val',(300,600))]:
            file=src/part/'n00000000/x.jpg';file.parent.mkdir(parents=True)
            Image.new('RGB',shape,color='red').save(file)
        prepare(src,dest)
        for file in dest.glob('*/*/*.jpg'):
            with Image.open(file) as image:assert min(image.size)==256
        times={str(p):p.stat().st_mtime_ns for p in dest.glob('*/*/*.jpg')}
        prepare(src,dest)
        assert times=={str(p):p.stat().st_mtime_ns for p in dest.glob('*/*/*.jpg')}
        profile=root/'profile.json';profile.write_text(json.dumps(dict(base_config_sha256=base_config_hash(),microbatch=25,use_checkpoint=False)))
        os.environ['GL_EXECUTION_PROFILE']=str(profile)
        assert load_config('main')['microbatch']==25 and load_config('tune')['microbatch']==25
    current=dict(source_sha256=digest_sources(),config=tune,arm='GL')
    old=dict(current,config=dict(tune,epochs=50,use_amp=False))
    assert not compatible_identity(old,current)
print('Tune10/main50 schedules, shared AMP, resize256 cache reuse, execution profile and incompatible old protocol: PASS')
print('No GPU allocation, AMP throughput or T4 fit measured locally')

# OOM attempts run in discarded children: verify50->25->10 and saved common profile.
import importlib.util
import io
from contextlib import redirect_stdout
from types import SimpleNamespace
from glcl.io import ROOT
spec=importlib.util.spec_from_file_location('probe_script',ROOT/'scripts/benchmark_throughput.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);calls=[]
    class Child:
        def __init__(self,command,**kwargs):
            microbatch=int(command[command.index('--microbatch')+1]);calls.append(microbatch)
            self.code=1 if microbatch>10 else 0
            self.stdout=['CUDA out of memory\n'] if self.code else ['probe complete\n']
            if not self.code:
                out=Path(command[command.index('--out')+1])
                (out/'benchmark.json').write_text(json.dumps(dict(status='PASS',gpu_total_bytes=16*1024**3,
                    vram_peak_reserved_bytes=10*1024**3,optimizer_steps=20,gpu_count=1,images_per_second=20)))
        def wait(self):return self.code
    with patch.object(probe.subprocess,'Popen',Child),patch.object(probe,'verify',lambda *args:None), \
         patch('sys.argv',['probe','--data',str(root),'--pretrained',str(root/'base'),'--out',str(root/'throughput_probe')]),redirect_stdout(io.StringIO()):
        probe.main()
    saved=json.loads((root/'execution_profile.json').read_text())
    assert calls==[50,25,10] and saved['microbatch']==10 and not saved['use_checkpoint']
print('Isolated OOM fallback order50/25/10 and shared profile persistence: PASS (mock children, no GPU)')

# A transient save I/O failure must not abort training when capacity is adequate.
from glcl.runner import Runtime
from glcl.storage import StoragePause
runtime=Runtime.__new__(Runtime);runtime.out=ROOT
failure=OSError(5,'injected temporary I/O failure')
def fail(*args,**kwargs):
    raise StoragePause('save failed') from failure
runtime._save=fail
with patch('glcl.runner.shutil.disk_usage',return_value=SimpleNamespace(free=8*1024**3)),redirect_stdout(io.StringIO()):
    assert runtime.save(None,0,1,None) is False
with patch('glcl.runner.shutil.disk_usage',return_value=SimpleNamespace(free=0)):
    try:runtime.save(None,0,1,None)
    except StoragePause:pass
    else:raise AssertionError('Low disk incorrectly continued training')
print('Transient save I/O continues; insufficient disk still pauses: PASS')
