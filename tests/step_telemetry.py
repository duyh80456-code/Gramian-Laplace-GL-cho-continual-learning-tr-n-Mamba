"""CPU-only checks of timing aggregation, progress, observer and throughput gate."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch
from glcl.io import ROOT, digest_sources
from glcl.telemetry import StepMeter, check_free
from glcl.storage import StoragePause
from scripts.live_performance import metric_event

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    output = io.StringIO()
    meter = StepMeter(root, 'MambaCL', lambda:dict(allocated=100, reserved=200))
    with redirect_stdout(output):
        for step in range(20):
            meter.record(0, 1+step//10, 1+step%10, 200, 2., 8., loss=.5, train_acc=75.)
            progress = json.loads((root/'progress.json').read_text())
            assert progress['optimizer_step'] == step+1
    lines = output.getvalue().splitlines()
    assert len(lines) == 4
    parsed = metric_event(lines[0], 'tune')
    assert parsed[0] == 'TRAIN_STEP' and parsed[1]['loss'] == .5
    assert parsed[1]['train_acc'] == 75.
    for field in ('images_per_second','dataloader_wait_fraction','vram_peak_allocated_bytes','gpu_utilization_percent'):
        assert field not in parsed[1] and field not in parsed[2] and field not in output.getvalue()
    assert 'loss=0.5000' in parsed[2] and 'train_acc=75.00%' in parsed[2]
    # Diagnostic files retain detailed measurements without printing them.
    assert progress['images_per_second'] == 20 and progress['dataloader_wait_fraction'] == .2
    assert progress['vram_peak_allocated_bytes'] == 100
    legacy=metric_event('STEP_METRICS '+json.dumps(dict(task=1,epoch=1,optimizer_step=5,loss=.4,images_per_second=99,vram_peak_allocated_bytes=200)), 'tune')
    assert 'images_per_second' not in legacy[1] and 'vram' not in legacy[2]
    assert metric_event('Task [1/10]:: Training time = 2h','tune') is None
    validation=metric_event(':: ** Results of task [1]: class_inc_last_acc=80.00% class_inc_last_forg=3.00%','tune')
    assert validation[2]=='Task [1] val_acc=80.00%' and 'forgetting' not in validation[1]
    assert meter.summary()['optimizer_steps'] == 20
    assert meter.summary()['images_per_second'] == 20
    with patch('glcl.telemetry.shutil.disk_usage') as disk:
        disk.return_value.free = 2*1024**3-1
        with patch.dict(os.environ, GL_MIN_FREE_BYTES=str(2*1024**3)):
            try:check_free(root)
            except StoragePause:pass
            else:raise AssertionError('Below2GiB did not pause')
    print('Every5-step aggregation, each-step progress, observer parsing and2GiB pause: PASS')
    spec=importlib.util.spec_from_file_location('benchmark_throughput', ROOT/'scripts/benchmark_throughput.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    data=root/'data'
    for part in ('train', 'val'):
        (data/part/'class').mkdir(parents=True)
        (data/part/'class/image.jpg').write_bytes(b'test')
    pretrained=root/'base.pth';pretrained.write_bytes(b'fixture')
    from glcl.runner import Runtime
    identity=dict(arm='MambaCL',stage='tune',config=__import__('glcl.configuration',fromlist=['load_config']).load_config('tune'),
                  source_sha256=digest_sources(),dataset_manifest=Runtime.data_manifest(data),
                  pretrained_sha256=Runtime.file_hash(pretrained))
    report=dict(status='PASS',optimizer_steps=20,gpu_count=1,images_per_second=20.,identity=identity,microbatch=50,use_checkpoint=False)
    path=root/'benchmark.json';path.write_text(json.dumps(report))
    assert module.verify(path,data,pretrained)['status']=='PASS'
    for bad in [dict(status='SLOW',images_per_second=9.),dict(optimizer_steps=19),dict(gpu_count=2)]:
        path.write_text(json.dumps(dict(report,**bad)))
        try:module.verify(path,data,pretrained)
        except RuntimeError:pass
        else:raise AssertionError('Invalid benchmark was accepted')
    changed=copy.deepcopy(report);changed['identity']['source_sha256']='wrong'
    path.write_text(json.dumps(changed))
    try:module.verify(path,data,pretrained)
    except RuntimeError:pass
    else:raise AssertionError('Stale source was accepted')
    print('Throughput gate rejects slow/incomplete/multi-GPU/stale reports: PASS')
# Exercise the real instrumented optimizer loop on a tiny CPU model, no upstream imports.
    from types import SimpleNamespace
    import torch
    from glcl.runner import BenchmarkComplete
    runtime = Runtime.__new__(Runtime)
    runtime.args = SimpleNamespace(arm='MambaCL',benchmark_steps=20,data_root=str(data),pretrained=str(pretrained))
    runtime.config = dict(workers=2,batch_size=2,microbatch=1)
    runtime.resume = None;runtime.meter = None;runtime.gpu_utilization = None;runtime.scaler = None;runtime.optimizer_updates = 0
    runtime.out = root/'probe';runtime.out.mkdir()
    runtime.identity = identity
    runtime.check_next_checkpoint = lambda *args:None
    gvm = SimpleNamespace(args=SimpleNamespace(temperature=1.),cl_mngr=SimpleNamespace(current_taskid=0))
    network = torch.nn.Linear(3,2)
    optimizer = torch.optim.SGD(network.parameters(),lr=.01)
    batches = [(torch.ones(2,3),torch.tensor([0,1])) for _ in range(11)]
    class NoSampler:
        def drain(self):return None
    with patch('glcl.telemetry.GPUUtilization',return_value=NoSampler()), patch('glcl.telemetry.check_free'), \
         patch('torch.cuda.synchronize'), patch('torch.cuda.reset_peak_memory_stats'), \
         patch('torch.cuda.get_device_properties',return_value=SimpleNamespace(total_memory=16*1024**3)), patch('torch.cuda.max_memory_allocated',return_value=0), patch('torch.cuda.max_memory_reserved',return_value=0), \
         patch.object(torch.Tensor,'cuda',lambda self,**kwargs:self), redirect_stdout(io.StringIO()):
        runtime.train_epoch(gvm,1,batches,network,torch.nn.CrossEntropyLoss(),optimizer)
        try:runtime.train_epoch(gvm,2,batches,network,torch.nn.CrossEntropyLoss(),optimizer)
        except BenchmarkComplete:pass
        else:raise AssertionError('Did not stop exactly at20 optimizer steps')
    result = json.loads((runtime.out/'benchmark.json').read_text())
    assert result['optimizer_steps']==20 and result['images']==40
    assert not (runtime.out/'latest.pt').exists()
    print('Real step-loop CPU fixture stops at20 updates across epochs without a training checkpoint: PASS')
print('Timing fixtures only; no GPU throughput measurement performed')
