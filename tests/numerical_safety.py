"""CPU regressions for nonfinite logging, fp32 SSM wrapping and probe rejection."""
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import tempfile
import torch
from glcl.numerics import NumericalError,require_finite,require_finite_parameters,enable_fp32_ssm
from glcl.telemetry import StepMeter
from glcl.runner import Runtime

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary)
    meter=StepMeter(root,'GL',lambda:dict(allocated=0,reserved=0))
    with redirect_stdout(io.StringIO()):meter.record(0,1,1,2,.1,.2,loss=float('nan'),train_acc=50.)
    text=(root/'progress.json').read_text()
    result=json.loads(text)
    assert result['loss'] is None and result['nonfinite_fields']==['loss']
    assert result['numerical_status']=='NONFINITE' and 'NaN' not in text
    for invalid in (float('nan'),float('inf'),-float('inf')):
        try:require_finite(torch.tensor(invalid),'loss')
        except NumericalError:pass
        else:raise AssertionError('Invalid loss accepted')
    try:require_finite_parameters([('bad_weight',torch.tensor(float('nan')))],'checkpoint')
    except NumericalError as error:assert 'bad_weight' in str(error)
    else:raise AssertionError('Invalid checkpoint accepted')
    print('NaN-safe diagnostic JSON and explicit nonfinite loss/weight rejection: PASS')

    class Mixer(torch.nn.Module):
        def __init__(self):super().__init__();self.A_log=torch.nn.Parameter(torch.zeros(1));self.seen=[]
        def forward_ssm(self,x,gate):
            self.seen.append((x.dtype,gate.dtype,torch.is_autocast_cpu_enabled()))
            return x*gate
    model=torch.nn.Sequential(Mixer())
    assert enable_fp32_ssm(model)==['0'] and enable_fp32_ssm(model)==[]
    x=torch.ones(2,dtype=torch.float16,requires_grad=True)
    with torch.autocast('cpu',dtype=torch.bfloat16):y=model[0].forward_ssm(x,x)
    assert model[0].seen==[(torch.float32,torch.float32,False)] and y.dtype==torch.float32
    y.sum().backward();assert torch.isfinite(x.grad).all()
    print('FP32 SSM island disables surrounding autocast, preserves gradients and is idempotent: PASS')

    # Inject a bad forward into the actual optimizer loop: it must fail before step.
    runtime=Runtime.__new__(Runtime)
    runtime.args=SimpleNamespace(arm='MambaCL',benchmark_steps=20)
    runtime.config=dict(use_amp=False,microbatch=1)
    runtime.resume=None;runtime.scaler=None;runtime.meter=None;runtime.gpu_utilization=None
    runtime.optimizer_updates=0;runtime.out=root/'probe';runtime.out.mkdir()
    runtime.check_next_checkpoint=lambda *args:None
    class Bad(torch.nn.Module):
        def __init__(self):super().__init__();self.head=torch.nn.Linear(3,2)
        def forward(self,x):return self.head(x)*float('nan')
    bad=Bad();before={k:v.detach().clone() for k,v in bad.state_dict().items()}
    optimizer=torch.optim.SGD(bad.parameters(),lr=.1)
    gvm=SimpleNamespace(args=SimpleNamespace(temperature=1.),cl_mngr=SimpleNamespace(current_taskid=0))
    with patch('glcl.telemetry.GPUUtilization',return_value=SimpleNamespace(drain=lambda:None)), \
         patch('glcl.telemetry.check_free'),patch('torch.cuda.synchronize'),patch('torch.cuda.reset_peak_memory_stats'), \
         patch.object(torch.Tensor,'cuda',lambda self,**kwargs:self),patch.object(optimizer,'step',wraps=optimizer.step) as update:
        try:runtime.train_epoch(gvm,1,[(torch.ones(2,3),torch.tensor([0,1]))],bad,torch.nn.CrossEntropyLoss(),optimizer)
        except NumericalError as error:assert 'logits' in str(error)
        else:raise AssertionError('Bad forward did not fail')
        update.assert_not_called()
    assert all(torch.equal(before[k],v) for k,v in bad.state_dict().items())
    assert not (runtime.out/'latest.pt').exists()
    print('Injected nonfinite forward stops before optimizer mutation/checkpoint: PASS')
print('CPU fixtures only; no claim that actual T4/SSM NaN is reproduced or resolved')

# Parent must propagate numerical failure and never write a fit profile or try smaller batches.
import importlib.util
from glcl.io import ROOT
spec=importlib.util.spec_from_file_location('numerical_probe_script',ROOT/'scripts/benchmark_throughput.py')
probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);calls=[]
    class FailedChild:
        def __init__(self,command,**kwargs):
            calls.append(command)
            directory=Path(command[command.index('--out')+1])
            (directory/'benchmark.json').write_text(json.dumps(dict(status='NUMERICAL_FAILURE',reason='Nonfinite logits',training_allowed=False)))
            self.stdout=['NUMERICAL_FAILURE: Nonfinite logits\n']
        def wait(self):return 76
    with patch.object(probe.subprocess,'Popen',FailedChild), \
         patch('sys.argv',['probe','--data',str(root),'--pretrained',str(root/'base'),'--out',str(root/'throughput_probe')]),redirect_stdout(io.StringIO()):
        try:probe.main()
        except SystemExit as result:assert result.code==76
        else:raise AssertionError('Invalid probe approved')
    assert len(calls)==1 and not (root/'execution_profile.json').exists()
    assert json.loads((root/'throughput_probe/benchmark.json').read_text())['status']=='NUMERICAL_FAILURE'
print('Probe parent preserves numerical reason, blocks tuning and writes no execution profile: PASS')
