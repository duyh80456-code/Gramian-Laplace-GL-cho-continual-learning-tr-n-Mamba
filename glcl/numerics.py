"""Explicit finite-state guards and an fp32 SSM island inside fp16 AMP."""
from functools import wraps
import torch


class NumericalError(RuntimeError):
    """Invalid model computation; never approve the probe or save bad weights."""


def require_finite(value, phase):
    if not bool(torch.isfinite(value.detach()).all()):
        raise NumericalError('Nonfinite '+phase)


def parameters_finite(named):
    values=list(named)
    if not values:return True
    devices={}
    for _,value in values:
        devices.setdefault(str(value.device),[]).append(torch.isfinite(value.detach()).all())
    return all(bool(torch.stack(flags).all()) for flags in devices.values())


def require_finite_parameters(named,phase):
    values=list(named)
    if not parameters_finite(values):
        bad=[name for name,value in values if not bool(torch.isfinite(value.detach()).all())]
        raise NumericalError('Nonfinite '+phase+': '+', '.join(bad[:12]))


def enable_fp32_ssm(model):
    """Keep projections/delta/scan/group norm/gate of forward_ssm in fp32.

    The upstream scan returns its input dtype before group normalization. Under
    fp16 AMP that can overflow before the subsequent .float() conversion.
    Outer input/output projections and conv still use their AMP policy.
    """
    changed=[]
    for name,module in model.named_modules():
        if not hasattr(module,'A_log') or not callable(getattr(module,'forward_ssm',None)):
            continue
        if getattr(module,'_glcl_fp32_ssm',False):continue
        original=module.forward_ssm
        @wraps(original)
        def forward(hidden_states,gate,_original=original):
            with torch.autocast(device_type=hidden_states.device.type,enabled=False):
                return _original(hidden_states.float(),gate.float())
        module.forward_ssm=forward
        module._glcl_fp32_ssm=True
        changed.append(name)
    return changed


def optimizer_state_tensors(optimizer):
    for index,state in enumerate(optimizer.state.values()):
        for key,value in state.items():
            if torch.is_tensor(value):yield f'optimizer_state[{index}].{key}',value
