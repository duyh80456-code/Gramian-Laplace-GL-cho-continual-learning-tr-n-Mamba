"""Online operator sum. Keep Kronecker summands separate: multiplying averaged
factors would introduce cross-task terms absent from the registered precision.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
from .io import atomic_torch
import torch
from .math import factor_values


def block_parameters(m):
    return dict(C=m.x_proj_C.weight, B=m.x_proj_B.weight,
                dt=m.x_proj_step.weight, A=m.A_log, up=m.dt_proj.weight,
                bias=m.dt_proj.bias, out=m.out_proj.weight)


def mixers(model):
    return {n: m for n, m in model.named_modules()
            if hasattr(m, 'x_proj_C') and hasattr(m, 'A_log')}


class _StreamQuadratic(torch.autograd.Function):
    @staticmethod
    def forward(ctx, state, names, device, *parameters):
        # Stream one task/block onto GPU. Save only the aggregate derivative.
        grads = [torch.zeros_like(p, device='cpu') for p in parameters]
        value = torch.zeros((), device=device)
        lookup = dict(zip(names, parameters))
        for block, history in state.history.items():
            for record in history:
                with torch.enable_grad():
                    d = {k: (lookup[block + ':' + k].detach() - v.to(device)).requires_grad_()
                         for k, v in state.anchor[block].items()}
                    stored = torch.load(Path(state.storage_dir) / record['file'], map_location='cpu') if 'file' in record else record['factors']
                    f = {k: v.to(device) for k, v in stored.items()}
                    v = sum(factor_values(d, f, record['A'].to(device)).values()) * record['weight']
                    derivatives = torch.autograd.grad(v, tuple(d.values()))
                value += v.detach()
                for k, g in zip(d, derivatives):
                    grads[names.index(block + ':' + k)].add_(g.cpu())
        for i, name in enumerate(names):
            block, k = name.rsplit(':', 1)
            delta = parameters[i].detach() - state.anchor[block][k].to(device)
            value += state.mu * delta.square().sum()
            grads[i].add_((2 * state.mu * delta).cpu())
        ctx.save_for_backward(*grads)
        ctx.device = device
        return value

    @staticmethod
    def backward(ctx, scale):
        return (None, None, None, *(g.to(ctx.device) * scale for g in ctx.saved_tensors))


class OnlineGL:
    def __init__(self, strength=1., mu=1e-4, gamma=.95, storage_dir=None):
        self.strength, self.mu, self.gamma = strength, mu, gamma
        self.history, self.anchor, self.anchor_A = {}, {}, {}
        self.storage_dir = str(storage_dir) if storage_dir is not None else None

    def penalty(self, model):
        # Strict zero bypass: no tensor op, RNG draw, hook, or modified optimizer.
        if self.strength == 0 or not self.anchor:
            return None
        names, parameters = [], []
        for name, m in mixers(model).items():
            for k, p in block_parameters(m).items():
                names.append(name + ':' + k)
                parameters.append(p)
        return .5 * self.strength * _StreamQuadratic.apply(self, names, parameters[0].device, *parameters)

    def consolidate(self, model, statistics):
        for history in self.history.values():
            for record in history:
                record['weight'] *= self.gamma
        for name, m in mixers(model).items():
            record = dict(weight=1., A=-m.A_log.detach().cpu().exp())
            if self.storage_dir is not None:
                filename = hashlib.sha256(name.encode()).hexdigest()[:16] + f'_{len(self.history.get(name, [])):02d}.pt'
                atomic_torch(statistics[name], Path(self.storage_dir) / filename)
                record.update(file=filename, floats=sum(v.numel() for v in statistics[name].values()))
            else:
                record['factors'] = statistics[name]
            self.history.setdefault(name, []).append(record)
            self.anchor[name] = {k: p.detach().cpu().clone() for k, p in block_parameters(m).items()}
            self.anchor_A[name] = -m.A_log.detach().cpu().exp()

    def state_dict(self):
        return vars(self).copy()

    def load_state_dict(self, state):
        self.__dict__.update(state)

    def floats(self):
        def count(x):
            if torch.is_tensor(x):
                return x.numel()
            if isinstance(x, dict):
                return sum(count(v) for v in x.values())
            if isinstance(x, list):
                return sum(count(v) for v in x)
            return 0
        return count(self.state_dict()) + sum(record.get('floats', 0) for history in self.history.values() for record in history)


@contextmanager
def preserve_rng():
    import random
    import numpy as np
    py, npstate = random.getstate(), np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(py)
        np.random.set_state(npstate)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
