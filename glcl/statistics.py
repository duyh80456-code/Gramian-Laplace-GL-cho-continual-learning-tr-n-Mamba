"""Collect one sampled-label Fisher per image without changing the official scan.

Store temporary activations on host, reconstruct recurrence one block at a time.
Hooks operate only during end-of-task statistics, never baseline training.
"""
import torch
import torch.nn.functional as F
from .math import core_factors
from .regularizer import mixers, preserve_rng


def collect(model, loader, max_samples=2000, seed=0, diagnostics=False):
    import utils.mamba_block as mb
    blocks = mixers(model)
    if not blocks:
        raise RuntimeError('Expected split official MambaMixer parameters')
    old_scan = mb.selective_scan_fn
    cache, totals, handles, current = {}, {}, [], [None]
    modes = {m: m.training for m in model.modules()}
    req = {p: p.requires_grad for p in model.parameters()}

    def capture(name):
        def hook(module, args):
            current[0] = name
        return hook

    def scan(x, delta, A, B, C, *args, **kwargs):
        name = current[0]
        result = old_scan(x, delta, A, B, C, *args, **kwargs)
        y = result[0] if isinstance(result, tuple) else result
        cache.setdefault(name, {}).update(dict(x=x.detach().transpose(1, 2).cpu(),
                           B=B.detach().transpose(1, 2).cpu(),
                           C=C.detach().transpose(1, 2).cpu()))
        y.register_hook(lambda grad, name=name: cache[name].update(w=grad.detach().square().transpose(1, 2).cpu()))
        return result

    def out_hook(name):
        def hook(module, args, output):
            cache[name]['out_x'] = args[0].detach().cpu()
            output.register_hook(lambda grad, name=name: cache[name].update(out_g=grad.detach().cpu()))
        return hook

    def linear_hook(name, key):
        def hook(module, args, output):
            cache.setdefault(name, {})[key + '_x'] = args[0].detach().cpu()
            output.register_hook(lambda grad, name=name, key=key: cache[name].update({key + '_g': grad.detach().cpu()}))
        return hook

    samples = 0
    try:
        model.eval()
        # Enable parameter gradients so raw scan outputs have a gradient graph.
        for m in blocks.values():
            m.A_log.requires_grad_(True)
            if diagnostics:
                from .regularizer import block_parameters
                for parameter in block_parameters(m).values():
                    parameter.requires_grad_(True)
        for name, m in blocks.items():
            handles.append(m.register_forward_pre_hook(capture(name)))
            handles.append(m.out_proj.register_forward_hook(out_hook(name)))
            if diagnostics:
                for key, linear in [('C', m.x_proj_C), ('B', m.x_proj_B), ('dt', m.x_proj_step), ('up', m.dt_proj)]:
                    handles.append(linear.register_forward_hook(linear_hook(name, key)))
        mb.selective_scan_fn = scan
        with preserve_rng():
            torch.manual_seed(seed)
            for images, _ in loader:
                for image in images:
                    if samples >= max_samples:
                        break
                    cache.clear()
                    model.zero_grad(set_to_none=True)
                    logits = model(image[None].to(next(model.parameters()).device)).float()
                    label = torch.multinomial(logits.softmax(-1).detach(), 1)
                    logits.log_softmax(-1).gather(1, label).sum().backward()
                    for name, m in blocks.items():
                        data = cache[name]
                        x, B, C, w = (data[k][0].float() for k in ('x', 'B', 'C', 'w'))
                        up = m.dt_proj.weight.detach().float().cpu()
                        u = F.linear(x, m.x_proj_step.weight.detach().float().cpu())
                        pre = F.linear(u, up, m.dt_proj.bias.detach().float().cpu())
                        rotation = None
                        if m.config.use_rope:
                            n, T = B.shape[-1], len(x)
                            # Build the exact linear map used by add_rope on CPU.
                            theta = (torch.arange(T).float() / m.config.rope_scale_factor)[:, None] * m.freq.detach().float().cpu()[None]
                            eye = torch.eye(n).expand(T, n, n)
                            rotated = torch.stack((-eye[:, 1::2], eye[:, ::2]), dim=2).flatten(1, 2)
                            rotation = eye * theta.cos()[:, :, None] + rotated * theta.sin()[:, :, None]
                        factors = core_factors(x, B, C, pre, -m.A_log.detach().float().cpu().exp(),
                                               up, u, w, m.config.delta_scale_factor, rotation)
                        ox, og = data['out_x'][0].float(), data['out_g'][0].float()
                        factors.update(out_x=ox.T @ ox / len(ox), out_g=og.T @ og / len(og))
                        if diagnostics:
                            from .regularizer import block_parameters
                            for key, parameter in block_parameters(m).items():
                                if parameter.grad is None:
                                    raise RuntimeError('Missing Fisher gradient: ' + name + ':' + key)
                                factors['ewc_' + key] = parameter.grad.detach().float().cpu().square()
                            for key in ('C', 'B', 'dt', 'up'):
                                inp, grad = data[key + '_x'][0].float(), data[key + '_g'][0].float()
                                factors['kfac_' + key + '_x'] = inp.T @ inp / len(inp)
                                factors['kfac_' + key + '_g'] = grad.T @ grad / len(grad)
                        if name not in totals:
                            totals[name] = factors
                        else:
                            for k, value in factors.items():
                                totals[name][k].add_(value)
                    samples += 1
                    if samples % 10 == 0:
                        print(f'GL statistics {samples}/{max_samples}', flush=True)
                if samples >= max_samples:
                    break
        if samples == 0:
            raise RuntimeError('Empty statistics loader')
        return {name: {k: v / samples for k, v in fs.items()} for name, fs in totals.items()}
    finally:
        mb.selective_scan_fn = old_scan
        for h in handles:
            h.remove()
        for m, mode in modes.items():
            m.training = mode
        for p, enabled in req.items():
            p.requires_grad_(enabled)
        model.zero_grad(set_to_none=True)
