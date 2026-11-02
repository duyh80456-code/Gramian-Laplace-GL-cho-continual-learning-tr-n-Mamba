"""§5: exact Jv via rows of the reverse-mode Jacobian of official logits.

This avoids double backward/forward-AD unsupported by selective_scan kernels.
64 temporary images, 50 fixed isotropic directions per parameter block. They are
measurements only; no diagnostic result affects optimization or selection.
"""
import torch
from .math import factor_values, kron_value
from .regularizer import block_parameters, mixers, preserve_rng
from .io import atomic_json


def measure(model, loader, statistics, output, samples=64, directions=50):
    from scipy.stats import spearmanr
    blocks = mixers(model)
    original_modes = {m: m.training for m in model.modules()}
    original_req = {p: p.requires_grad for p in model.parameters()}
    entries = []
    for name, m in blocks.items():
        for key, parameter in block_parameters(m).items():
            if key != 'bias':
                entries.append((name, key, parameter))
    # Bias is registered as part of up and perturbed separately only in the loss.
    exact = {(name, key): torch.zeros(directions, dtype=torch.float64) for name, key, p in entries}
    seeds = {(name, key): 5000 + i for i, (name, key, p) in enumerate(entries)}
    def noise(name, key, parameter):
        g = torch.Generator().manual_seed(seeds[name, key])
        vectors = torch.randn(directions, parameter.numel(), generator=g, dtype=torch.float32)
        return vectors / vectors.norm(dim=1, keepdim=True)
    count = 0
    try:
        model.eval()
        for name, key, parameter in entries:
            parameter.requires_grad_(True)
        with preserve_rng():
            torch.manual_seed(3000)
            for images, _ in loader:
                for image in images:
                    if count >= samples:
                        break
                    logits = model(image[None].to(next(model.parameters()).device)).float()[0]
                    p = logits.softmax(-1).detach().cpu().double()
                    projections = {(name, key): torch.empty(len(logits), directions, dtype=torch.float64) for name, key, parameter in entries}
                    for c in range(len(logits)):
                        gradients = torch.autograd.grad(logits[c], [e[2] for e in entries], retain_graph=c + 1 < len(logits), allow_unused=False)
                        for (name, key, parameter), gradient in zip(entries, gradients):
                            vectors = noise(name, key, parameter)
                            projections[name, key][c] = (vectors @ gradient.detach().float().cpu().flatten()).double()
                    for name, key, parameter in entries:
                        jv = projections[name, key]
                        exact[name, key] += (p[:, None] * jv.square()).sum(0) - (p[:, None] * jv).sum(0).square()
                    count += 1
                    print(f'Diagnostics {count}/{samples}', flush=True)
                if count >= samples:
                    break
            if count != samples:
                raise RuntimeError('Diagnostics requires exactly 64 images')
            rows = []
            for name, key, parameter in entries:
                fs = statistics[name]
                m = blocks[name]
                A = -m.A_log.detach().float().cpu().exp()
                vectors = noise(name, key, parameter).reshape(directions, *parameter.shape)
                scores = {method: [] for method in ('GL', 'KFAC', 'EWC', 'MambaCL_input_proxy')}
                params = block_parameters(m)
                for vector in vectors:
                    delta = {k: torch.zeros_like(v, device='cpu') for k, v in params.items()}
                    delta[key] = vector
                    scores['GL'].append(float(factor_values(delta, fs, A)[key]))
                    ewc = float((vector.square() * fs['ewc_' + key]).sum())
                    scores['EWC'].append(ewc)
                    if key == 'A':
                        scores['KFAC'].append(ewc)
                        # Original MambaCL A projector uses token-position delta covariance;
                        # it cannot act on A_log's state axis without its native update map.
                        scores['MambaCL_input_proxy'].append(float('nan'))
                    else:
                        left = fs['out_x'] if key == 'out' else fs['kfac_' + key + '_x']
                        right = fs['out_g'] if key == 'out' else fs['kfac_' + key + '_g']
                        scores['KFAC'].append(float(kron_value(vector, left, right)))
                        # dt_proj is not covered by the original projector; up proxy
                        # is merely its input-only covariance and labeled as such.
                        scores['MambaCL_input_proxy'].append(float(((vector @ left) * vector).sum()))
                values = {}
                for method, predicted in scores.items():
                    rho = float(spearmanr(predicted, exact[name, key].numpy()).statistic)
                    values[method] = rho if torch.isfinite(torch.tensor(rho)) else None
                rows.append(dict(block=name, parameter=key, spearman=values))
            names = list(blocks)
            averages = {}
            for depth, selected in [('shallow', names[:len(names)//2]), ('deep', names[len(names)//2:])]:
                averages[depth] = {}
                for method in scores:
                    values = [row['spearman'][method] for row in rows if row['block'] in selected and row['spearman'][method] is not None]
                    averages[depth][method] = sum(values) / len(values) if values else None
            atomic_json(dict(samples=count, directions=directions, rows=rows, depth_average=averages,
                             jvp='Exact Jv using reverse Jacobian rows; no finite difference',
                             missing_proxy_A='Original A projector acts on token positions; no compatible input-only state metric specified.'), output)
    finally:
        for m, mode in original_modes.items():
            m.training = mode
        for p, req in original_req.items():
            p.requires_grad_(req)
        model.zero_grad(set_to_none=True)
