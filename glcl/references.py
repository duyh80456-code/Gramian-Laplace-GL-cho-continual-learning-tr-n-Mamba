"""Exact tiny-core references: TEST/DIAGNOSTICS ONLY, never imported by training."""
import torch


def exact_C_matrix(x, h, w):
    # Column-major vectorization of n×D W: vec = W.T.flatten().
    D, n = x.shape[-1], h.shape[-1]
    return torch.einsum('ti,tj,td,tdn,tdm->injm', x, x, w, h, h).reshape(D*n, D*n)


def up_response(C, transitions, v, pre, u, perturbation, delta_scale=1.):
    # Full response: keep all past injections BEFORE squaring the output.
    delta_pre = u @ perturbation.T
    injections = v * (pre.sigmoid() * delta_pre / delta_scale)[:, :, None]
    dh = torch.zeros_like(v[0])
    outputs = []
    for t in range(len(u)):
        dh = transitions[t] * dh + injections[t]
        outputs.append(torch.einsum('dn,n->d', dh, C[t]))
    return torch.stack(outputs)
