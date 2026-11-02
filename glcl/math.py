"""Prompt §4, with explicit sample/token conventions. Never form dense Kroneckers.

Inputs to core_factors have no batch axis: accumulate per sample, then average.
Q contains future output-diagonal weights, not cross-time GGN terms.
"""
import torch
import torch.nn.functional as F


def recurrence(x, B, C, delta, A):
    h = x.new_zeros(A.shape)
    states, transitions = [], []
    for t in range(len(x)):
        a = torch.exp(delta[t, :, None] * A)
        h = a * h + delta[t, :, None] * x[t, :, None] * B[t]
        transitions.append(a)
        states.append(h)
    h = torch.stack(states)
    return torch.einsum('tdn,tn->td', h, C), h, torch.stack(transitions)


def backward_gramian(C, transitions, w):
    """Q_s = w_s C_s C_s^T + diag(a_{s+1}) Q_{s+1} diag(a_{s+1})."""
    T, D, n = transitions.shape
    q = C.new_zeros(D, n, n)
    result = []
    for t in range(T - 1, -1, -1):
        if t < T - 1:
            a = transitions[t + 1]
            q = q * a[:, :, None] * a[:, None, :]
        q = q + w[t, :, None, None] * torch.outer(C[t], C[t])[None]
        result.append(q)
    return torch.stack(result[::-1])


def explicit_gramian(C, transitions, w):
    T, D, n = transitions.shape
    qs = []
    for s in range(T):
        phi = C.new_ones(D, n)
        q = C.new_zeros(D, n, n)
        for t in range(s, T):
            if t > s:
                phi = phi * transitions[t]
            v = phi * C[t]
            q = q + w[t, :, None, None] * v[:, :, None] * v[:, None, :]
        qs.append(q)
    return torch.stack(qs)


def kron_value(dw, left, right, scale=1.):
    return scale * torch.sum((dw @ left) * (right @ dw))


def trace_scale(numerator, left, right):
    denominator = left.trace() * right.trace()
    if denominator.item() <= 0:
        return denominator.new_zeros(())
    return numerator / denominator


def core_factors(x, B, C, pre, A, up, u, w, delta_scale=1., rotation=None):
    """§4.2–4.4 for a single sample; RoPE factors pulled back to x_proj rows.

    rotation[t] maps unrotated state coefficients to actual B_t/C_t.
    Returned trace numerators must be averaged BEFORE computing c.
    """
    T, D = x.shape
    delta = F.softplus(pre) / delta_scale
    _, h, a = recurrence(x, B, C, delta, A)
    Q = backward_gramian(C, a, w)
    prev = torch.cat([h.new_zeros(1, *h.shape[1:]), h[:-1]])
    # Avoid materializing a T×D×n×n outer-product for h.
    if rotation is None:
        hc = h
        qb = Q.sum(1)
    else:
        hc = torch.einsum('tdn,tnm->tdm', h, rotation)
        qb = torch.einsum('tni,tnm,tmj->tij', rotation, Q.sum(1), rotation)
    P = torch.einsum('td,tdn,tdm->nm', w, hc, hc)
    z = (delta * x).sum(1, keepdim=True) * x
    sx, sz = x.T @ x, z.T @ z
    qw = qb.sum(0) / T
    cnum = (w * hc.square().sum(-1) * x.square().sum(-1)[:, None]).sum()
    bnum = (z.square().sum(-1) * qb.diagonal(dim1=-2, dim2=-1).sum(-1)).sum() / T
    q = delta[:, :, None] * a * prev
    HA = torch.einsum('tdn,tdm,tdnm->dnm', q, q, Q)
    v = A * a * prev + x[:, :, None] * B[:, None, :]
    kappa = torch.einsum('tdn,tdnm,tdm->td', v, Q, v)
    s = (torch.sigmoid(pre) / delta_scale).square() * kappa
    Hup = torch.einsum('td,tr,ts->drs', s, u, u)
    sb = s.sum(0)
    right_dt = up.T @ ((sb / T)[:, None] * up)
    dt_num = (x.square().sum(-1)[:, None] * s * up.square().sum(1)[None]).sum()
    return dict(sx=sx, sz=sz, P=P, Q=qw, cnum=cnum, bnum=bnum,
                HA=HA, Hup=Hup, bias=sb, dt_right=right_dt, dt_num=dt_num)


def factor_values(delta, factors, A):
    """Per-parameter quadratic contributions at an online anchor."""
    f = factors
    return {
        'C': kron_value(delta['C'], f['sx'], f['P'], trace_scale(f['cnum'], f['sx'], f['P'])),
        'B': kron_value(delta['B'], f['sz'], f['Q'], trace_scale(f['bnum'], f['sz'], f['Q'])),
        'dt': kron_value(delta['dt'], f['sx'], f['dt_right'], trace_scale(f['dt_num'], f['sx'], f['dt_right'])),
        'A': torch.einsum('dn,dnm,dm->', A * delta['A'], f['HA'], A * delta['A']),
        'up': torch.einsum('dr,drs,ds->', delta['up'], f['Hup'], delta['up']),
        'bias': (delta['bias'].square() * f['bias']).sum(),
        'out': kron_value(delta['out'], f['out_x'], f['out_g']),
    }
