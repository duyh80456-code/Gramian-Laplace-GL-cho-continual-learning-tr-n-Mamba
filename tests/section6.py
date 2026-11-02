"""Only the five checks in prompt §6. Run: python -m tests.section6.

No xfail masking: the user-amended approximation thresholds are enforced.
InfSSM is explicitly deferred with all phase-two implementations.
"""
import importlib.util
import json
from pathlib import Path
import torch
from glcl.math import recurrence, backward_gramian, explicit_gramian, kron_value
from glcl.regularizer import OnlineGL


def fixture():
    torch.manual_seed(7)
    dtype = torch.float64
    T, D, n, r = 20, 6, 4, 2
    x = torch.randn(T, D, dtype=dtype) * .3
    B, C = [x @ (torch.randn(D, n, dtype=dtype) * .3) for _ in range(2)]
    A = -torch.rand(D, n, dtype=dtype) - .1
    up = torch.randn(D, r, dtype=dtype) * .2
    u = x @ (torch.randn(D, r, dtype=dtype) * .3)
    pre = u @ up.T - 1
    return x, B, C, A, up, u, pre


def check_C():
    from scipy.stats import spearmanr
    from glcl.math import core_factors, trace_scale
    from glcl.references import exact_C_matrix
    x, B, C, A, up, u, pre = fixture()
    _, h, _ = recurrence(x, B, C, torch.nn.functional.softplus(pre), A)
    w = torch.ones(20, 6, dtype=x.dtype)
    f = core_factors(x, B, C, pre, A, up, u, w)
    M = exact_C_matrix(x, h, w)
    generator = torch.Generator().manual_seed(19)
    exacts, approx, proxies, errors = [], [], [], []
    coefficient = trace_scale(f['cnum'], f['sx'], f['P'])
    for i in range(100):
        dw = torch.randn(4, 6, generator=generator, dtype=x.dtype)
        dy = torch.einsum('tdn,tn->td', h, x @ dw.T)
        exact = dy.square().sum()
        vector = dw.T.flatten()
        errors.append(float(abs(vector @ M @ vector - exact)))
        exacts.append(float(exact))
        approx.append(float(kron_value(dw, f['sx'], f['P'], coefficient)))
        proxies.append(float(kron_value(dw, f['sx'], torch.eye(4, dtype=x.dtype))))
    rho = float(spearmanr(exacts, approx).statistic)
    proxy_rho = float(spearmanr(exacts, proxies).statistic)
    median = float(torch.tensor([abs(a-b)/b for a,b in zip(approx,exacts)], dtype=x.dtype).median())
    passed = max(errors) <= 1e-8 and rho > proxy_rho and rho >= .8
    return dict(status='PASS' if passed else 'FAIL',
                exact_reference=dict(max_absolute_error=max(errors), tolerance=1e-8),
                approximation=dict(noises=100, noise_seed=19, spearman=rho, minimum=.8,
                                   proxy='Sigma_x tensor I (MambaCL-style input-only)',
                                   proxy_spearman=proxy_rho, exceeds_proxy=rho > proxy_rho,
                                   median_relative_error=median, trace_corrected=True))


def check_Q():
    x, B, C, A, up, u, pre = fixture()
    _, _, a = recurrence(x, B, C, torch.nn.functional.softplus(pre), A)
    w = torch.rand(20, 6, dtype=x.dtype)
    error = float((backward_gramian(C, a, w) - explicit_gramian(C, a, w)).abs().max())
    return dict(status='PASS' if error <= 1e-12 else 'FAIL', max_error=error)


def check_up():
    from scipy.stats import spearmanr
    from glcl.math import core_factors
    from glcl.references import up_response
    x, B, C, A, up, u, pre = fixture()
    _, h, a = recurrence(x, B, C, torch.nn.functional.softplus(pre), A)
    previous = torch.cat([torch.zeros_like(h[:1]), h[:-1]])
    v = A * a * previous + x[:, :, None] * B[:, None]
    f = core_factors(x, B, C, pre, A, up, u, torch.ones(20, 6, dtype=x.dtype))
    d = 2
    generator = torch.Generator().manual_seed(23)
    exacts, approximations, proxies, errors = [], [], [], []
    sigma_u = u.T @ u
    def fun(weight):
        return recurrence(x, B, C, torch.nn.functional.softplus(u @ weight.T - 1), A)[0]
    for i in range(100):
        direction = torch.randn(2, generator=generator, dtype=x.dtype)
        perturb = torch.zeros_like(up)
        perturb[d] = direction
        response = up_response(C, a, v, pre, u, perturb)
        _, jvp = torch.autograd.functional.jvp(fun, up, perturb)
        errors.append(float((response-jvp).norm()/jvp.norm()))
        exacts.append(float(jvp.square().sum()))
        approximations.append(float(direction @ f['Hup'][d] @ direction))
        proxies.append(float(direction @ sigma_u @ direction))
    rho = float(spearmanr(exacts, approximations).statistic)
    proxy_rho = float(spearmanr(exacts, proxies).statistic)
    median = float(torch.tensor([abs(a-b)/b for a,b in zip(approximations,exacts)], dtype=x.dtype).median())
    passed = max(errors) <= 1e-6 and rho > proxy_rho and rho >= .85
    return dict(status='PASS' if passed else 'FAIL',
                exact_reference=dict(max_relative_JVP_error=max(errors), tolerance=1e-6),
                approximation=dict(noises=100, noise_seed=23, row=d, spearman=rho, minimum=.85,
                                   proxy='Sigma_u (unweighted input covariance)',
                                   proxy_spearman=proxy_rho, exceeds_proxy=rho > proxy_rho,
                                   median_relative_error=median, cross_time_terms=False))


def check_zero():
    from copy import deepcopy
    spec = importlib.util.spec_from_file_location('official_mod_adam', Path(__file__).resolve().parents[1] / 'upstream/mamba-cl/utils/mod_adam.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = {}
    for projected in (False, True):
        torch.manual_seed(9)
        baseline = torch.nn.Linear(6, 4, dtype=torch.float64)
        candidate = deepcopy(baseline)
        x, y = torch.randn(8, 6, dtype=torch.float64), torch.arange(8) % 4
        opts = []
        for model in (baseline, candidate):
            projection = {id(model.weight): {'interm_reader_out': torch.eye(6, dtype=torch.float64) * .9}} if projected else {}
            opts.append(module.ModAdam(model.parameters(), update_projection_dict=projection, arg_dict={}, lr=.01, foreach=True))
        regularizer = OnlineGL(strength=0)
        for _ in range(20):
            for i, (model, opt) in enumerate(zip((baseline, candidate), opts)):
                opt.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(x), y)
                penalty = regularizer.penalty(model) if i else None
                if penalty is not None:
                    loss = loss + penalty
                loss.backward()
                opt.step()
        results['MambaCL' if projected else 'Seq'] = all(torch.equal(a, b) for a, b in zip(baseline.parameters(), candidate.parameters()))
    return dict(status='PASS' if all(results.values()) else 'FAIL', bitwise_20_steps=results,
                scope='Tiny CPU model and official ModAdam; no end-to-end training.')


def main():
    torch.set_num_threads(1)
    results = {}
    for name, fn in [('1_C_exactness', check_C), ('2_Q_recursion', check_Q), ('3_up_JVP', check_up), ('4_lambda_zero', check_zero)]:
        try:
            results[name] = fn()
        except Exception as exc:
            results[name] = dict(status='ERROR', reason=repr(exc))
    results['5_InfSSM'] = dict(status='DEFERRED', reason='Phase two: prompt §3 explicitly says not to implement InfSSM before phase-one approval.')
    report = dict(dtype='float64', device='cpu', results=results,
                  training_allowed=all(v['status'] == 'PASS' for k, v in results.items() if not k.startswith('5_')))
    Path('reports').mkdir(exist_ok=True)
    Path('reports/section6.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['training_allowed'] else 1)

if __name__ == '__main__':
    main()
