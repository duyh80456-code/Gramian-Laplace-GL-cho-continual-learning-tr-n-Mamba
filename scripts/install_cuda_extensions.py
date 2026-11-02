"""Install pinned official CUDA wheels, without source builds or Torch upgrades."""
import argparse
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = (
    ('causal-conv1d', 'causal_conv1d', '1.4.0', 'Dao-AILab/causal-conv1d'),
    ('mamba-ssm', 'mamba_ssm', '2.2.2', 'state-spaces/mamba'),
)
API_CHECK = """
import sys
from importlib.metadata import version
import torch
import causal_conv1d_cuda
import selective_scan_cuda
from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
sys.path.insert(0, sys.argv[1])
from utils.mamba_block import is_fast_path_available
assert is_fast_path_available, 'Official Mamba-CL fast-path APIs are unavailable'
assert version('causal-conv1d').split('+')[0] == '1.4.0'
assert version('mamba-ssm').split('+')[0] == '2.2.2'
print('CUDA extension imports and official Mamba-CL fast-path APIs: PASSED')
"""


def wheel_plan(info):
    if info['python'] != '3.11' or info['platform'] != 'linux_x86_64':
        raise RuntimeError('Expected the isolated Linux x86_64 Python3.11 environment; got ' + str(info))
    if info['torch'].split('+')[0] != '2.2.2':
        raise RuntimeError('Pinned Torch2.2.2 is required; rerun the base dependency cell')
    # Same mapping as both official v1.4.0/v2.2.2 setup.py wheel selectors:
    # Torch CUDA11.x -> cu118; CUDA12.x -> cu122, independent of host nvcc.
    major = str(info['cuda']).split('.')[0]
    cuda = {'11': '118', '12': '122'}.get(major)
    if cuda is None:
        raise RuntimeError('Expected CUDA-enabled Torch with CUDA11.x/12.x; got ' + str(info['cuda']))
    abi = str(info['cxx11_abi']).upper()
    if abi not in ('TRUE', 'FALSE'):
        raise RuntimeError('Invalid Torch C++ ABI value')
    urls = []
    for distribution, module, release, repository in PACKAGES:
        wheel = f'{module}-{release}+cu{cuda}torch2.2cxx11abi{abi}-cp311-cp311-linux_x86_64.whl'
        urls.append(f'https://github.com/{repository}/releases/download/v{release}/' + quote(wheel, safe=''))
    return urls


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def run_logged(command, log, env):
    print('RUN:', ' '.join(command), flush=True)
    log.write('RUN: ' + ' '.join(command) + '\n')
    log.flush()
    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        return process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--force-reinstall', action='store_true')
    args = parser.parse_args()
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    result = dict(status='FAILED', method='official binary wheels; no source fallback')
    env = dict(os.environ, PYTHONUNBUFFERED='1', PIP_DISABLE_PIP_VERSION_CHECK='1')
    for key in ('MAMBA_FORCE_BUILD', 'CAUSAL_CONV1D_FORCE_BUILD'):
        env.pop(key, None)
    try:
        import torch
        info = dict(python=f'{sys.version_info.major}.{sys.version_info.minor}',
                    executable=sys.executable, platform=f'{sys.platform}_{platform.machine()}',
                    torch=torch.__version__, cuda=torch.version.cuda,
                    cxx11_abi=bool(torch._C._GLIBCXX_USE_CXX11_ABI))
        result['runtime'] = info
        urls = wheel_plan(info)
        result['wheel_urls'] = urls
        print('CUDA wheel environment:', json.dumps(info), flush=True)
        check = [sys.executable, '-c', API_CHECK, str(ROOT / 'upstream/mamba-cl')]
        installed = {}
        for distribution, module, release, repository in PACKAGES:
            try:
                installed[distribution] = version(distribution)
            except PackageNotFoundError:
                installed[distribution] = None
        matching = all(installed[distribution] and installed[distribution].split('+')[0] == release
                       for distribution, module, release, repository in PACKAGES)
        with log_path.open('a') as log:
            reused = matching and not args.force_reinstall and run_logged(check, log, env) == 0
            if not reused:
                # Direct .whl URLs and --only-binary prohibit a hidden compiler
                # fallback. --no-deps preserves Torch/vision and all pinned deps.
                command = [sys.executable, '-m', 'pip', 'install', '--no-deps',
                           '--only-binary=:all:', '--force-reinstall', '--progress-bar', 'off', *urls]
                if run_logged(command, log, env) != 0:
                    raise RuntimeError('Official wheel install failed; the complete pip output is in ' + str(log_path))
                if run_logged(check, log, env) != 0:
                    raise RuntimeError('Installed wheels did not load against this runtime; read ' + str(log_path))
            result.update(status='PASSED', reused_existing_install=bool(reused),
                          installed_versions={item[0]: version(item[0]) for item in PACKAGES},
                          verification='imports/API availability only; no model forward or benchmark')
        atomic_json(result, args.report)
        print('CUDA extensions ready; log:', log_path, 'report:', args.report, flush=True)
    except Exception as error:
        result['error'] = str(error)
        with log_path.open('a') as log:
            traceback.print_exc(file=log)
        atomic_json(result, args.report)
        print('CUDA installation failed. Full log:', log_path, file=sys.stderr, flush=True)
        raise


if __name__ == '__main__':
    main()
