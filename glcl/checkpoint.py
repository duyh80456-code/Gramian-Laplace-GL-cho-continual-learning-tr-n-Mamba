"""Lossless model adapters plus immutable task context for bounded checkpoints."""
from pathlib import Path
import re
import zlib
import numpy as np
import torch
from .io import atomic_torch
from .storage import sha256_file, artifact_path

_TRIANGLES = {}


def encode_cache(value, minimum_bytes=1024**2):
    """Lossless packing of large square covariances/projectors.

    Store the upper triangle and XOR bits for the lower triangle. The XOR
    preserves every asymmetric rounding bit, signed zero and NaN payload.
    No symmetry assumption, symmetrization or precision reduction is used.
    """
    if (torch.is_tensor(value) and value.device.type == 'cpu' and value.is_contiguous()
            and value.ndim == 2 and value.shape[0] == value.shape[1]
            and value.dtype in (torch.float32, torch.float64)
            and value.numel()*value.element_size() >= minimum_bytes):
        array = value.detach().numpy()
        n, width = array.shape[0], array.dtype.itemsize
        if n not in _TRIANGLES:
            rows, cols = np.triu_indices(n)
            off = rows != cols
            _TRIANGLES[n] = rows, cols, off
        rows, cols, off = _TRIANGLES[n]
        unsigned = np.dtype('u' + str(width))
        upper = array[rows, cols].copy()
        xor = np.bitwise_xor(array[cols[off], rows[off]].view(unsigned), upper[off].view(unsigned))
        def compress(a):
            shuffled = a.view(np.uint8).reshape(-1, width).T.copy().tobytes()
            return zlib.compress(shuffled, level=1)
        packed = dict(__glcl_square_bits_v1__=True, dtype=array.dtype.str, n=n,
                      upper=compress(upper), lower_xor=compress(xor))
        if len(packed['upper']) + len(packed['lower_xor']) < array.nbytes:
            return packed
    if isinstance(value, dict):
        return type(value)((k, encode_cache(v, minimum_bytes)) for k, v in value.items())
    if isinstance(value, list):
        return [encode_cache(v, minimum_bytes) for v in value]
    if isinstance(value, tuple):
        return tuple(encode_cache(v, minimum_bytes) for v in value)
    return value


def decode_cache(value):
    if isinstance(value, dict) and value.get('__glcl_square_bits_v1__') is True:
        dtype = np.dtype(value['dtype'])
        n, width = int(value['n']), dtype.itemsize
        rows, cols = np.triu_indices(n)
        off = rows != cols
        def decompress(data, target_dtype):
            raw = np.frombuffer(zlib.decompress(data), dtype=np.uint8)
            return raw.reshape(width, -1).T.copy().reshape(-1).view(target_dtype)
        upper = decompress(value['upper'], dtype)
        xor = decompress(value['lower_xor'], np.dtype('u' + str(width)))
        array = np.empty((n, n), dtype=dtype)
        array[rows, cols] = upper
        lower = np.bitwise_xor(upper[off].view(xor.dtype), xor).view(dtype)
        array[cols[off], rows[off]] = lower
        return torch.from_numpy(array)
    if isinstance(value, dict):
        return type(value)((k, decode_cache(v)) for k, v in value.items())
    if isinstance(value, list):
        return [decode_cache(v) for v in value]
    if isinstance(value, tuple):
        return tuple(decode_cache(v) for v in value)
    return value


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return type(value)((k, cpu_tree(v)) for k, v in value.items())
    if isinstance(value, list):
        return [cpu_tree(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(v) for v in value)
    return value


def model_keys(model, patterns):
    return ({name for name, parameter in model.named_parameters() if any(s in name for s in patterns)}
            | {name for name, buffer in model.named_buffers()})


def model_snapshot(model, patterns, cpu=True):
    keys = model_keys(model, patterns)
    return {name: tensor.detach().cpu() if cpu else tensor.detach()
            for name, tensor in model.state_dict().items() if name in keys}


def load_model(model, state, patterns, model_format=None):
    if model_format is None:  # Existing full-state epoch1 checkpoints.
        return model.load_state_dict(state, strict=True)
    if model_format != 'trainable_plus_buffers_v1':
        raise ValueError('Unknown checkpoint model format: ' + str(model_format))
    expected = model_keys(model, patterns)
    if set(state) != expected:
        raise ValueError('Adapter keys do not match the configured trainable set: '
                         + str(dict(missing=sorted(expected-set(state)), extra=sorted(set(state)-expected))))
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise ValueError(str(result))
    return result


def write_context(out, context):
    out = Path(out)
    pending = out / 'contexts/context-pending.pt'
    atomic_torch(cpu_tree(context), pending)
    digest = sha256_file(pending)
    relative = Path('contexts') / f'context-{digest}.pt'
    target = out / relative
    if target.exists():
        pending.unlink()
    else:
        pending.replace(target)
    return str(relative), digest


def load_context(out, relative, digest):
    return torch.load(artifact_path(out, relative, digest), map_location='cpu')


def prune_contexts(out, keep):
    # One latest.pt per run is the only local commit point. Only generated,
    # unreferenced immutable contexts are removed AFTER that commit succeeds.
    directory = Path(out) / 'contexts'
    if directory.is_dir():
        for path in directory.iterdir():
            if (re.fullmatch(r'context-[0-9a-f]{64}\.pt', path.name)
                    and str(path.relative_to(out)) != keep):
                path.unlink()
