"""Use the pinned official task loop through narrowly scoped insertion points.

No training runs on import. Section-six gate runs BEFORE CUDA/model/data loading.
"""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch
from .io import ROOT, atomic_json, atomic_torch, digest_sources, require_tests
from .regularizer import OnlineGL, preserve_rng


def load_official(runtime):
    path = ROOT / 'upstream/mamba-cl'
    sys.path.insert(0, str(path))
    source = (path / 'train_eval.py').read_text()
    # Keep upstream source pristine; assert each insertion point is unique.
    patches = {
        'loss: Tensor = criterion(logits / temperature, target)':
            'loss: Tensor = criterion(logits / temperature, target)\n        loss = GVM.gl_runtime.add_loss(loss, model)',
        '    torch.cuda.empty_cache()\n    for epoch':
            '    GVM.gl_runtime.restore_optimizer(GVM, model, optimizer, scheduler)\n    torch.cuda.empty_cache()\n    for epoch',
        '        scheduler.step(epoch)\n\n    cache_state':
            '        if not GVM.gl_runtime.skip_epoch(GVM, epoch):\n            scheduler.step(epoch)\n        GVM.gl_runtime.epoch_end(GVM, taskid, epoch, model, optimizer, scheduler)\n\n    cache_state',
    }
    for old, new in patches.items():
        if source.count(old) != 1:
            raise RuntimeError('Official source insertion point changed: ' + old)
        source = source.replace(old, new)
    spec = importlib.util.spec_from_loader('glcl_official', loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(path / 'train_eval.py')
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, 'exec'), module.__dict__)
    runtime.official = module
    # The upstream train epoch remains available for literal full-batch operation.
    runtime.original_epoch = module.train_one_epoch
    module.train_one_epoch = runtime.train_epoch
    return module


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


def restore_rng(s):
    random.setstate(s['python'])
    np.random.set_state(s['numpy'])
    torch.set_rng_state(s['cpu'])
    torch.cuda.set_rng_state_all(s['cuda'])


def pack_cache(gvm, model):
    lookup = {id(p): n for n, p in model.named_parameters()}
    result = deepcopy(gvm.cache_dict)
    for key in ('null_param_id_dict', 'update_proj_dict', 'interm_tensor_dict'):
        if key in result:
            result[key] = {lookup[pid]: value for pid, value in result[key].items()}
    return result


def unpack_cache(cache, model):
    parameters = dict(model.named_parameters())
    result = deepcopy(cache)
    for key in ('null_param_id_dict', 'update_proj_dict', 'interm_tensor_dict'):
        if key in result:
            result[key] = {id(parameters[n]): value for n, value in result[key].items()}
    # Projectors/intermediate covariance must live on the same device as updates.
    for key in ('update_proj_dict', 'interm_tensor_dict'):
        for pid, values in result.get(key, {}).items():
            result[key][pid] = {k: v.cuda() for k, v in values.items()}
    return result


class Runtime:
    def __init__(self, args, config):
        self.args, self.config = args, config
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.started = time.monotonic()
        self.elapsed = 0.
        self.gl = OnlineGL(args.strength, config['mu'], config['gamma'], storage_dir=self.out / 'curvature')
        self.resume = None
        self.restored = False
        self.valid_paths = set()
        self.identity = dict(source_sha256=digest_sources(), config=config, arm=args.arm,
                             stage=args.stage, strength=args.strength, seed=0,
                             dataset_manifest=self.data_manifest(args.data_root),
                             pretrained_sha256=self.file_hash(args.pretrained))
        if (self.out / 'latest.pt').exists():
            self.resume = torch.load(self.out / 'latest.pt', map_location='cpu')
            if self.resume['identity'] != self.identity:
                raise RuntimeError('Resume identity mismatch (source, config, arm, data, or checkpoint weights)')
            self.elapsed = self.resume['elapsed']
            self.gl.load_state_dict(self.resume['gl'])
            self.gl.storage_dir = str(self.out / 'curvature')
        atomic_json(self.identity, self.out / 'identity.json')

    @staticmethod
    def file_hash(path):
        digest = hashlib.sha256()
        with open(path, 'rb') as stream:
            for data in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                digest.update(data)
        return digest.hexdigest()

    @staticmethod
    def data_manifest(root):
        root = Path(root)
        manifest = []
        for part in ('train', 'val'):
            for p in sorted((root / part).rglob('*')):
                if p.is_file():
                    manifest.append((str(p.relative_to(root)), p.stat().st_size))
        if not manifest:
            raise RuntimeError('Expected official ImageNet-R train/val directory tree')
        return hashlib.sha256(json.dumps(manifest).encode()).hexdigest()

    def add_loss(self, loss, model):
        if self.args.arm == 'MambaCL':
            return loss
        penalty = self.gl.penalty(model)
        return loss if penalty is None else loss + penalty

    def train_epoch(self, gvm, epoch, loader, model, criterion, optimizer):
        if self.resume and not self.resume['task_complete'] and gvm.cl_mngr.current_taskid == self.resume['task'] and epoch <= self.resume['epoch']:
            return 'restored epoch'
        if self.config['microbatch'] >= self.config['batch_size']:
            return self.original_epoch(gvm, epoch, loader, model, criterion, optimizer)
        # Preserve upstream effective minibatches and one ModAdam step per batch.
        # Only split memory use; CE is sample weighted, GL applied once per update.
        total_loss, total_correct, total = 0., 0, 0
        for images, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            N = len(images)
            for start in range(0, N, self.config['microbatch']):
                x = images[start:start + self.config['microbatch']].cuda(non_blocking=True)
                y = labels[start:start + self.config['microbatch']].cuda(non_blocking=True)
                logits = model(x)
                loss = criterion(logits / gvm.args.temperature, y) * len(x) / N
                loss.backward()
                total_loss += float(loss.detach()) * N
                total_correct += int((logits.argmax(-1) == y).sum())
            if self.args.arm != 'MambaCL':
                penalty = self.gl.penalty(model)
                if penalty is not None:
                    penalty.backward()
            optimizer.step()
            total += N
        return f'loss={total_loss / total:.4f} acc_top1={total_correct / total:.2%}'

    def skip_epoch(self, gvm, epoch):
        return bool(self.resume and not self.resume['task_complete'] and gvm.cl_mngr.current_taskid == self.resume['task'] and epoch <= self.resume['epoch'])

    def restore_optimizer(self, gvm, model, optimizer, scheduler):
        if self.resume and not self.resume['task_complete'] and not self.restored:
            optimizer.load_state_dict(self.resume['optimizer'])
            scheduler.load_state_dict(self.resume['scheduler'])
            restore_rng(self.resume['rng'])
            self.restored = True

    def save(self, gvm, task, epoch, model, optimizer=None, scheduler=None, complete=False):
        state = dict(identity=self.identity, task=task, epoch=epoch, task_complete=complete,
                     model={n: v.detach().cpu() for n, v in model.state_dict().items()},
                     cache=pack_cache(gvm, model), params=gvm.param_dict, labels=gvm.label_map_g2l,
                     accuracy=gvm.acc_mat_dict, gl=self.gl.state_dict(), rng=rng_state(),
                     elapsed=self.elapsed + time.monotonic() - self.started,
                     optimizer=optimizer.state_dict() if optimizer is not None else None,
                     scheduler=scheduler.state_dict() if scheduler is not None else None)
        atomic_torch(state, self.out / 'latest.pt')
        atomic_json(dict(task=task + 1, epoch=epoch, task_complete=complete,
                         elapsed_seconds=state['elapsed'], statistics_floats=self.gl.floats()), self.out / 'progress.json')

    def epoch_end(self, gvm, task, epoch, model, optimizer, scheduler):
        if self.resume and not self.resume['task_complete'] and task == self.resume['task'] and epoch <= self.resume['epoch']:
            return
        self.save(gvm, task, epoch, model, optimizer, scheduler)
        self.deadline()

    def deadline(self):
        if time.time() >= float(os.environ.get('GL_DEADLINE', 'inf')):
            print('PAUSED at checkpoint; rerun identical notebook with previous Output', flush=True)
            raise SystemExit(75)

    def split_validation(self, gvm):
        # Per-class deterministic 10% split, filename order independent of host.
        train = gvm.path_data_dict['train']
        valid = deepcopy(train)
        for cls, paths in train.class_imgpath_dict.items():
            ordered = sorted(paths, key=lambda p: hashlib.sha256(('glval0:' + str(Path(p).relative_to(self.args.data_root))).encode()).digest())
            n = max(1, int(len(ordered) * self.config['validation_fraction']))
            train.class_imgpath_dict[cls] = sorted(ordered[n:])
            valid.class_imgpath_dict[cls] = sorted(ordered[:n])
        gvm.path_data_dict['eval'] = valid
        atomic_json({str(k): [str(Path(p).relative_to(self.args.data_root)) for p in v] for k, v in valid.class_imgpath_dict.items()}, self.out / 'validation_paths.json')

    def run(self):
        official = load_official(self)
        argv = sys.argv
        cfg = self.config
        sys.argv = ['train_eval.py', '-d', cfg['dataset'], '--data_root', self.args.data_root,
                    '--pretrained_path', self.args.pretrained, '--seed', '0', '--null_eta', str(cfg['null_eta']),
                    '--epochs', str(cfg['epochs']), '--decay_milestones', *map(str, cfg['decay_milestones']),
                    '--batch_size', str(cfg['batch_size']), '--workers', str(cfg['workers']),
                    '--eval_workers', str(cfg['eval_workers']), '--eval_batch_size', str(cfg['eval_batch_size']),
                    '--sample_type', 'path', '--training_string', *(cfg['baseline_training_string'] if self.args.arm == 'MambaCL' else cfg['training_string'])]
        if self.args.arm != 'GL':
            sys.argv += ['--use_null_space']
        try:
            args = official.get_args()
        finally:
            sys.argv = argv
        official.args = args
        official.seed_etc_options(0)
        # PIL decoding, same RGB tensors; accimage is absent on Kaggle.
        official.torchvision.set_image_backend('PIL')
        gvm = official.GlobalVarsManager()
        gvm.init_from_args(args)
        gvm.gl_runtime = self
        if self.args.stage == 'tune':
            self.split_validation(gvm)
        gvm.cache_dict['exp_start_time'] = time.time()
        official.GVM = gvm
        model = official.build_model(official.get_config(str(ROOT / 'upstream/mamba-cl/utils/defocus_mamba_large_22k.yaml')))
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        result = model.load_state_dict(checkpoint['model'], strict=False)
        atomic_json(dict(missing=result.missing_keys, unexpected=result.unexpected_keys), self.out / 'pretrained_keys.json')
        setattr(model, 'pretrained_cfg', dict(pretrained_path=args.pretrained_path))
        model.reload_params()
        gvm.cache_dict.update(pretrained_cfg=deepcopy(model.pretrained_cfg), training_string=args.training_string)
        model.cuda()
        if self.resume:
            outdim = self.resume['model']['head.weight'].shape[0]
            model.head = torch.nn.Linear(model.head.in_features, outdim, model.head.bias is not None).cuda()
            model.load_state_dict(self.resume['model'], strict=True)
            gvm.cache_dict = unpack_cache(self.resume['cache'], model)
            gvm.param_dict, gvm.label_map_g2l = self.resume['params'], self.resume['labels']
            gvm.acc_mat_dict = self.resume['accuracy']
        stop_task = 3 if self.args.stage == 'tune' else 10
        for task, classes in gvm.cl_mngr:
            if task >= stop_task:
                break
            if self.resume and (task < self.resume['task'] or (task == self.resume['task'] and self.resume['task_complete'])):
                continue
            # Set global used by upstream get_head_dim_arg_dict.
            official.current_task_classes = classes
            if not (self.resume and task == self.resume['task'] and not self.resume['task_complete']):
                gvm.update_label_maps(task, classes)
            model = official.train_one_task(gvm, task, classes, model)
            if self.args.arm != 'MambaCL' and self.args.strength != 0:
                from .statistics import collect
                dataset = official.define_dataset(gvm, classes, training=True, transform_type=args.transform_type,
                                                   target_map_to_local=True, use_eval_transform=True, expand_times=1)
                # Deterministic random subset over ALL classes, not first-class prefix.
                with preserve_rng():
                    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(task + 1000))[:cfg['statistics_max_samples']]
                    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, indices.tolist()), batch_size=1, shuffle=False, num_workers=cfg['eval_workers'])
                    stats = collect(model, loader, cfg['statistics_max_samples'], seed=task + 2000, diagnostics=task == 0 and self.args.stage == 'main')
                    if task == 0 and self.args.stage == 'main':
                        from .diagnostics import measure
                        measure(model, loader, stats, self.out / 'diagnostics.json')
                stats = {name: {k: v for k, v in fs.items() if not k.startswith(('ewc_', 'kfac_'))} for name, fs in stats.items()}
                self.gl.consolidate(model, stats)
            official.evaluate_tasks_sofar(gvm, task, model)
            official.task_ending_info(gvm)
            self.save(gvm, task, args.epochs, model, complete=True)
            self.resume = None
            self.deadline()
        acc = gvm.acc_mat_dict['AccClassIncList'][:stop_task]
        summary = dict(arm=self.args.arm, seed=0, stage=self.args.stage, strength=self.args.strength,
                       AA=float(acc[-1]) * 100, AIA=float(acc.mean()) * 100,
                       forgetting=float(official.misc.calc_forgetting(gvm.acc_mat_dict['AccClassIncMat'], stop_task - 1)) * 100,
                       accuracy_matrix=gvm.acc_mat_dict['AccClassIncMat'][:stop_task, :stop_task].tolist(),
                       gpu_hours=(self.elapsed + time.monotonic() - self.started) / 3600,
                       statistics_floats=self.gl.floats(), reimplemented=False,
                       preliminary=True, paper_AA=81.67, paper_forgetting=4.07)
        atomic_json(summary, self.out / 'summary.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['MambaCL', 'GL', 'MambaCL+GL'])
    p.add_argument('--stage', required=True, choices=['tune', 'main'])
    p.add_argument('--strength', type=float, default=1.)
    p.add_argument('--selection')
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    require_tests()
    cfg = json.loads((ROOT / 'configs/phase1.json').read_text())
    if a.stage == 'main' and a.arm != 'MambaCL':
        if not a.selection:
            p.error('Main GL arms require --selection from validation tuning')
        selection = json.loads(Path(a.selection).read_text())
        if selection['source_sha256'] != digest_sources():
            raise RuntimeError('Selection belongs to different source')
        a.strength = selection['selected'][a.arm]
    runtime = Runtime(a, cfg)
    if a.stage == 'main' and a.arm != 'MambaCL':
        for key in ('dataset_manifest', 'pretrained_sha256'):
            if selection[key] != runtime.identity[key]:
                raise RuntimeError('Selection mismatch: ' + key)
    runtime.run()

if __name__ == '__main__':
    main()
