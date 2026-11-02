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
import shutil
import errno
import re
import sys
import time
import numpy as np
import torch
from .io import ROOT, atomic_json, atomic_torch, digest_sources, require_tests
from .regularizer import OnlineGL, preserve_rng, mixers
from .checkpoint import (cpu_tree, model_snapshot, load_model, write_context, load_context,
                         prune_contexts, encode_cache, decode_cache)
from .storage import (StoragePause, compatible_identity, compatible_source, require_space,
                      tensor_bytes, artifact_path)


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
    result = cpu_tree(gvm.cache_dict)
    for key in ('null_param_id_dict', 'update_proj_dict', 'interm_tensor_dict'):
        if key in result:
            result[key] = {lookup[pid]: value for pid, value in result[key].items()}
    return result


def unpack_cache(cache, model):
    parameters = dict(model.named_parameters())
    result = {k: v for k, v in cache.items()}
    for key in ('null_param_id_dict', 'update_proj_dict', 'interm_tensor_dict'):
        if key in result:
            result[key] = {id(parameters[n]): value for n, value in result[key].items()}
    # Projectors/intermediate covariance must live on the same device as updates.
    for key in ('update_proj_dict', 'interm_tensor_dict'):
        for pid, values in result.get(key, {}).items():
            result[key][pid] = {k: v.cuda() for k, v in values.items()}
    return result


class BenchmarkComplete(Exception):
    pass


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
        self.context_file = None
        self.context_sha256 = None
        self._encoded_cache = None
        self._committed_context = None
        self._committed_factors = set()
        self.valid_paths = set()
        self.meter = None
        self.gpu_utilization = None
        self.scaler = None
        self.optimizer_updates = 0
        self.identity = dict(source_sha256=digest_sources(), config=config, arm=args.arm,
                             stage=args.stage, strength=args.strength, seed=0,
                             dataset_manifest=self.data_manifest(args.data_root),
                             pretrained_sha256=self.file_hash(args.pretrained))
        if (self.out / 'latest.pt').exists():
            self.resume = torch.load(self.out / 'latest.pt', map_location='cpu')
            if not compatible_identity(self.resume['identity'], self.identity):
                raise RuntimeError('Resume identity mismatch (source, config, arm, data, or checkpoint weights)')
            self.elapsed = self.resume['elapsed']
            if self.resume.get('context_file'):
                self.context_file = self.resume['context_file']
                self.context_sha256 = self.resume['context_sha256']
                context = load_context(self.out, self.context_file, self.context_sha256)
                self.resume.update(cache=decode_cache(context['cache']), gl=context['gl'])
            self.gl.load_state_dict(self.resume['gl'])
            self.gl.storage_dir = str(self.out / 'curvature')
            if self.resume['identity']['source_sha256'] != self.identity['source_sha256']:
                atomic_json(dict(previous=self.resume['identity']['source_sha256'],
                                 current=self.identity['source_sha256'],
                                 change='audited storage/checkpoint migration; training protocol unchanged'),
                            self.out / 'storage_migration.json')
            # Finished files from an interrupted, uncommitted statistics/context
            # pass are scratch data, never dependencies of latest.pt.
            self._committed_context = self.context_file
            self._committed_factors = {record['file'] for history in self.gl.history.values()
                                      for record in history if 'file' in record}
            self.prune_uncommitted()
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
        if self.scaler is None:
            self.scaler=torch.cuda.amp.GradScaler(enabled=self.config.get('use_amp', False))
            if self.resume and self.resume.get('scaler'):
                self.scaler.load_state_dict(self.resume['scaler'])
        self.check_next_checkpoint(gvm, model, optimizer)
        # Preserve upstream effective minibatches and one ModAdam step per batch.
        # Only split memory use; CE is sample weighted, GL applied once per update.
        total_loss, total_correct, total = 0., 0, 0
        from .telemetry import StepMeter, GPUUtilization, check_free
        if self.meter is None:
            self.gpu_utilization = GPUUtilization()
            torch.cuda.reset_peak_memory_stats()
            self.meter = StepMeter(self.out, self.args.arm,
                lambda: dict(allocated=torch.cuda.max_memory_allocated(),
                             reserved=torch.cuda.max_memory_reserved()), self.gpu_utilization)
        iterator_start = time.perf_counter()
        iterator = iter(loader)
        iterator_setup = time.perf_counter() - iterator_start
        step_in_epoch = 0
        while True:
            check_free(self.out)
            if (self.out.parent.parent / '.pause_requested').exists():
                raise StoragePause('Launcher requested pause; last committed epoch retained.')
            waiting = time.perf_counter()
            try:
                images, labels = next(iterator)
            except StopIteration:
                break
            wait = time.perf_counter() - waiting
            if step_in_epoch == 0:
                wait += iterator_setup
            torch.cuda.synchronize()
            compute_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            N = len(images)
            for start in range(0, N, self.config['microbatch']):
                x = images[start:start + self.config['microbatch']].cuda(non_blocking=True)
                y = labels[start:start + self.config['microbatch']].cuda(non_blocking=True)
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.config.get('use_amp', False)):
                    logits = model(x)
                    loss = criterion(logits / gvm.args.temperature, y) * len(x) / N
                self.scaler.scale(loss).backward()
                total_loss += float(loss.detach()) * N
                total_correct += int((logits.argmax(-1) == y).sum())
            if self.args.arm != 'MambaCL':
                with torch.autocast(device_type='cuda', enabled=False):
                    penalty = self.gl.penalty(model)
                if penalty is not None:
                    self.scaler.scale(penalty).backward()
            previous_scale=self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            update_skipped=self.scaler.get_scale()<previous_scale
            self.optimizer_updates=getattr(self,'optimizer_updates',0)+int(not update_skipped)
            total += N
            torch.cuda.synchronize()
            compute = time.perf_counter() - compute_start
            step_in_epoch += 1
            self.meter.record(gvm.cl_mngr.current_taskid, epoch, step_in_epoch, N, wait, compute, loss=total_loss/total, train_acc=100*total_correct/total)
            if getattr(self.args, 'benchmark_steps', 0) and self.optimizer_updates == self.args.benchmark_steps:
                result = self.meter.summary()
                result.update(optimizer_attempts=self.meter.step,optimizer_steps=self.optimizer_updates,
                              amp_skipped_updates=self.meter.step-self.optimizer_updates)
                fraction = result['dataloader_wait_fraction']
                result.update(status='PASS' if result['images_per_second'] >= 10 else 'SLOW',
                    minimum_images_per_second=10, identity=self.identity,
                    vram_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    vram_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                    gpu_count=1, dataloader_workers=self.config['workers'],
                    host_cpu_count=os.cpu_count(), effective_batch=self.config['batch_size'],
                    microbatch=self.config['microbatch'], data_root=str(Path(self.args.data_root).resolve()),
                    pretrained=str(Path(self.args.pretrained).resolve()),
                    timing='20 real optimizer steps; includes loader startup/wait; excludes setup/checkpoint/progress I/O; no warmup excluded',
                    loader_diagnosis='Exposed loader wait >=20%: loader/CPU bottleneck suspected' if fraction >= .2
                        else 'Exposed loader wait <20%; compute dominates this single-process probe',
                    limitation='One process with2 workers; two-process CPU contention must be assessed from live telemetry.')
                atomic_json(result, self.out / 'benchmark.json')
                print('BENCHMARK_RESULT ' + json.dumps(result), flush=True)
                raise BenchmarkComplete()
        return f'loss={total_loss / total:.4f} acc_top1={total_correct / total:.2%}'

    def skip_epoch(self, gvm, epoch):
        return bool(self.resume and not self.resume['task_complete'] and gvm.cl_mngr.current_taskid == self.resume['task'] and epoch <= self.resume['epoch'])

    def restore_optimizer(self, gvm, model, optimizer, scheduler):
        if self.resume and not self.resume['task_complete'] and not self.restored:
            optimizer.load_state_dict(self.resume['optimizer'])
            scheduler.load_state_dict(self.resume['scheduler'])
            if self.scaler is None:
                self.scaler=torch.cuda.amp.GradScaler(enabled=self.config.get('use_amp',False))
            if self.resume.get('scaler'):
                self.scaler.load_state_dict(self.resume['scaler'])
            restore_rng(self.resume['rng'])
            self.optimizer_updates=self.resume.get('optimizer_updates',0)
            self.restored = True
            self.resume.pop('optimizer', None)
            self.resume.pop('scheduler', None)

    def check_next_checkpoint(self, gvm, model, optimizer):
        planned = dict(model=model_snapshot(model, gvm.args.training_string, cpu=False),
                       optimizer=optimizer.state_dict(), params=gvm.param_dict)
        incoming = tensor_bytes(planned)
        if not optimizer.state:
            incoming += 2 * sum(p.numel()*p.element_size() for group in optimizer.param_groups for p in group['params'])
        if self.context_file is None:
            if self._encoded_cache is None:
                self._encoded_cache = encode_cache(pack_cache(gvm, model))
            incoming += tensor_bytes(dict(cache=self._encoded_cache, gl=self.gl.state_dict()))
        require_space(self.out / 'latest.pt', incoming)

    def check_task_storage(self, gvm, model):
        floats = 0
        for mixer in mixers(model).values():
            D, n, r = mixer.intermediate_size, mixer.ssm_state_size, mixer.time_step_rank
            H = mixer.out_proj.out_features
            floats += 3*D*D + 2*n*n + D*n*n + D*r*r + D + r*r + 3 + H*H
        trained = tensor_bytes(model_snapshot(model, gvm.args.training_string, cpu=False))
        if self._encoded_cache is None:
            self._encoded_cache = encode_cache(pack_cache(gvm, model))
        context = tensor_bytes(dict(cache=self._encoded_cache, gl=self.gl.state_dict()))
        # New precision, refreshed anchor/context, and a completed-task adapter.
        require_space(self.out / 'task-end-statistics.pt', 4*floats + context + 2*trained)

    def prune_uncommitted(self):
        prune_contexts(self.out, self._committed_context)
        for path in (self.out / 'curvature').glob('*.pt'):
            if re.fullmatch(r'[0-9a-f]{16}_\d+\.pt', path.name) and path.name not in self._committed_factors:
                path.unlink()

    def save(self, gvm, task, epoch, model, optimizer=None, scheduler=None, complete=False):
        try:
            return self._save(gvm,task,epoch,model,optimizer,scheduler,complete)
        except (StoragePause,OSError) as error:
            cause=error.__cause__ if isinstance(error,StoragePause) else error
            reserve=int(os.environ.get('GL_MIN_FREE_BYTES',str(2*1024**3)))
            capacity_errno=getattr(cause,'errno',None) in (errno.ENOSPC,getattr(errno,'EDQUOT',122))
            # Non-capacity I/O failure: retain the previous checkpoint, continue
            # training and retry the save at the next normal epoch boundary.
            if (cause is not None and not capacity_errno
                    and shutil.disk_usage(self.out).free >= reserve):
                print('CHECKPOINT I/O WARNING: save skipped; previous checkpoint retained; '
                      'training continues and next epoch retries. '+str(error),flush=True)
                return False
            raise

    def _save(self, gvm, task, epoch, model, optimizer=None, scheduler=None, complete=False):
        if self.context_file is None:
            if self._encoded_cache is None:
                self._encoded_cache = encode_cache(pack_cache(gvm, model))
            self.context_file, self.context_sha256 = write_context(
                self.out, dict(cache=self._encoded_cache, gl=self.gl.state_dict()))
            self._encoded_cache = None
        state = dict(identity=self.identity, task=task, epoch=epoch, task_complete=complete,
                     model=model_snapshot(model, gvm.args.training_string),
                     model_format='trainable_plus_buffers_v1',
                     context_file=self.context_file, context_sha256=self.context_sha256,
                     params=gvm.param_dict, labels=gvm.label_map_g2l,
                     accuracy=gvm.acc_mat_dict, rng=rng_state(),
                     elapsed=self.elapsed + time.monotonic() - self.started,
                     optimizer_updates=self.optimizer_updates,
                     scaler=self.scaler.state_dict() if self.scaler is not None else None,
                     optimizer=optimizer.state_dict() if optimizer is not None else None,
                     scheduler=scheduler.state_dict() if scheduler is not None else None)
        atomic_torch(state, self.out / 'latest.pt')
        self._committed_context = self.context_file
        self._committed_factors = {record['file'] for history in self.gl.history.values()
                                  for record in history if 'file' in record}
        prune_contexts(self.out, self.context_file)
        dependencies = [dict(path=str(self.out.relative_to(self.out.parent.parent) / self.context_file), sha256=self.context_sha256)]
        for history in self.gl.history.values():
            for record in history:
                if 'file' in record:
                    dependencies.append(dict(path=str(self.out.relative_to(self.out.parent.parent) / 'curvature' / record['file']),
                                             sha256=record.get('sha256')))
        atomic_json(dict(required_artifacts=dependencies,
                         instruction='Keep earlier Output shards containing these immutable artifacts attached on resume'),
                    self.out / 'resume_dependencies.json')
        atomic_json(dict(task=task + 1, epoch=epoch, task_complete=complete,
                         elapsed_seconds=state['elapsed'], statistics_floats=self.gl.floats()), self.out / 'progress.json')

    def epoch_end(self, gvm, task, epoch, model, optimizer, scheduler):
        if getattr(self.args, 'benchmark_steps', 0):
            return
        if self.resume and not self.resume['task_complete'] and task == self.resume['task'] and epoch <= self.resume['epoch']:
            return
        self.save(gvm, task, epoch, model, optimizer, scheduler)
        self.deadline()

    def deadline(self):
        if (time.time() >= float(os.environ.get('GL_DEADLINE', 'inf'))
                or (self.out.parent.parent / '.pause_requested').exists()):
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
        if getattr(self.args, 'benchmark_steps', 0) and torch.cuda.device_count() != 1:
            raise RuntimeError('The20-step benchmark must expose exactly one CUDA GPU')
        official = load_official(self)
        argv = sys.argv
        cfg = self.config
        sys.argv = ['train_eval.py', '-d', cfg['dataset'], '--data_root', self.args.data_root,
                    '--pretrained_path', self.args.pretrained, '--seed', '0', '--null_eta', str(cfg['null_eta']),
                    '--model',cfg.get('model_name','defocus_mamba_large'), '--epochs', str(cfg['epochs']), '--decay_milestones', *map(str, cfg['decay_milestones']),
                    '--batch_size', str(cfg['batch_size']), '--workers', str(cfg['workers']),
                    '--eval_workers', str(cfg['eval_workers']), '--eval_batch_size', str(cfg['eval_batch_size']),
                    '--expand_times',str(cfg['expand_times']), '--use_amp', str(cfg['use_amp']), '--persistent_workers', str(cfg['persistent_workers']),
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
        model_config=official.get_config(str(ROOT / cfg.get('model_config','upstream/mamba-cl/utils/defocus_mamba_large_22k.yaml')))
        model_config.defrost()
        model_config.TRAIN.USE_CHECKPOINT=cfg['use_checkpoint']
        model_config.freeze()
        model = official.build_model(model_config)
        atomic_json(dict(config=cfg, args=vars(args), gradient_checkpointing=cfg['use_checkpoint'],
                         checkpointed_blocks=sum(bool(getattr(m,'use_checkpoint',False)) for m in model.modules())), self.out/'effective_config.json')
        checkpoint = torch.load(args.pretrained_path, map_location='cpu')
        weights=checkpoint['model']
        result = model.load_state_dict(weights, strict=False)
        if cfg.get('backbone')=='base':
            # Original base classification checkpoint includes an unused auxiliary
            # head. CL uses the same single-head policy as large01.
            missing=[name for name in result.missing_keys]
            unexpected=[name for name in result.unexpected_keys if not name.startswith('head2.')]
            if missing or unexpected:
                raise RuntimeError('Base checkpoint does not match classification backbone: '+str(dict(missing=missing,unexpected=unexpected)))
        atomic_json(dict(missing=result.missing_keys, unexpected=result.unexpected_keys), self.out / 'pretrained_keys.json')
        setattr(model, 'pretrained_cfg', dict(pretrained_path=args.pretrained_path))
        model.reload_params()
        gvm.cache_dict.update(pretrained_cfg=deepcopy(model.pretrained_cfg), training_string=args.training_string)
        model.cuda()
        if self.resume:
            outdim = self.resume['model']['head.weight'].shape[0]
            model.head = torch.nn.Linear(model.head.in_features, outdim, model.head.bias is not None).cuda()
            load_model(model, self.resume['model'], args.training_string, self.resume.get('model_format'))
            gvm.cache_dict = unpack_cache(self.resume['cache'], model)
            gvm.param_dict, gvm.label_map_g2l = self.resume['params'], self.resume['labels']
            gvm.acc_mat_dict = self.resume['accuracy']
            # A completed-task checkpoint must restore RNG BEFORE initializing
            # the next task's head/loader, too (not only in optimizer restore).
            restore_rng(self.resume['rng'])
            self.resume.pop('model', None)
            self.resume.pop('cache', None)
            self.resume.pop('gl', None)
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
            # The official end-task projector/cache and GL anchor change here;
            # persist a new immutable context before committing task completion.
            self.context_file = None
            self._encoded_cache = None
            if self.args.arm != 'MambaCL' and self.args.strength != 0:
                self.check_task_storage(gvm, model)
                from .statistics import collect
                dataset = official.define_dataset(gvm, classes, training=True, transform_type=args.transform_type,
                                                   target_map_to_local=True, use_eval_transform=True, expand_times=1)
                # Deterministic random subset over ALL classes, not first-class prefix.
                with preserve_rng(), torch.autocast(device_type='cuda',enabled=False):
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
                       preliminary=True, backbone=cfg.get('backbone','large'),
                       paper_AA=81.67 if cfg.get('backbone','large')=='large' else None,
                       paper_forgetting=4.07 if cfg.get('backbone','large')=='large' else None)
        atomic_json(summary, self.out / 'summary.json')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', required=True, choices=['MambaCL', 'GL', 'MambaCL+GL'])
    p.add_argument('--stage', required=True, choices=['tune', 'main'])
    p.add_argument('--strength', type=float, default=1.)
    p.add_argument('--selection')
    p.add_argument('--benchmark-steps', type=int, default=0)
    p.add_argument('--microbatch',type=int)
    p.add_argument('--checkpointing',type=int,choices=[0,1])
    p.add_argument('--data-root', required=True)
    p.add_argument('--pretrained', required=True)
    p.add_argument('--out', required=True)
    a = p.parse_args()
    if a.benchmark_steps and (a.benchmark_steps != 20 or a.arm != 'MambaCL' or a.stage != 'tune'):
        p.error('Benchmark requires exactly20 steps, arm MambaCL, stage tune')
    if a.benchmark_steps and (Path(a.out)/'latest.pt').exists():
        p.error('Benchmark output must be isolated from training checkpoints')
    require_tests()
    from .configuration import load_config
    cfg = load_config(a.stage, a.microbatch, bool(a.checkpointing) if a.checkpointing is not None else None)
    if a.stage == 'main' and a.arm != 'MambaCL':
        if not a.selection:
            p.error('Main GL arms require --selection from validation tuning')
        selection = json.loads(Path(a.selection).read_text())
        if not compatible_source(selection['source_sha256']):
            raise RuntimeError('Selection belongs to different source')
        a.strength = selection['selected'][a.arm]
    runtime = Runtime(a, cfg)
    if a.stage == 'main' and a.arm != 'MambaCL':
        for key in ('dataset_manifest', 'pretrained_sha256'):
            if selection[key] != runtime.identity[key]:
                raise RuntimeError('Selection mismatch: ' + key)
    try:
        runtime.run()
    except BenchmarkComplete:
        return
    except StoragePause:
        try:
            runtime.prune_uncommitted()
        except OSError as error:
            print('Could not remove uncommitted scratch artifacts:', str(error), flush=True)
        raise
    finally:
        if runtime.gpu_utilization is not None:
            runtime.gpu_utilization.close()

if __name__ == '__main__':
    try:
        main()
    except StoragePause as error:
        print('STORAGE PAUSED:', str(error), flush=True)
        raise SystemExit(75)
