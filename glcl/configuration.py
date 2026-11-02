"""Shared AMP/resize protocol, stage schedules and measured execution profile."""
import hashlib
import json
import os
from .io import ROOT


def base_config_hash():
    return hashlib.sha256((ROOT/'configs/phase1.json').read_bytes()).hexdigest()


def load_config(stage, microbatch=None, use_checkpoint=None):
    cfg=json.loads((ROOT/'configs/phase1.json').read_text())
    if stage=='tune':
        cfg['epochs']=cfg['tune_epochs']
        cfg['decay_milestones']=cfg['tune_decay_milestones']
    profile=os.environ.get('GL_EXECUTION_PROFILE')
    if profile:
        data=json.loads(open(profile).read())
        if data['base_config_sha256']!=base_config_hash():
            raise RuntimeError('Execution profile belongs to a different protocol')
        cfg.update(microbatch=data['microbatch'],use_checkpoint=data['use_checkpoint'])
    if microbatch is not None:cfg['microbatch']=microbatch
    if use_checkpoint is not None:cfg['use_checkpoint']=use_checkpoint
    return cfg
