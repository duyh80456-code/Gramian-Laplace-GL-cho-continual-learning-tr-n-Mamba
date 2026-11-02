"""CPU-only backbone routing/checkpoint isolation; no full model allocation."""
from pathlib import Path
import tempfile
import os
import json
from unittest.mock import patch
import torch
from glcl.configuration import load_config,base_config_hash
from glcl.io import digest_sources
from glcl.storage import restore_direct,checkpoint_metadata
from scripts.prepare_imagenet_r import find_checkpoint

with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary)
    with patch.dict(os.environ,GL_BACKBONE='large'):
        os.environ.pop('GL_EXECUTION_PROFILE',None)
        large=load_config('tune');large_hash=base_config_hash()
    with patch.dict(os.environ,GL_BACKBONE='base'):
        os.environ.pop('GL_EXECUTION_PROFILE',None)
        base=load_config('tune')
        assert base['backbone']=='base' and base['epochs']==10 and base['decay_milestones']==[5,8]
        assert base['batch_size']==large['batch_size']==200 and base['use_amp']
        assert base_config_hash()!=large_hash
        weights=root/'weights';weights.mkdir()
        (weights/'defocus_mamba_base_cls.pth').write_bytes(b'base fixture')
        (weights/'defocus_mamba_large_cls_21k.pth').write_bytes(b'large fixture')
        assert find_checkpoint(weights,base['checkpoint_name']).name=='defocus_mamba_base_cls.pth'
        for name,cfg in [('base',base),('large',large)]:
            job=root/'inputs'/name/'glcl_outputs/tune/GL_lambda0.1';job.mkdir(parents=True)
            identity=dict(source_sha256=digest_sources(),config=cfg,arm='GL',stage='tune',strength=.1)
            (job/'identity.json').write_text(json.dumps(identity))
            torch.save(dict(identity=identity,task=0,epoch=1,task_complete=False,model={'head.weight':torch.ones(2,2)}),job/'latest.pt')
            (job.parent.parent/'execution_profile.json').write_text(json.dumps(dict(backbone=name)))
        assert checkpoint_metadata(root/'inputs/base/glcl_outputs/tune/GL_lambda0.1/latest.pt')['backbone']=='base'
        output=root/'new/glcl_outputs';assert restore_direct(root/'inputs',output)
        loaded=torch.load(output/'tune/GL_lambda0.1/latest.pt',map_location='cpu')
        assert loaded['identity']['config']['backbone']=='base'
        assert json.loads((output/'execution_profile.json').read_text())['backbone']=='base'
print('Base config/schedule, distinct profile hash, checkpoint filename lookup and cross-backbone restore isolation: PASS')
print('No full-model build, CUDA import, T4 fit or throughput measurement performed')
