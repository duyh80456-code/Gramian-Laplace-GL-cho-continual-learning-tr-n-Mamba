import sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import copy,importlib.util,json,os,random,shutil,tempfile,time
from types import SimpleNamespace
import numpy as np
import torch
from glcl.io import atomic_torch, atomic_json, digest_sources
from glcl.storage import StoragePause, checkpoint_position, sha256_file, restore_direct
from glcl.checkpoint import model_snapshot, load_model, encode_cache, decode_cache
from glcl.runner import Runtime, rng_state, restore_rng

class Tiny(torch.nn.Module):
 def __init__(self):
  super().__init__();self.backbone=torch.nn.Linear(6,6,dtype=torch.float64);self.head=torch.nn.Linear(6,4,dtype=torch.float64)
  self.backbone.requires_grad_(False)
 def forward(self,x):return self.head(self.backbone(x))

torch.set_num_threads(1);torch.manual_seed(47)
with tempfile.TemporaryDirectory(prefix='glcl-storage-check-') as temp:
 root=Path(temp);data=root/'data';(data/'train/a').mkdir(parents=True);(data/'val/a').mkdir(parents=True)
 (data/'train/a/x').write_bytes(b'train fixture');(data/'val/a/y').write_bytes(b'val fixture')
 model=Tiny();base=copy.deepcopy(model.state_dict());pretrained=root/'pretrained.pt';torch.save(dict(model=base),pretrained)
 cfg=json.loads(Path('configs/phase1.json').read_text())
 args=SimpleNamespace(arm='GL',stage='tune',strength=.1,out=str(root/'session/glcl_outputs/tune/GL_lambda0.1'),data_root=str(data),pretrained=str(pretrained))
 run=Runtime(args,cfg)
 gvm=SimpleNamespace(args=SimpleNamespace(training_string=['head']),cache_dict={'static_covariance':torch.eye(8,dtype=torch.float64)},param_dict={},label_map_g2l={},acc_mat_dict={'AccClassIncList':np.zeros(10)})
 optimizer=torch.optim.Adam(model.head.parameters(),lr=.01)
 scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=3)
 x=torch.randn(8,6,dtype=torch.float64);y=torch.arange(8)%4
 for _ in range(2):
  optimizer.zero_grad();loss=torch.nn.functional.cross_entropy(model(x),y);loss.backward();optimizer.step()
 factor=run.out/'curvature/aaaaaaaaaaaaaaaa_00.pt';atomic_torch({'sx':torch.eye(2,dtype=torch.float64)},factor)
 run.gl.history={'fixture':[dict(file=factor.name,sha256=sha256_file(factor),weight=1.,floats=4,A=torch.zeros(2,2,dtype=torch.float64))]}
 run.save(gvm,0,1,model,optimizer,scheduler)
 state=torch.load(run.out/'latest.pt',map_location='cpu')
 assert set(state['model'])=={'head.weight','head.bias'}
 assert 'cache' not in state and 'gl' not in state and state['context_file']
 restored=Tiny();restored.load_state_dict(base);load_model(restored,state['model'],['head'],state['model_format'])
 assert all(torch.equal(a,b) for a,b in zip(model.state_dict().values(),restored.state_dict().values()))
 resumed_optimizer=torch.optim.Adam(restored.head.parameters(),lr=.01);resumed_optimizer.load_state_dict(state['optimizer'])
 for network,opt in ((model,optimizer),(restored,resumed_optimizer)):
  opt.zero_grad();loss=torch.nn.functional.cross_entropy(network(x),y);loss.backward();opt.step()
 assert all(torch.equal(a,b) for a,b in zip(model.parameters(),restored.parameters()))
 old_context=state['context_file'];run.save(gvm,0,2,model,optimizer,scheduler)
 newer=torch.load(run.out/'latest.pt',map_location='cpu')
 assert newer['context_file']==old_context and len(list((run.out/'contexts').glob('context-*.pt')))==1
 print('Compact checkpoint restores exact model/optimizer; immutable task context is reused: PASS')
 # Simulated ENOSPC preserves the previous commit and removes the failed file.
 before=sha256_file(run.out/'latest.pt');original=torch.save
 def failed_save(value,stream):stream.write(b'partial');raise RuntimeError('PytorchStreamWriter failed writing file data/0: file write failed')
 torch.save=failed_save
 try:
  try:atomic_torch({'x':torch.ones(3)},run.out/'latest.pt')
  except StoragePause:pass
  else:raise AssertionError('Write failure was not converted to controlled pause')
 finally:torch.save=original
 assert sha256_file(run.out/'latest.pt')==before and not (run.out/'latest.pt.tmp').exists()
 prior=os.environ.get('GL_MIN_FREE_BYTES');os.environ['GL_MIN_FREE_BYTES']=str(2**62)
 try:
  try:atomic_torch({'x':torch.ones(3)},run.out/'latest.pt')
  except StoragePause:pass
  else:raise AssertionError('Disk preflight did not pause')
 finally:
  if prior is None:os.environ.pop('GL_MIN_FREE_BYTES',None)
  else:os.environ['GL_MIN_FREE_BYTES']=prior
 assert sha256_file(run.out/'latest.pt')==before
 print('Low-disk guard and injected write failure preserve latest.pt and remove .tmp: PASS')
 # Mount a prior Output and restore only the small checkpoint/metadata.
 source=root/'inputs/previous/glcl_outputs';source.parent.mkdir(parents=True);shutil.copytree(run.out.parent.parent,source)
 dest=root/'next/glcl_outputs';assert restore_direct(root/'inputs',dest)
 target=dest/'tune/GL_lambda0.1';assert not (target/'contexts').exists() and not (target/'curvature').exists()
 newargs=copy.copy(args);newargs.out=str(target);second=Runtime(newargs,cfg)
 assert torch.equal(second.resume['cache']['static_covariance'],gvm.cache_dict['static_covariance'])
 path=second.gl.factor_path(second.gl.history['fixture'][0]);assert str(path).startswith(str(source))
 print('Resume uses checked Input context/curvature blobs without copying them into Output: PASS')
 # Legacy full-state checkpoint from the exact approved pre-fix source.
 legacy_args=copy.copy(args);legacy_args.out=str(root/'legacy/glcl_outputs/tune/GL_lambda0.1')
 Path(legacy_args.out).mkdir(parents=True)
 legacy=dict(state);legacy.pop('context_file');legacy.pop('context_sha256');legacy.pop('model_format');legacy['model']=model.state_dict()
 legacy['cache']=gvm.cache_dict;legacy['gl']=run.gl.state_dict();legacy['identity']=dict(run.identity,source_sha256='ddf74bae8905a19b6d4a83ac0fba297407a8815fbfa197814ff5498923afa07a')
 torch.save(legacy,Path(legacy_args.out)/'latest.pt')
 migrated=Runtime(legacy_args,cfg);assert migrated.resume['epoch']==1 and (migrated.out/'storage_migration.json').exists()
 reloaded=Tiny();load_model(reloaded,migrated.resume['model'],['head'])
 assert all(torch.equal(a,b) for a,b in zip(model.parameters(),reloaded.parameters()))
 assert checkpoint_position(migrated.out/'latest.pt')==(0,1,0)
 print('Legacy full checkpoint and approved source migration retain epoch1 and exact weights: PASS')
 saved=rng_state();first=(random.random(),np.random.rand(),torch.rand(4));restore_rng(saved);again=(random.random(),np.random.rand(),torch.rand(4))
 assert first[:2]==again[:2] and torch.equal(first[2],again[2])
 print('RNG state roundtrip: PASS')
 # Standalone browser-renamed legacy checkpoints are discovered by identity.
 standalone=root/'standalone';standalone.mkdir()
 torch.save(legacy,standalone/'downloaded checkpoint (1).pt')
 restored_root=root/'standalone-restored/glcl_outputs'
 assert restore_direct(standalone,restored_root)
 assert checkpoint_position(restored_root/'tune/GL_lambda0.1/latest.pt')==(0,1,0)
 print('Standalone renamed legacy checkpoint detection: PASS')
 prior_budget=os.environ.get('GL_OUTPUT_BUDGET_BYTES')
 os.environ['GL_OUTPUT_BUDGET_BYTES']='1'
 try:
  try:atomic_torch({'x':torch.ones(3)},run.out/'latest.pt')
  except StoragePause:pass
  else:raise AssertionError('Output quota guard did not pause')
 finally:
  if prior_budget is None:os.environ.pop('GL_OUTPUT_BUDGET_BYTES',None)
  else:os.environ['GL_OUTPUT_BUDGET_BYTES']=prior_budget
 assert sha256_file(run.out/'latest.pt')==before
 print('Logical saved-Output quota guard preserves latest.pt: PASS')
# Exercise square packing with asymmetric ULP bits, signed zero and NaN payloads.
for dtype in (torch.float32,torch.float64):
 matrix=torch.eye(64,dtype=dtype)
 a=matrix.numpy();unsigned=np.dtype('u'+str(a.dtype.itemsize))
 bits=a.view(unsigned);bits[2,1]=1;bits[3,2]=1 << (a.dtype.itemsize*8-1)
 bits[4,3]=0x7fc00011 if dtype==torch.float32 else 0x7ff8000000000011
 packed=encode_cache({'nested':[matrix]},minimum_bytes=0)
 assert isinstance(packed['nested'][0],dict)
 unpacked=decode_cache(packed)['nested'][0]
 assert unpacked.numpy().tobytes()==a.tobytes()
print('Lossless square-cache compression preserves asymmetric bits/NaN payloads/signed zeros: PASS')
print('Storage-only CPU checks complete; no Mamba/CUDA training or full pytest run')
