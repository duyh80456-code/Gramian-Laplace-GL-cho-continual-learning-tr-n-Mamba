from pathlib import Path
import importlib.util,os,shutil,subprocess,sys,tempfile,torch
from glcl.kaggle import archive
from glcl.storage import checkpoint_position
spec=importlib.util.spec_from_file_location('prepare_storage','scripts/prepare_storage.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
with tempfile.TemporaryDirectory(prefix='glcl-bootstrap-check-') as temp:
 root=Path(temp);working=root/'working/glcl';out=working/'glcl_outputs';a=out/'tune/GL_lambda0.1';b=out/'tune/MambaCL+GL_lambda0.1';a.mkdir(parents=True);b.mkdir(parents=True)
 def save(path,epoch):torch.save(dict(task=0,epoch=epoch,task_complete=False,model={'head.weight':torch.ones(2,2)}),path)
 save(a/'latest.pt',1);save(b/'latest.pt',1);save(b/'latest.pt.tmp',2)
 (a/'latest.pt.tmp').write_bytes(b'broken');(working/'glcl_outputs.tar.tmp').write_bytes(b'partial archive')
 old=a.joinpath('latest.pt').read_bytes();valid_archive=working/'glcl_outputs.tar.gz';valid_archive.write_bytes(b'older valid archive fixture')
 env=working.parent/'glcl_env';subprocess.run([sys.executable,'-m','venv','--without-pip',str(env)],check=True)
 moved=root/'ephemeral';(working/'input_data').mkdir();(working/'input_data/marker').write_text('retain image cache')
 report=m.prepare(working,moved)
 assert (a/'latest.pt').read_bytes()==old and not (a/'latest.pt.tmp').exists()
 assert checkpoint_position(b/'latest.pt')==(0,2,0) and not (b/'latest.pt.tmp').exists()
 assert valid_archive.read_bytes()==b'older valid archive fixture'
 assert not (working/'glcl_outputs.tar.tmp').exists()
 assert (moved/'input_data/marker').read_text()=='retain image cache'
 prefix=subprocess.check_output([str(moved/'env/bin/python'),'-c','import sys; print(sys.prefix)'],text=True).strip()
 assert Path(prefix).resolve()==(moved/'env').resolve()
 print('Scratch cleanup, recovery of complete temporary checkpoint, existing checkpoint/archive preservation and venv/cache relocation: PASS')
 # Fault-inject the optional archive writer; it must not replace the last valid archive.
 import glcl.kaggle as k
 original=k.tarfile.open
 class BrokenTar:
  def __enter__(self):
   self.file=valid_archive.with_suffix('.tmp').open('wb');self.file.write(b'partial');return self
  def add(self,*args,**kwargs):raise OSError(28,'No space left on device')
  def __exit__(self,*args):self.file.close()
 k.tarfile.open=lambda *args,**kwargs:BrokenTar()
 try:
  try:archive(out,valid_archive)
  except OSError:pass
  else:raise AssertionError('Injected archive write failure did not fire')
 finally:k.tarfile.open=original
 assert valid_archive.read_bytes()==b'older valid archive fixture'
 assert not valid_archive.with_suffix('.tmp').exists()
 print('Optional archive failure cleans its partial file and preserves previous archive: PASS')
print('Bootstrap/filesystem checks only; no training process launched')
