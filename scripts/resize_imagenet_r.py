"""Create one shared short-side256 JPEG cache, retaining exact official filenames."""
import argparse
import hashlib
import json
from pathlib import Path
from PIL import Image


def prepare(source,destination):
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Resize destination must differ from source')
    manifest=[]
    for part in ('train','val'):
        for file in sorted((source/part).rglob('*')):
            if file.is_file():manifest.append((str(file.relative_to(source)),file.stat().st_size))
    fingerprint=hashlib.sha256(json.dumps(manifest).encode()).hexdigest()
    marker=destination/'resize_manifest.json'
    policy=dict(source_manifest=fingerprint,short_side=256,interpolation='PIL bicubic',format='JPEG',quality=95)
    reuse=marker.exists() and json.loads(marker.read_text()).get('policy')==policy
    if marker.exists() and not reuse:raise RuntimeError('Resize cache protocol mismatch; use a fresh destination')
    for index,(relative,_) in enumerate(manifest,1):
        target=destination/relative
        if reuse and target.is_file():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(source/relative) as original:
            image=original.convert('RGB');w,h=image.size
            size=(256,max(256,int(h*256/w))) if w<=h else (max(256,int(w*256/h)),256)
            image=image.resize(size,Image.Resampling.BICUBIC)
            temporary=target.with_suffix(target.suffix+'.tmp')
            try:image.save(temporary,format='JPEG',quality=95);temporary.replace(target)
            finally:temporary.unlink(missing_ok=True)
        if index%1000==0:print(f'Resize256 {index}/{len(manifest)}',flush=True)
    marker.write_text(json.dumps(dict(policy=policy,files=len(manifest)),indent=2)+'\n')
    print('Shared resized dataset:',destination,flush=True)
    return destination

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True)
    a=p.parse_args();prepare(a.source,a.destination)
