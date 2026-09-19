"""Durable full-data preparation + bounded CTC training trial.

No scheduler: this is an ordinary local process. Stage/status files survive UI
disconnects. It never starts training from an incomplete or corrupted archive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--download-pid',type=int)
    args=p.parse_args()
    os.chdir(ROOT)
    folder=ROOT/'runs/ctc-full'
    folder.mkdir(parents=True,exist_ok=True)
    def status(stage,**details):
        value=dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**details)
        tmp=folder/'pipeline.partial'
        tmp.write_text(json.dumps(value,indent=2))
        os.replace(tmp,folder/'pipeline.json')
        print(json.dumps(value),flush=True)
    def run(command,log):
        with (folder/log).open('a',buffering=1) as f:
            subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,
                           env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
    try:
        if args.download_pid:
            status('waiting_for_official_download',download_pid=args.download_pid)
            while Path(f'/proc/{args.download_pid}').exists():
                time.sleep(5)
        archive=ROOT/'data/mathwriting-2024.tgz'
        if archive.with_suffix('.tgz.aria2').exists():
            raise RuntimeError('Download incomplete; aria2 control file still exists')
        status('verifying_archive')
        digest=hashlib.md5()
        with archive.open('rb') as f:
            for chunk in iter(lambda:f.read(8*1024*1024),b''):
                digest.update(chunk)
        expected='f2d59c44a545347a5f67ac70fef7a13d'
        if archive.stat().st_size!=3096141721 or digest.hexdigest()!=expected:
            raise RuntimeError('Archive size/MD5 does not match official GCS metadata')
        (folder/'archive_verified.json').write_text(json.dumps(dict(bytes=archive.stat().st_size,
            md5=digest.hexdigest(),official_url='https://storage.googleapis.com/mathwriting_data/mathwriting-2024.tgz'),indent=2))
        cache=ROOT/'data/ctc-full'
        if not (cache/'preparation.json').exists():
            status('preparing_full_dataset')
            run([sys.executable,'-u','ctc_data.py','--archive',str(archive),'--out',str(cache),'--workers','8'],'preparation.log')
        prepared=json.loads((cache/'preparation.json').read_text())
        if prepared['limit_per_split']!=0:
            raise RuntimeError('Full run requires an unrestricted cache')
        status('training',target_steps=args.steps,cache_counts=prepared['counts'])
        command=[sys.executable,'-u','ctc_train.py','--cache',str(cache),
            '--out',str(folder),'--steps',str(args.steps),'--microbatch','8',
            '--effective-batch','256','--max-frames','512','--valid-samples','512','--eval-every','100']
        if (folder/'latest.pt').exists():
            command.append('--resume')
        run(command,'training.log')
        status('completed',training_status=json.loads((folder/'status.json').read_text()))
    except Exception as e:
        status('failed',error=str(e))
        raise


if __name__=='__main__':
    main()
