"""Download official archive in short, verified HTTP byte ranges; resumable."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import time
import requests

URL='https://storage.googleapis.com/download/storage/v1/b/mathwriting_data/o/mathwriting-2024.tgz?alt=media'
SIZE=3096141721
MD5='f2d59c44a545347a5f67ac70fef7a13d'
CHUNK=8*1024*1024


def main():
    base=Path(__file__).resolve().parent/'data'
    base.mkdir(exist_ok=True)
    path=base/'mathwriting-2024-ranged.partial'
    manifest=base/'download_ranges.json'
    done=set(json.loads(manifest.read_text())['completed']) if path.exists() and manifest.exists() else set()
    fd=os.open(path,os.O_CREAT|os.O_RDWR,0o644)
    os.ftruncate(fd,SIZE)
    def fetch(index):
        start=index*CHUNK
        end=min(SIZE,start+CHUNK)-1
        for attempt in range(12):
            try:
                with requests.get(URL,headers={'Range':f'bytes={start}-{end}'},timeout=(15,30)) as r:
                    r.raise_for_status()
                    if r.status_code!=206 or r.headers.get('Content-Range')!=f'bytes {start}-{end}/{SIZE}':
                        raise ValueError('Server did not honor byte range')
                    content=r.content
                    if len(content)!=end-start+1:
                        raise ValueError('Incomplete range')
                    written=0
                    while written<len(content):
                        written+=os.pwrite(fd,content[written:],start+written)
                return index
            except Exception:
                if attempt==11:
                    raise
                time.sleep(min(2**attempt,10))
    count=(SIZE+CHUNK-1)//CHUNK
    start_time=time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures=[pool.submit(fetch,i) for i in range(count) if i not in done]
            for future in as_completed(futures):
                done.add(future.result())
                # Only record a range after its bytes have been persisted.
                os.fdatasync(fd)
                tmp=manifest.with_suffix('.partial')
                tmp.write_text(json.dumps(dict(completed=sorted(done),total=count)))
                os.replace(tmp,manifest)
                if len(done)%16==0 or len(done)==count:
                    print(json.dumps(dict(ranges=len(done),total=count,elapsed=time.monotonic()-start_time)),flush=True)
    finally:
        os.close(fd)
    digest=hashlib.md5()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(CHUNK),b''):
            digest.update(chunk)
    if digest.hexdigest()!=MD5:
        raise RuntimeError('Official archive checksum mismatch')
    target=base/'mathwriting-2024.tgz'
    os.replace(path,target)
    control=base/'mathwriting-2024.tgz.aria2'
    if control.exists():
        control.rename(base/'superseded-download.aria2')
    print(json.dumps(dict(verified=True,md5=MD5,bytes=SIZE)),flush=True)


if __name__=='__main__':
    main()
