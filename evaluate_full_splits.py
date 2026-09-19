"""Frozen best checkpoint evaluation; all valid/test records, fixed training vocabulary."""
import json,shutil,sqlite3,tarfile,time,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import torch
from ctc_data import convert
from ctc_train import Data,CTCModel,collate,precision,collapse,edit_distance
from probe import tokenize

def main():
 torch.set_num_threads(2)
 out=Path('runs/full-split-evaluation');out.mkdir(exist_ok=True)
 frozen=out/'frozen.pt'
 if not frozen.exists():shutil.copy2('runs/ctc-50k/best.pt',frozen)
 state=torch.load(frozen,map_location='cpu',weights_only=False)
 vocab=json.loads(Path('data/ctc-full/vocab.json').read_text())
 cache=Path('data/ctc-test-eval');cache.mkdir(exist_ok=True)
 if not (cache/'features.sqlite').exists():
  db=sqlite3.connect(cache/'features.partial');db.execute('CREATE TABLE samples (id TEXT, split TEXT, label TEXT, tokens TEXT, length INTEGER, min_frames INTEGER, features BLOB)')
  errors=[];count=0;group=[]
  with ProcessPoolExecutor(max_workers=4) as pool:
   def consume(group):
    nonlocal count
    for row,error in pool.map(convert,group,chunksize=16):
     if error:errors.append(error)
     else:db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',row);count+=1
    db.commit();print(json.dumps(dict(stage='prepare_test',count=count,errors=len(errors))),flush=True)
   with tarfile.open('data/mathwriting-2024.tgz','r|gz') as tar:
    for e in tar:
     if e.isfile() and e.name.endswith('.inkml') and Path(e.name).parent.name=='test':
      group.append((e.name,tar.extractfile(e).read()))
      if len(group)==512:consume(group);group=[]
    if group:consume(group)
  db.close();(cache/'features.partial').rename(cache/'features.sqlite')
  (cache/'preparation.json').write_text(json.dumps(dict(samples=count,errors=errors),indent=2))
  if errors:raise RuntimeError('Test preprocessing errors: inspect report')
 (cache/'vocab.json').write_text(json.dumps(vocab))
 model=CTCModel(len(vocab)).cuda();model.load_state_dict(state['model']);model.eval()
 report=dict(checkpoint_step=state['step'],checkpoint_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),selection='best on 2048 validation samples before test access',decoding='greedy CTC',repeat=2,length_filter=False,splits={})
 for split,folder in [('valid',Path('data/ctc-full')),('test',cache)]:
  data=Data(folder,max_frames=1000000)
  ids=data.db.execute('SELECT rowid,length FROM samples WHERE split=? ORDER BY length',(split,)).fetchall()
  n=correct=errors=tokens=unknown=overlong=0;maxframes=0;start=time.monotonic()
  with (out/f'{split}-predictions.jsonl').open('w') as stream,torch.no_grad():
   for off in range(0,len(ids),8):
    samples=[data.load(i) for i,_ in ids[off:off+8]];x,y,il,ol=collate(samples,'cuda')
    with precision('cuda'):pred=model(x,il).argmax(-1).cpu()
    for j,sample in enumerate(samples):
     truth=tokenize(sample[3]);hyp=[vocab[k] for k in collapse(pred[j,:il[j]].tolist())]
     distance=edit_distance(truth,hyp);n+=1;correct+=truth==hyp;errors+=distance;tokens+=len(truth);unknown+=sum(t not in data.lookup for t in truth);overlong+=int(il[j]>512);maxframes=max(maxframes,int(il[j]))
     stream.write(json.dumps(dict(id=sample[2],truth=sample[3],prediction=''.join(hyp),errors=distance),ensure_ascii=False)+'\n')
    if off%1024==0:print(json.dumps(dict(stage='evaluate',split=split,processed=n,total=len(ids))),flush=True)
  result=dict(samples=n,correct=correct,exact_match=correct/n,token_errors=errors,reference_tokens=tokens,token_error_rate=errors/tokens,unknown_reference_tokens=unknown,inputs_over_512=overlong,max_frames=maxframes,seconds=time.monotonic()-start)
  report['splits'][split]=result;(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(dict(split=split,**result)),flush=True)
 print('COMPLETE',flush=True)
if __name__=='__main__':main()
