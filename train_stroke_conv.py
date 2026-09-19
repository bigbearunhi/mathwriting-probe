"""Fresh 5k pilot: tolerance .02 and stroke-local convolution; no old weights loaded."""
import json,random,time,shutil,os
from pathlib import Path
import torch
from stroke_conv import StrokeConvCTC
from ctc_train import Data,collate,loss_for,evaluate,precision,save

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
out=Path('runs/ctc-bezier002-strokeconv-5k');out.mkdir(exist_ok=True)
if (out/'config.json').exists():raise RuntimeError('Run already exists; do not overwrite')
rng=random.Random(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
data=Data(Path('data/ctc-tolerance-002'));train=data.rows['train']+data.rows['synthetic']
import sqlite3
old=sqlite3.connect('file:data/ctc-full/features.sqlite?mode=ro',uri=True)
old_ids=json.loads(Path('runs/ctc-75k/config.json').read_text())['validation_ids']
names=[old.execute('SELECT id FROM samples WHERE rowid=?',(i,)).fetchone()[0] for i in old_ids]
eligible={i for i,n in data.rows['valid']};ids=[];excluded=[]
for name in names:
 row=data.db.execute("SELECT rowid FROM samples WHERE split='valid' AND id=?",(name,)).fetchone()
 if row and row[0] in eligible:ids.append(row[0])
 else:excluded.append(name)
if not ids:raise RuntimeError('No eligible validation samples')
model=StrokeConvCTC(len(data.vocab)).cuda()
opt=torch.optim.Adam(model.parameters(),lr=.0001)
config=dict(lr=.0001,steps=5000,microbatch=16,effective_batch=256,eval_every=500,
 seed=42,cache='data/ctc-tolerance-002',tolerance=.02,repeat=2,max_frames=512,
 architecture='StrokeConvCTC',dim=512,layers=11,heads=8,ff=2048,dropout=.15,
 conv_channels=[10,64,512],conv_kernel=3,conv_stride=1,conv_activation='SiLU',
 validation_ids=ids,validation_sample_ids=names,validation_excluded=excluded,
 valid_samples=len(ids),data_exclusions=data.excluded,source='random initialization',
 best_criterion='minimum token_error_rate, higher exact_match breaks ties')
(out/'config.json').write_text(json.dumps(config,indent=2))
def status(stage,step,**kw):
 p=out/'status.partial';p.write_text(json.dumps(dict(stage=stage,step=step,target=5000,updated_unix=time.time(),**kw),indent=2));os.replace(p,out/'status.json')
best=(-float('inf'),0.)
status('training',0)
start=time.monotonic()
for step in range(1,5001):
 model.train();opt.zero_grad(set_to_none=True)
 selected=sorted(rng.choices(train,k=256),key=lambda r:r[1]);losses=[]
 for off in range(0,256,16):
  x,y,il,ol=collate([data.load(i) for i,n in selected[off:off+16]],'cuda')
  with precision('cuda'):logits=model(x,il)
  loss=loss_for(logits,y,il,ol)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  (loss/16).backward();losses.append(loss.item())
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
 opt.step()
 if step%10==0:
  status('training',step,loss=sum(losses)/16,elapsed_seconds=time.monotonic()-start)
  print(json.dumps(dict(step=step,loss=sum(losses)/16,elapsed_seconds=time.monotonic()-start)),flush=True)
 if step%500==0:
  v=evaluate(model,data,ids,'cuda',16)
  (out/f'validation_{step:06d}.json').write_text(json.dumps(v,indent=2))
  save(out/'latest.pt',model,opt,step,config,rng)
  shutil.copy2(out/'latest.pt',out/f'checkpoint_{step:06d}.pt')
  score=(-v['token_error_rate'],v['exact_match'])
  if score>best:
   shutil.copy2(out/'latest.pt',out/'best.partial');os.replace(out/'best.partial',out/'best.pt');best=score
   (out/'best.json').write_text(json.dumps(dict(step=step,exact_match=score[1],token_error_rate=-score[0])))
  summary={k:x for k,x in v.items() if k!='examples'}
  print(json.dumps(dict(step=step,validation=summary)),flush=True)
else:status('completed',5000,validation=summary)
