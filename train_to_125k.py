"""Resume the 75k checkpoint; 2048 validation samples, microbatch16, batch256."""
import json,random,time,shutil,os
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,loss_for,evaluate,precision,save

torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
out=Path('runs/ctc-125k');out.mkdir(exist_ok=True)
if (out/'config.json').exists():raise RuntimeError('Run already exists; do not overwrite')
source=Path('runs/ctc-75k/latest.pt')
state=torch.load(source,map_location='cpu',weights_only=False)
assert state['step']==75000
rng=random.Random();rng.setstate(state['random_state'])
data=Data(Path('data/ctc-full'));train=data.rows['train']+data.rows['synthetic']
valid=data.rows['valid'].copy();random.Random(123).shuffle(valid);ids=[i for i,n in valid[:2048]]
model=CTCModel(len(data.vocab)).cuda();model.load_state_dict(state['model'])
opt=torch.optim.Adam(model.parameters(),lr=.0001);opt.load_state_dict(state['optimizer'])
assert all(g['lr']==.0001 for g in opt.param_groups)
torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
config=dict(state['config']);config.update(steps=125000,microbatch=16,effective_batch=256,eval_every=500,validation_ids=ids,valid_samples=2048,source=str(source),patience_steps=None,best_criterion='minimum token_error_rate, higher exact_match breaks ties')
(out/'config.json').write_text(json.dumps(config,indent=2))
def status(stage,step,**kw):
 p=out/'status.partial';p.write_text(json.dumps(dict(stage=stage,step=step,target=125000,updated_unix=time.time(),**kw),indent=2));os.replace(p,out/'status.json')
status('baseline_validation',75000)
v=evaluate(model,data,ids,'cuda',16)
(out/'validation_075000.json').write_text(json.dumps(v,indent=2))
best=(-v['token_error_rate'],v['exact_match'])
save(out/'latest.pt',model,opt,75000,config,rng);shutil.copy2(out/'latest.pt',out/'best.pt')
(out/'best.json').write_text(json.dumps(dict(step=75000,exact_match=best[1],token_error_rate=-best[0])))
previous_best=json.loads(Path('runs/ctc-75k/best.json').read_text())
previous_score=(-previous_best['token_error_rate'],previous_best['exact_match'])
if previous_score>best:
 shutil.copy2('runs/ctc-75k/best.pt',out/'best.pt');best=previous_score
 (out/'best.json').write_text(json.dumps(previous_best))
print(json.dumps(dict(step=75000,baseline={k:x for k,x in v.items() if k!='examples'})),flush=True)
del state
start=time.monotonic()
for step in range(75001,125001):
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
  score=(-v['token_error_rate'],v['exact_match'])
  if score>best:
   shutil.copy2(out/'latest.pt',out/'best.partial');os.replace(out/'best.partial',out/'best.pt');best=score
   (out/'best.json').write_text(json.dumps(dict(step=step,exact_match=score[1],token_error_rate=-score[0])))
  summary={k:x for k,x in v.items() if k!='examples'}
  print(json.dumps(dict(step=step,validation=summary)),flush=True)
else:status('completed',125000,validation=summary)
