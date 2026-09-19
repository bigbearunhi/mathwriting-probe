"""Fixed training-only samples: compare learning rates without changing production training."""
import json, random
from pathlib import Path
import torch
from ctc_train import Data, CTCModel, collate, loss_for, evaluate, precision

torch.set_num_threads(4)
data=Data(Path('data/ctc-full'))
rows=[(i,n) for i,n in data.rows['train'] if 20<=n<=100]
ids=[i for i,n in random.Random(567).sample(rows,8)]
samples=[data.load(i) for i in ids]
x,y,ilen,olen=collate(samples,'cuda')
out=Path('runs/ctc-diagnosis'); out.mkdir(exist_ok=True)
print(json.dumps({'ids':ids,'labels':[s[3] for s in samples], 'lengths':ilen.tolist(),'targets':olen.tolist(),'feature_abs_max':x.abs().amax((0,1)).tolist()}),flush=True)
for lr in [.001,.0001]:
 torch.manual_seed(42)
 model=CTCModel(len(data.vocab)).cuda()
 opt=torch.optim.Adam(model.parameters(),lr=lr)
 log=[]
 for step in range(401):
  if step%50==0:
   result=evaluate(model,data,ids,'cuda',8)
   row=dict(lr=lr,step=step,**result); log.append(row)
   print(json.dumps(row),flush=True)
   (out/f'lr-{lr}.json').write_text(json.dumps(log,indent=2))
  if step==400: break
  model.train(); opt.zero_grad()
  with precision('cuda'): logits=model(x,ilen)
  loss=loss_for(logits,y,ilen,olen)
  loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
 del model,opt
 torch.cuda.empty_cache()
