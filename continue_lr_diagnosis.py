"""Controlled fresh full-data trial at lr=1e-4, otherwise existing model/data settings."""
import json,random,time,argparse
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,loss_for,evaluate,precision,save

parser=argparse.ArgumentParser(); parser.add_argument('--steps',type=int,default=600); args=parser.parse_args()
torch.set_num_threads(4); torch.manual_seed(42)
torch.backends.cuda.matmul.allow_tf32=True
rng=random.Random(42);data=Data(Path('data/ctc-full'))
train=data.rows['train']+data.rows['synthetic']; valid=data.rows['valid'].copy()
random.Random(123).shuffle(valid); ids=[i for i,n in valid[:512]]
model=CTCModel(len(data.vocab)).cuda();opt=torch.optim.Adam(model.parameters(),lr=.0001)
out=Path('runs/ctc-full-lr-diagnosis');out.mkdir(exist_ok=True)
config=dict(lr=.0001,steps=300,seed=42,effective_batch=256,microbatch=8,source='fresh initialization',validation_ids=ids)
state=torch.load(out/'diagnostic.pt',map_location='cuda',weights_only=False)
model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
rng.setstate(state['random_state']);torch.set_rng_state(state['torch_rng'].cpu())
torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
config=state['config'];config['steps']=args.steps
(out/'continuation_config.json').write_text(json.dumps(config,indent=2))
start=time.monotonic()
for step in range(state['step']+1,args.steps+1):
 model.train();opt.zero_grad(set_to_none=True)
 selected=rng.choices(train,k=256);selected.sort(key=lambda r:r[1]);losses=[]
 for off in range(0,256,8):
  x,y,il,ol=collate([data.load(i) for i,n in selected[off:off+8]],'cuda')
  with precision('cuda'): logits=model(x,il)
  loss=loss_for(logits,y,il,ol)
  if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
  (loss/32).backward();losses.append(loss.item())
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
 if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
 opt.step()
 if step%10==0: print(json.dumps(dict(step=step,loss=sum(losses)/32,seconds=time.monotonic()-start)),flush=True)
 if step%100==0 or step==args.steps:
  v=evaluate(model,data,ids,'cuda',8)
  (out/f'validation_{step:06d}.json').write_text(json.dumps(v,indent=2))
  print(json.dumps(dict(step=step,validation={k:v for k,v in v.items() if k!='examples'})),flush=True)
  save(out/'diagnostic.pt',model,opt,step,config,rng)
