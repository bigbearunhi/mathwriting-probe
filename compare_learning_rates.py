"""Paired continuation from one frozen checkpoint, with identical data/RNG states."""
import json,random,time,subprocess,hashlib
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,precision,loss_for,evaluate,save

out=Path('runs/lr-comparison')
def status(**r):
 p=out/'status.partial';p.write_text(json.dumps(r,indent=2));p.replace(out/'status.json')
def main():
 torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
 state=torch.load(out/'source.pt',map_location='cpu',weights_only=False)
 data=Data(Path('data/ctc-full'));train=data.rows['train']+data.rows['synthetic'];ids=state['config']['validation_ids']
 results={};base=state['step'];start=time.monotonic()
 (out/'design.json').write_text(json.dumps(dict(source_step=base,additional_steps_per_arm=1000,learning_rates=[.0001,.00005],microbatch=16,effective_batch=256,validation_samples=len(ids),checkpoint_sha256=hashlib.sha256((out/'source.pt').read_bytes()).hexdigest(),criterion='Compare terminal and intermediate validation CER, same samples and dropout RNG; test unused'),indent=2))
 for lr in [.0001,.00005]:
  model=CTCModel(len(data.vocab)).cuda();model.load_state_dict(state['model'])
  opt=torch.optim.Adam(model.parameters(),lr=lr);opt.load_state_dict(state['optimizer'])
  for g in opt.param_groups:g['lr']=lr
  # Keep source optimizer tensors independent from each arm.
  rng=random.Random();rng.setstate(state['random_state']);torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
  arm=out/str(lr);arm.mkdir(exist_ok=True);losses_window=[];history=[]
  for offset in range(0,1001):
   if offset%250==0:
    v=evaluate(model,data,ids,'cuda',16);v.pop('examples',None)
    row=dict(offset=offset,step=base+offset,validation=v,mean_training_loss=sum(losses_window)/len(losses_window) if losses_window else None)
    history.append(row);(arm/'metrics.json').write_text(json.dumps(history,indent=2));losses_window=[]
    print(json.dumps(dict(lr=lr,**row)),flush=True)
   if offset==1000:break
   model.train();opt.zero_grad(set_to_none=True);selected=sorted(rng.choices(train,k=256),key=lambda r:r[1]);batch_losses=[]
   for off in range(0,256,16):
    x,y,il,ol=collate([data.load(i) for i,n in selected[off:off+16]],'cuda')
    with precision('cuda'):logits=model(x,il)
    loss=loss_for(logits,y,il,ol)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    (loss/16).backward();batch_losses.append(loss.item())
   norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
   if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
   opt.step();losses_window.append(sum(batch_losses)/16)
   if offset%10==0:status(stage='running',lr=lr,source_step=base,arm_updates=offset+1,target_updates=1000,elapsed_seconds=time.monotonic()-start)
  config=dict(state['config']);config['lr']=lr
  save(arm/'final.pt',model,opt,base+1000,config,rng);results[str(lr)]=history
  del model,opt,x,y,logits,loss
  torch.cuda.empty_cache()
 (out/'results.json').write_text(json.dumps(results,indent=2));status(stage='completed',source_step=base,elapsed_seconds=time.monotonic()-start)
if __name__=='__main__':
 try:main()
 except Exception as e:
  status(stage='failed',error=str(e));raise
 finally:
  subprocess.run(['systemctl','--user','kill','--kill-who=main','--signal=SIGCONT','mathwriting-ctc-75k.service'],check=False)
