"""Disposable throughput benchmark; never writes training checkpoints."""
import argparse,gc,json,random,statistics,time
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,loss_for,precision
p=argparse.ArgumentParser();p.add_argument('--microbatch',type=int,required=True);a=p.parse_args();b=a.microbatch
torch.set_num_threads(4);torch.manual_seed(123);torch.backends.cuda.matmul.allow_tf32=True
torch.cuda.set_per_process_memory_fraction(.85)
d=Data(Path('data/ctc-full'));rows=d.rows['train']+d.rows['synthetic'];rng=random.Random(987)
batches=[]
for _ in range(14):
 batch=rng.choices(rows,k=256);batch.sort(key=lambda r:r[1]);batches.append(batch)
m=CTCModel(len(d.vocab)).cuda();opt=torch.optim.Adam(m.parameters(),lr=.0001)
result={'microbatch':b,'effective_batch':256,'accumulation':256//b}
try:
 durations=[]
 for j,batch in enumerate(batches):
  torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
  for off in range(0,256,b):
   samples=[d.load(i) for i,n in batch[off:off+b]]
   x,y,il,ol=collate(samples,'cuda')
   with precision('cuda'):logits=m(x,il)
   loss=loss_for(logits,y,il,ol)
   if not torch.isfinite(loss):raise RuntimeError('nonfinite loss')
   (loss/(256//b)).backward()
  torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();torch.cuda.synchronize()
  if j>=2:durations.append(time.perf_counter()-start)
 result.update(median_seconds=statistics.median(durations),mean_seconds=statistics.mean(durations),peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,seconds=durations)
 # Worst-length microbatch, using genuine train targets and 512-frame inputs.
 opt.zero_grad(set_to_none=True);gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
 longest=sorted(rows,key=lambda r:r[1],reverse=True)[:b]
 x,y,il,ol=collate([d.load(i) for i,n in longest],'cuda')
 with precision('cuda'):logits=m(x,il)
 loss=loss_for(logits,y,il,ol);loss.backward()
 result.update(longest_frames=int(il.max()),long_batch_allocated_mib=torch.cuda.max_memory_allocated()/2**20,long_batch_reserved_mib=torch.cuda.max_memory_reserved()/2**20,long_batch_ok=bool(torch.isfinite(loss)))
except torch.cuda.OutOfMemoryError:
 result['oom']=True
print(json.dumps(result),flush=True)
Path(f'runs/microbatch-{b}.json').write_text(json.dumps(result,indent=2))
