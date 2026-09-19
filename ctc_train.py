"""Independent MathWriting CTC baseline; see CTC_PLAN.md for deviations."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.checkpoint import checkpoint
from probe import tokenize


def collapse(ids):
    out, previous = [], None
    for i in ids:
        if i != previous and i != 0:
            out.append(i)
        previous = i
    return out


def edit_distance(a, b):
    row = list(range(len(b)+1))
    for i, x in enumerate(a, 1):
        new = [i]
        for j, y in enumerate(b, 1):
            new.append(min(new[-1]+1, row[j]+1, row[j-1]+(x != y)))
        row = new
    return row[-1]


class Block(nn.Module):
    def __init__(self, dim, heads, ff, dropout, pre_norm=True):
        super().__init__()
        self.attention=nn.MultiheadAttention(dim,heads,dropout=dropout,batch_first=True)
        self.ff=nn.Sequential(nn.Linear(dim,ff),nn.SiLU(),nn.Dropout(dropout),nn.Linear(ff,dim))
        self.norm1,self.norm2=nn.LayerNorm(dim),nn.LayerNorm(dim)
        self.drop=nn.Dropout(dropout)
        self.pre_norm=pre_norm

    def forward(self,x,padding):
        if self.pre_norm:
            normalized=self.norm1(x)
            a=self.attention(normalized,normalized,normalized,key_padding_mask=padding,need_weights=False)[0]
            x=x+self.drop(a)
            return x+self.drop(self.ff(self.norm2(x)))
        a=self.attention(x,x,x,key_padding_mask=padding,need_weights=False)[0]
        x=self.norm1(x+self.drop(a))
        return self.norm2(x+self.drop(self.ff(x)))


class CTCModel(nn.Module):
    def __init__(self,vocab_size,dim=512,layers=11,heads=8,ff=2048,dropout=.15,checkpointing=False,pre_norm=True):
        super().__init__()
        self.dim,self.checkpointing=dim,checkpointing
        self.input=nn.Linear(10,dim)
        self.layers=nn.ModuleList([Block(dim,heads,ff,dropout,pre_norm) for _ in range(layers)])
        self.drop=nn.Dropout(dropout)
        self.final_norm=nn.LayerNorm(dim) if pre_norm else nn.Identity()
        self.output=nn.Linear(dim,vocab_size)

    def forward(self,x,lengths):
        x=self.input(x)
        pos=torch.arange(x.size(1),device=x.device,dtype=torch.float32)[:,None]
        freq=torch.exp(torch.arange(0,self.dim,2,device=x.device)*(-math.log(10000)/self.dim))
        pe=torch.zeros(x.size(1),self.dim,device=x.device)
        pe[:,0::2],pe[:,1::2]=(pos*freq).sin(),(pos*freq).cos()
        x=self.drop(x+pe.to(x.dtype))
        padding=torch.arange(x.size(1),device=x.device)[None,:]>=lengths.to(x.device)[:,None]
        for layer in self.layers:
            x=checkpoint(layer,x,padding,use_reentrant=False) if self.checkpointing and self.training else layer(x,padding)
        return self.output(self.final_norm(x))


class Data:
    def __init__(self,cache,max_frames=512,repeat=2):
        self.db=sqlite3.connect(f'file:{(cache/"features.sqlite").resolve()}?mode=ro',uri=True)
        self.vocab=json.loads((cache/'vocab.json').read_text())
        self.lookup={t:i for i,t in enumerate(self.vocab)}
        self.repeat=repeat
        self.rows,self.excluded={},{}
        for split in ('train','synthetic','valid'):
            rows=self.db.execute('SELECT rowid,length,min_frames FROM samples WHERE split=?',(split,)).fetchall()
            self.rows[split]=[(i,n*repeat) for i,n,m in rows if m<=n*repeat<=max_frames]
            self.excluded[split]=dict(total=len(rows),retained=len(self.rows[split]),
                too_short=sum(n*repeat<m for _,n,m in rows),too_long=sum(n*repeat>max_frames for _,n,m in rows))

    def load(self,index):
        name,label,tokens,n,blob=self.db.execute('SELECT id,label,tokens,length,features FROM samples WHERE rowid=?',(index,)).fetchone()
        x=np.frombuffer(blob,dtype='<f4').reshape(n,10).copy()
        # Fixed repeat supplies additional CTC alignment positions, independent of labels.
        x=torch.from_numpy(x).repeat_interleave(self.repeat,dim=0)
        target=[self.lookup.get(t,1) for t in json.loads(tokens)]
        return x,torch.tensor(target,dtype=torch.long),name,label


def collate(samples,device):
    x=pad_sequence([s[0] for s in samples],batch_first=True).to(device)
    ilen=torch.tensor([len(s[0]) for s in samples],dtype=torch.long)
    y=torch.cat([s[1] for s in samples]).to(device)
    olen=torch.tensor([len(s[1]) for s in samples],dtype=torch.long)
    return x,y,ilen,olen


def precision(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device=='cuda' else nullcontext()


def loss_for(logits,y,ilen,olen):
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0,1),y,ilen,olen,
                      blank=0,reduction='mean',zero_infinity=False)


@torch.no_grad()
def evaluate(model,data,ids,device,batch_size):
    model.eval()
    total_loss,errors,tokens,exact,blank,frames,unknown=0.,0,0,0,0,0,0
    examples=[]
    for start in range(0,len(ids),batch_size):
        samples=[data.load(i) for i in ids[start:start+batch_size]]
        x,y,ilen,olen=collate(samples,device)
        with precision(device):
            logits=model(x,ilen)
        loss=loss_for(logits,y,ilen,olen)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite validation loss')
        total_loss+=loss.item()*len(samples)
        pred=logits.argmax(-1).cpu()
        for row,sample in enumerate(samples):
            path=pred[row,:ilen[row]].tolist()
            hyp=collapse(path)
            truth=sample[1].tolist()
            reference_tokens=tokenize(sample[3])
            predicted_tokens=[data.vocab[i] for i in hyp]
            errors+=edit_distance(reference_tokens,predicted_tokens)
            tokens+=len(reference_tokens)
            unknown+=truth.count(1)
            exact+=int(hyp==truth and 1 not in truth)
            blank+=path.count(0)
            frames+=len(path)
            if len(examples)<12:
                examples.append(dict(id=sample[2],truth=sample[3],
                    prediction=''.join(data.vocab[i] for i in hyp)))
    return dict(samples=len(ids),ctc_loss=total_loss/max(len(ids),1),
                token_error_rate=errors/max(tokens,1),exact_match=exact/max(len(ids),1),
                blank_fraction=blank/max(frames,1),unknown_reference_tokens=unknown,examples=examples)


def save(path,model,optimizer,step,config,rng):
    temp=path.with_suffix('.partial')
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,
                    config=config,random_state=rng.getstate(),torch_rng=torch.get_rng_state(),
                    cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),temp)
    os.replace(temp,path)


def run(args):
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'config.json').exists() and not args.resume:
        raise FileExistsError('Run exists: use --resume or a fresh --out directory')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng=random.Random(42)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    if device=='cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This trial requires BF16-capable GPU')
    torch.backends.cuda.matmul.allow_tf32=True
    data=Data(args.cache,args.max_frames,args.repeat)
    train=data.rows['train']+data.rows['synthetic']
    valid=data.rows['valid'].copy()
    random.Random(123).shuffle(valid)
    valid_ids=[i for i,_ in valid[:args.valid_samples]]
    if not train or not valid_ids:
        raise ValueError('Empty train or validation set')
    if args.effective_batch % args.microbatch:
        raise ValueError('effective_batch must be divisible by microbatch')
    model=CTCModel(len(data.vocab),checkpointing=args.checkpointing,pre_norm=args.norm_order=='pre').to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=.001)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(parameters=sum(p.numel() for p in model.parameters()),device=device,
                  torch_version=torch.__version__,layers=11,dim=512,heads=8,head_dim=64,
                  ff=2048,dropout=.15,activation='SiLU/Swish',optimizer='Adam',lr=.001,
                  precision='BF16 forward / FP32 CTC and optimizer',vocab=data.vocab,
                  filtering=data.excluded,validation_ids=valid_ids,
                  deviations=['Independent Bezier approximation','head_dim64/FFN2048 chosen to match ~35M',
                  'sinusoidal position and normalization order are implementation choices',
                  'fixed curve repetition and length filtering','gradient clipping norm1',
                  'single-GPU accumulation; pilot budget, not full 100k'])
    config['source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ('ctc_train.py','ctc_data.py','probe.py')}
    first_step=0
    if args.resume:
        state=torch.load(args.out/'latest.pt',map_location=device,weights_only=False)
        old=state['config']
        for key in ('vocab','max_frames','repeat','effective_batch','microbatch','cache','norm_order'):
            if key=='norm_order' and key not in old:
                old[key]='post'
            if old[key]!=config[key]:
                raise ValueError(f'Resume configuration mismatch: {key}')
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        first_step=state['step']
        rng.setstate(state['random_state'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
    (args.out/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2))
    print(json.dumps({k:config[k] for k in ('parameters','device','microbatch','effective_batch','filtering') }),flush=True)
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    initial=evaluate(model,data,valid_ids,device,args.microbatch)
    (args.out/f'validation_{first_step:06d}.json').write_text(json.dumps(initial,ensure_ascii=False,indent=2))
    print(json.dumps(dict(step=first_step,validation={k:v for k,v in initial.items() if k!='examples'})),flush=True)
    start=time.monotonic()
    accum=args.effective_batch//args.microbatch
    log=(args.out/'metrics.jsonl').open('a',buffering=1)
    train_seconds=0.
    for step in range(first_step+1,args.steps+1):
        tick=time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        # Shuffle across both train sources; sort each effective batch to reduce padding.
        selected=rng.choices(train,k=args.effective_batch)
        selected.sort(key=lambda r:r[1])
        losses=[]
        for offset in range(0,len(selected),args.microbatch):
            samples=[data.load(i) for i,_ in selected[offset:offset+args.microbatch]]
            x,y,ilen,olen=collate(samples,device)
            with precision(device):
                logits=model(x,ilen)
            loss=loss_for(logits,y,ilen,olen)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite CTC loss at step {step}')
            (loss/accum).backward()
            losses.append(loss.item())
        norm=nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):
            raise RuntimeError('Nonfinite gradients')
        optimizer.step()
        if device=='cuda':
            torch.cuda.synchronize()
        seconds=time.monotonic()-tick
        train_seconds+=seconds
        row=dict(step=step,train_ctc_loss=sum(losses)/len(losses),gradient_norm=norm.item(),
                 step_seconds=seconds,elapsed_seconds=time.monotonic()-start,
                 max_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device=='cuda' else 0,
                 max_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else 0)
        log.write(json.dumps(row)+'\n')
        if step==first_step+1 or step%10==0:
            print(json.dumps(row),flush=True)
        if step%args.eval_every==0 or step==args.steps:
            validation=evaluate(model,data,valid_ids,device,args.microbatch)
            (args.out/f'validation_{step:06d}.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
            save(args.out/'latest.pt',model,optimizer,step,config,rng)
            summary=dict(step=step,validation={k:v for k,v in validation.items() if k!='examples'},
                         mean_training_step_seconds=train_seconds/(step-first_step),**{k:row[k] for k in ('max_allocated_mib','max_reserved_mib')})
            (args.out/'status.json').write_text(json.dumps(summary,indent=2))
            print(json.dumps(summary),flush=True)
    log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=200)
    p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--effective-batch',type=int,default=256)
    p.add_argument('--max-frames',type=int,default=512)
    p.add_argument('--repeat',type=int,default=2)
    p.add_argument('--valid-samples',type=int,default=512)
    p.add_argument('--eval-every',type=int,default=50)
    p.add_argument('--checkpointing',action='store_true')
    p.add_argument('--norm-order',choices=['pre','post'],default='pre')
    p.add_argument('--resume',action='store_true')
    run(p.parse_args())
