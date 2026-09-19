"""Resume a trusted StrokeConvCTC checkpoint with optimizer and RNG states."""
import argparse,json,os,random,shutil,time
from pathlib import Path
import torch
from ctc_train import Data,collate,evaluate,loss_for,precision,save
from stroke_conv import StrokeConvCTC

def restore(path,model,opt,rng):
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['config'].get('architecture')!='StrokeConvCTC':
        raise ValueError('Expected a StrokeConvCTC checkpoint')
    model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer'])
    rng.setstate(state['random_state']);torch.set_rng_state(state['torch_rng'])
    if next(model.parameters()).is_cuda and state.get('cuda_rng'):
        torch.cuda.set_rng_state(state['cuda_rng'][0],device=next(model.parameters()).device)
    return state

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=Path('data/ctc-tolerance-002'))
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=5000,help='Absolute total optimizer steps')
    p.add_argument('--microbatch',type=int,default=16)
    p.add_argument('--check-only',action='store_true',help='Load and run one forward pass, without updates')
    args=p.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA GPU required')
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=True
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=state['config'].copy();start_step=state['step'];del state
    if cfg.get('repeat')!=2:raise ValueError('This convolution expects repeat=2')
    batch=cfg['effective_batch']
    if args.microbatch<=0 or batch%args.microbatch:raise ValueError('Microbatch must divide effective batch')
    report=json.loads((args.cache/'preparation.json').read_text())
    if report['tolerance']!=cfg['tolerance']:raise ValueError('Cache tolerance mismatch')
    data=Data(args.cache,max_frames=cfg['max_frames'],repeat=cfg['repeat'])
    eligible={i for i,n in data.rows['valid']};ids=[]
    excluded=set(cfg.get('validation_excluded',[]))
    for name in cfg['validation_sample_ids']:
        if name in excluded:continue
        row=data.db.execute("SELECT rowid FROM samples WHERE split='valid' AND id=?",(name,)).fetchone()
        if not row or row[0] not in eligible:raise ValueError('Validation cache differs: '+name)
        ids.append(row[0])
    if len(ids)!=cfg['valid_samples']:raise ValueError('Validation sample count mismatch')
    model=StrokeConvCTC(len(data.vocab),**{k:cfg[k] for k in ['dim','layers','heads','ff','dropout']}).cuda()
    opt=torch.optim.Adam(model.parameters(),lr=cfg['lr']);rng=random.Random()
    state=restore(args.checkpoint,model,opt,rng);del state
    if args.check_only:
        model.eval()
        with torch.no_grad(),precision('cuda'):
            x,y,il,ol=collate([data.load(ids[0])],'cuda');z=model(x,il)
        if not torch.isfinite(z).all():raise RuntimeError('Nonfinite forward output')
        print(json.dumps(dict(restored_step=start_step,validation_samples=len(ids),output_shape=list(z.shape),lr=opt.param_groups[0]['lr'])));return
    if args.steps<=start_step:raise ValueError('--steps must exceed checkpoint step')
    if args.out.exists():raise FileExistsError('Use a new output directory')
    args.out.mkdir(parents=True)
    cfg.update(source=str(args.checkpoint),steps=args.steps,microbatch=args.microbatch,validation_ids=ids,cache=str(args.cache))
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2))
    def status(stage,step,**kw):
        f=args.out/'status.partial';f.write_text(json.dumps(dict(stage=stage,step=step,target=args.steps,updated_unix=time.time(),**kw)));os.replace(f,args.out/'status.json')
    status('baseline_validation',start_step)
    v=evaluate(model,data,ids,'cuda',args.microbatch)
    best=(-v['token_error_rate'],v['exact_match'])
    shutil.copy2(args.checkpoint,args.out/'best.pt')
    (args.out/'best.json').write_text(json.dumps(dict(step=start_step,token_error_rate=-best[0],exact_match=best[1])))
    (args.out/f'validation_{start_step:06d}.json').write_text(json.dumps(v,indent=2))
    train=data.rows['train']+data.rows['synthetic'];started=time.monotonic();accum=batch//args.microbatch
    for step in range(start_step+1,args.steps+1):
        model.train();opt.zero_grad(set_to_none=True);losses=[]
        chosen=sorted(rng.choices(train,k=batch),key=lambda r:r[1])
        for off in range(0,batch,args.microbatch):
            x,y,il,ol=collate([data.load(i) for i,n in chosen[off:off+args.microbatch]],'cuda')
            with precision('cuda'):z=model(x,il)
            loss=loss_for(z,y,il,ol)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            (loss/accum).backward();losses.append(loss.item())
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradient')
        opt.step()
        if step%10==0:
            metrics=dict(loss=sum(losses)/len(losses),elapsed_seconds=time.monotonic()-started)
            status('training',step,**metrics);print(json.dumps(dict(step=step,**metrics)),flush=True)
        if step%cfg['eval_every']==0 or step==args.steps:
            status('validation',step);v=evaluate(model,data,ids,'cuda',args.microbatch)
            (args.out/f'validation_{step:06d}.json').write_text(json.dumps(v,indent=2))
            save(args.out/'latest.pt',model,opt,step,cfg,rng)
            shutil.copy2(args.out/'latest.pt',args.out/f'checkpoint_{step:06d}.pt')
            score=(-v['token_error_rate'],v['exact_match'])
            if score>best:
                shutil.copy2(args.out/'latest.pt',args.out/'best.pt');best=score
                (args.out/'best.json').write_text(json.dumps(dict(step=step,token_error_rate=-score[0],exact_match=score[1])))
            print(json.dumps(dict(step=step,validation={k:x for k,x in v.items() if k!='examples'})),flush=True)
    status('completed',args.steps)
if __name__=='__main__':main()
