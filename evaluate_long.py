import json,random,collections
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,precision,collapse,edit_distance

torch.set_num_threads(2)
d=Data(Path('data/ctc-full'),max_frames=2000)
rows=d.db.execute("select rowid,length,min_frames from samples where split='synthetic' and length*2>512 order by length desc").fetchall()
chosen=rows[:30]+random.Random(672).sample(rows[30:],170)
s=torch.load('runs/ctc-50k/latest.pt',map_location='cpu',weights_only=False)
m=CTCModel(len(d.vocab)).cuda();m.load_state_dict(s['model']);m.eval();records=[]
with torch.no_grad():
 for k,(i,n,_) in enumerate(chosen):
  sample=d.load(i);x,y,il,ol=collate([sample],'cuda')
  with precision('cuda'):p=m(x,il).argmax(-1)[0,:il[0]].tolist()
  hyp=collapse(p);truth=sample[1].tolist();err=edit_distance(truth,hyp)
  records.append(dict(id=sample[2],frames=n*2,tokens=len(truth),truth=sample[3],prediction=''.join(d.vocab[t] for t in hyp),errors=err,token_error_rate=err/len(truth),correct=hyp==truth))
  if k%25==0:print('processed',k+1,flush=True)
r=dict(step=s['step'],samples=len(records),selection='30 longest + 170 seeded random remaining excluded synthetic samples; never used by our training',correct=sum(x['correct'] for x in records),token_error_rate=sum(x['errors'] for x in records)/sum(x['tokens'] for x in records),min_frames=min(x['frames'] for x in records),max_frames=max(x['frames'] for x in records),examples=sorted(records,key=lambda x:(x['token_error_rate'],-x['tokens']))[:8])
Path('runs/long-formula-results.json').write_text(json.dumps(dict(summary=r,records=records),ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False),flush=True)
