import json,random
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,precision,collapse

torch.set_num_threads(4)
p=Path('runs/ctc-full-lr-diagnosis')
s=torch.load(p/'diagnostic.pt',map_location='cpu',weights_only=False)
d=Data(Path('data/ctc-full'));m=CTCModel(len(d.vocab)).cuda();m.load_state_dict(s['model']);m.eval()
correct=[];ids=s['config']['validation_ids']
with torch.no_grad():
 for off in range(0,len(ids),8):
  samples=[d.load(i) for i in ids[off:off+8]]
  x,y,il,ol=collate(samples,'cuda')
  with precision('cuda'):pred=m(x,il).argmax(-1).cpu()
  for j,sample in enumerate(samples):
   hyp=collapse(pred[j,:il[j]].tolist())
   if hyp==sample[1].tolist() and 1 not in hyp:
    correct.append(dict(id=sample[2],truth=sample[3],prediction=''.join(d.vocab[k] for k in hyp)))
r=dict(step=s['step'],samples=len(ids),correct_count=len(correct),correct=correct)
Path('runs/correct-evidence.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False))
