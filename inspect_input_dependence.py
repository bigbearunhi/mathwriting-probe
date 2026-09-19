"""Validation-only diagnostic: replace pen features with zeros, retaining lengths."""
import json
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,evaluate

torch.set_num_threads(4)
class ZeroData:
 def __init__(self,data):self.data=data;self.vocab=data.vocab
 def load(self,i):
  x,*rest=self.data.load(i)
  return (torch.zeros_like(x),*rest)
data=Data(Path('data/ctc-full'))
results={}
for name,path in [('old','runs/ctc-full/latest.pt'),('lower_lr','runs/ctc-full-lr-diagnosis/diagnostic.pt')]:
 state=torch.load(path,map_location='cpu',weights_only=False)
 model=CTCModel(len(data.vocab)).cuda();model.load_state_dict(state['model'])
 ids=state['config']['validation_ids'][:128]
 results[name]={'step':state['step']}
 for mode,d in [('real',data),('zero_features',ZeroData(data))]:
  r=evaluate(model,d,ids,'cuda',8)
  results[name][mode]={k:v for k,v in r.items() if k!='examples'}
 del model
print(json.dumps(results,indent=2))
Path('runs/ctc-input-dependence.json').write_text(json.dumps(results,indent=2))
