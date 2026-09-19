"""Read-only validation error analysis of a snapshot; no test data or training edits."""
import collections,json
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,precision,collapse

def align(a,b):
 dp=[[0]*(len(b)+1) for _ in range(len(a)+1)]
 for i in range(len(a)+1):dp[i][0]=i
 for j in range(len(b)+1):dp[0][j]=j
 for i in range(1,len(a)+1):
  for j in range(1,len(b)+1):dp[i][j]=min(dp[i-1][j]+1,dp[i][j-1]+1,dp[i-1][j-1]+(a[i-1]!=b[j-1]))
 i,j=len(a),len(b);ops=[]
 while i or j:
  if i and j and dp[i][j]==dp[i-1][j-1]+(a[i-1]!=b[j-1]):
   if a[i-1]!=b[j-1]:ops.append(('substitute',a[i-1],b[j-1]))
   i-=1;j-=1
  elif i and dp[i][j]==dp[i-1][j]+1:ops.append(('delete',a[i-1],''));i-=1
  else:ops.append(('insert','',b[j-1]));j-=1
 return ops[::-1]
def unbalanced(tokens):
 n=0
 for t in tokens:
  if t=='{':n+=1
  if t=='}':
   n-=1
   if n<0:return True
 return n!=0

torch.set_num_threads(4)
source=Path('runs/ctc-50k/latest.pt');state=torch.load(source,map_location='cpu',weights_only=False)
d=Data(Path('data/ctc-full'));ids=state['config']['validation_ids']
m=CTCModel(len(d.vocab)).cuda();m.load_state_dict(state['model']);m.eval()
counts=collections.Counter();details=collections.Counter();categories=collections.Counter();records=[];total_tokens=0
with torch.no_grad():
 for off in range(0,len(ids),8):
  samples=[d.load(i) for i in ids[off:off+8]];x,y,il,ol=collate(samples,'cuda')
  with precision('cuda'):pred=m(x,il).argmax(-1).cpu()
  for j,sample in enumerate(samples):
   a=[d.vocab[k] for k in sample[1].tolist()];b=[d.vocab[k] for k in collapse(pred[j,:il[j]].tolist())];ops=align(a,b);total_tokens+=len(a)
   for op,u,v in ops:
    counts[op]+=1;details[(op,u,v)]+=1
    # Strict, disjoint groups. These are token classes, not diagnoses of syntax validity.
    involved=[t for t in (u,v) if t]
    category='braces_only' if all(t in ['{','}'] for t in involved) else 'scripts_only' if all(t in ['^','_'] for t in involved) else 'other'
    categories[category]+=1
   records.append(dict(id=sample[2],truth=sample[3],prediction=''.join(b),correct=not ops,unbalanced_braces=unbalanced(b),reference_unbalanced=unbalanced(a),brace_only_error=bool(ops) and all(all(t in ['{','}'] for t in (u,v) if t) for op,u,v in ops),edits=ops))
  if off%256==0:print('processed',off+len(samples),flush=True)
wrong=[r for r in records if not r['correct']]
report=dict(step=state['step'],samples=len(records),correct=len(records)-len(wrong),wrong=len(wrong),reference_tokens=total_tokens,token_errors=sum(counts.values()),token_error_rate=sum(counts.values())/total_tokens,operations=dict(counts),token_classes=dict(categories),unbalanced_predictions=sum(r['unbalanced_braces'] for r in records),unbalanced_references=sum(r['reference_unbalanced'] for r in records),brace_only_wrong_samples=sum(r['brace_only_error'] for r in records),top_edits=[dict(operation=k[0],reference=k[1],prediction=k[2],count=v) for k,v in details.most_common(25)],examples=wrong[:15],limitations='Greedy CTC, one minimum-edit alignment; brace check is not a full LaTeX parser; balanced output may still be invalid.')
p=Path('runs/token-error-analysis');p.mkdir(exist_ok=True)
(p/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));(p/'predictions.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False),flush=True)
