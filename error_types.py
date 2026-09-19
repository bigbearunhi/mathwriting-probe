"""Classify saved truth/prediction pairs; CPU-only, no checkpoint mutation.
Usage: python3 error_types.py --input runs/token-error-analysis/predictions.json
"""
import argparse,collections,csv,json,os,re,shutil,subprocess,tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
TOKEN=re.compile(r'\\(?:mathbb\{[a-zA-Z]\}|begin\{[a-z]+\}|end\{[a-z]+\}|operatorname\*|[a-zA-Z]+|.)|[^\\]',re.S)
VOCAB=Path(__file__).parent/'data/ctc-full/vocab.json'
ALLOWED=set(json.loads(VOCAB.read_text()))
STRUCT={'{','}','^','_',r'\frac',r'\sqrt',r'\hat',r'\tilde',r'\overline',r'\underline',r'\vec',r'\dot',r'\mathbb','&',r'\\'}
def tokenize(s):
 t=TOKEN.findall(s)
 if ''.join(t)!=s:raise ValueError('Incomplete tokenization')
 return t

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

def kind(t):
 if not t:return None
 if t.isascii() and t.isdigit():return 'number'
 if t.isspace():return 'whitespace'
 if t in STRUCT or t.startswith((r'\begin{',r'\end{')):return 'structure'
 return 'symbol'

def classify_edits(truth,prediction):
 ops=align(tokenize(truth),tokenize(prediction));edits=[];kinds=set()
 for op,a,b in ops:
  ks=sorted({kind(t) for t in (a,b) if t});kinds.update(ks)
  edits.append(dict(operation=op,reference=a,prediction=b,kinds=ks))
 return dict(number_error='number' in kinds,symbol_error='symbol' in kinds,structure_token_error='structure' in kinds,whitespace_error='whitespace' in kinds,edits=edits)

def check_syntax(s):
 try:tokens=tokenize(s)
 except ValueError:return dict(status='invalid',reason='Incomplete command')
 # Only MathWriting math tokens, never arbitrary TeX file I/O or macro definitions.
 if len(s)>10000 or any(t not in ALLOWED for t in tokens):return dict(status='unknown',reason='Outside supported MathWriting vocabulary')
 if not shutil.which('pdflatex'):return dict(status='unknown',reason='pdflatex unavailable')
 pre=r'\documentclass{article}\usepackage{amsmath,amssymb,amsfonts}\pagestyle{empty}\begin{document}\['
 with tempfile.TemporaryDirectory(prefix='mathwriting-syntax-') as tmp:
  Path(tmp,'check.tex').write_text(pre+' '.join(tokens)+r'\]\end{document}')
  env=dict(os.environ,openin_any='p',openout_any='p')
  try:
   r=subprocess.run(['pdflatex','-no-shell-escape','-interaction=nonstopmode','-halt-on-error','check.tex'],cwd=tmp,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=8)
  except subprocess.TimeoutExpired:return dict(status='unknown',reason='Compiler timeout')
  log=r.stdout.decode(errors='replace')
  if r.returncode==0:return dict(status='valid',reason='Compiles with amsmath/amssymb/amsfonts')
  match=re.search(r'^! (.+)',log,re.M);reason=match.group(1) if match else log[-250:]
  unknown=any(x in log for x in ['Undefined control sequence','not found','TeX capacity exceeded','Emergency stop'])
  return dict(status='unknown' if unknown else 'invalid',reason=reason)

def symbol_breakdown(rows):
 stats=collections.defaultdict(collections.Counter);pairs=collections.Counter()
 for r in rows:
  for t in tokenize(r['truth']):
   if kind(t)=='symbol':stats[t]['reference_count']+=1
  for op,a,b in align(tokenize(r['truth']),tokenize(r['prediction'])):
   if op=='substitute':
    if kind(a)=='symbol':stats[a]['substituted']+=1
    if kind(b)=='symbol':stats[b]['false_substitution']+=1
    if kind(a)=='symbol' or kind(b)=='symbol':pairs[a,b]+=1
   elif op=='delete' and kind(a)=='symbol':stats[a]['deleted']+=1
   elif op=='insert' and kind(b)=='symbol':stats[b]['inserted']+=1
 symbols=[]
 for t,c in stats.items():
  row=dict(symbol=t,**{k:c[k] for k in ['reference_count','substituted','deleted','false_substitution','inserted']})
  row['reference_errors']=c['substituted']+c['deleted']
  row['error_rate']=row['reference_errors']/c['reference_count'] if c['reference_count'] else None
  symbols.append(row)
 symbols.sort(key=lambda r:(-r['reference_errors'],-r['reference_count'],r['symbol']))
 return dict(symbols=symbols,confusions=[dict(reference=a,prediction=b,count=n) for (a,b),n in pairs.most_common()],method='Reference error rate=(substitutions+deletions)/reference occurrences. False substitutions and insertions reported separately. One deterministic edit alignment, not visual character detection; rare symbol rates are unstable.')

def write_symbol_breakdown(rows,out):
 report=symbol_breakdown(rows);out.mkdir(parents=True,exist_ok=True)
 (out/'symbol_breakdown.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 for key,name in [('symbols','symbol_errors.csv'),('confusions','symbol_confusions.csv')]:
  records=report[key]
  with (out/name).open('w',newline='') as f:
   if records:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 return report

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,default=Path('runs/error-types'));p.add_argument('--workers',type=int,default=4);args=p.parse_args()
 rows=json.loads(args.input.read_text());strings=sorted({r[k] for r in rows for k in ['truth','prediction']})
 cache={}
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  for i,(s,result) in enumerate(zip(strings,pool.map(check_syntax,strings))):
   cache[s]=result
   if (i+1)%250==0:print(f'Checked {i+1}/{len(strings)} unique strings',flush=True)
 counts=collections.Counter();output=[]
 for r in rows:
  c=classify_edits(r['truth'],r['prediction']);ref=cache[r['truth']];pred=cache[r['prediction']]
  wrong=bool(c['edits']);counts['wrong']+=wrong;counts['samples']+=1
  syntax_error=pred['status']=='invalid'
  record=dict(id=r['id'],truth=r['truth'],prediction=r['prediction'],**c,syntax_error=syntax_error,syntax=pred,reference_syntax=ref)
  for key in ['number_error','symbol_error','structure_token_error','whitespace_error']:
   counts[key]+=int(c[key])
  counts['syntax_error']+=syntax_error;counts['syntax_error_on_wrong_samples']+=wrong and syntax_error
  counts['introduced_syntax_error']+=syntax_error and ref['status']=='valid'
  counts['syntax_unknown']+=pred['status']=='unknown';counts['reference_invalid']+=ref['status']=='invalid';counts['reference_unknown']+=ref['status']=='unknown'
  output.append(record)
 args.out.mkdir(parents=True,exist_ok=True)
 write_symbol_breakdown(rows,args.out)
 summary=dict(input=str(args.input.resolve()),counts=dict(counts),percent_of_wrong_samples={k:100*counts[k]/max(counts['wrong'],1) for k in ['number_error','symbol_error','syntax_error_on_wrong_samples','structure_token_error','whitespace_error']},method='Deterministic minimum Token edit alignment; categories may overlap. LaTeX compiled independently of reference, with shell escape disabled and math vocabulary allowlist.',limitations=['Digit involvement is token-level, not segmentation accuracy; digit/letter substitution counts in both categories.','Structure-token edits are not automatically syntax errors.','Compilation success is not semantic correctness or canonical-label equivalence.','Unsupported input/compiler failures are unknown, not syntax errors.','Single minimum-edit alignment may be ambiguous.'])
 (args.out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2));(args.out/'details.json').write_text(json.dumps(output,ensure_ascii=False,indent=2))
 with (args.out/'details.csv').open('w',newline='') as f:
  keys=['id','truth','prediction','number_error','symbol_error','syntax_error','structure_token_error','whitespace_error'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows({k:r[k] for k in keys} for r in output)
 print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
