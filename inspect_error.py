"""Create a standalone searchable handwriting/error viewer from saved predictions."""
import argparse,difflib,html,json,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
from error_types import align,tokenize

def select_records(rows,sample_id=None,reference=None,prediction=None):
 result=[]
 for r in rows:
  if sample_id and r['id']!=sample_id:continue
  if reference is not None or prediction is not None:
   if not any((reference is None or a==reference) and (prediction is None or b==prediction) for _,a,b in align(tokenize(r['truth']),tokenize(r['prediction']))):continue
  result.append(r)
 return result

def token_diff(truth,prediction):
 a,b=tokenize(truth),tokenize(prediction);parts=[]
 for op,i,j,k,l in difflib.SequenceMatcher(a=a,b=b,autojunk=False).get_opcodes():
  if op=='equal':parts.append(html.escape(''.join(a[i:j])))
  else:
   if i<j:parts.append('<del>'+html.escape(''.join(a[i:j]))+'</del>')
   if k<l:parts.append('<ins>'+html.escape(''.join(b[k:l]))+'</ins>')
 return ''.join(parts)

def stroke_svg(strokes):
 points=[p for s in strokes for p in s];xmin=min(p[0] for p in points);ymin=min(p[1] for p in points)
 width=max(max(p[0] for p in points)-xmin,1e-3);height=max(max(p[1] for p in points)-ymin,1e-3)
 scale=min(1100/width,280/height);w=width*scale+30;h=height*scale+30
 parts=[f'<svg role="img" aria-label="原始笔迹" viewBox="0 0 {w:.2f} {h:.2f}" xmlns="http://www.w3.org/2000/svg">']
 for s in strokes:
  coords=[((p[0]-xmin)*scale+15,(p[1]-ymin)*scale+15) for p in s]
  if len(coords)==1:
   x,y=coords[0];parts.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="1.6" fill="#152e4c"/>')
  else:
   pp=' '.join(f'{x:.2f},{y:.2f}' for x,y in coords);parts.append(f'<polyline points="{pp}" fill="none" stroke="#152e4c" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>')
 return ''.join(parts)+'</svg>'

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,default=Path('runs/error-types/details.json'));p.add_argument('--archive',type=Path,default=Path('data/mathwriting-2024.tgz'));p.add_argument('--id');p.add_argument('--reference');p.add_argument('--prediction');p.add_argument('--out',type=Path,default=Path('runs/error-viewer'));args=p.parse_args()
 loaded=json.loads(args.input.read_text())
 rows=select_records(loaded['records'] if isinstance(loaded,dict) else loaded,args.id,args.reference,args.prediction)
 if not rows:raise SystemExit('No matching error samples')
 wanted={r['id'] for r in rows};raw={};ns='{http://www.w3.org/2003/InkML}'
 with tarfile.open(args.archive,'r|gz') as tar:
  for member in tar:
   sid=Path(member.name).stem
   if sid not in wanted or not member.name.endswith('.inkml'):continue
   root=ET.fromstring(tar.extractfile(member).read());meta={a.attrib['type']:a.text for a in root.findall(ns+'annotation')}
   strokes=[[[float(v) for v in point.split()] for point in trace.text.split(',')] for trace in root.findall(ns+'trace')]
   raw[sid]=dict(split=Path(member.name).parent.name,label=meta['normalizedLabel'],svg=stroke_svg(strokes),strokes=strokes)
   if len(raw)==len(wanted):break
 if wanted-set(raw):raise RuntimeError('Missing source samples: '+str(wanted-set(raw)))
 cards=[]
 for r in rows:
  original=raw[r['id']]
  if original['label']!=r['truth']:raise ValueError('Reference/source mismatch: '+r['id'])
  edits=align(tokenize(r['truth']),tokenize(r['prediction']))
  content=f'<h2>{html.escape(r["id"])} · {("识别正确" if r["truth"]==r["prediction"] else "识别错误")}</h2><p>来源：{html.escape(original["split"])} · {len(edits)} 处 Token 编辑差异</p>'+original['svg']
  content+='<label>标准 LaTeX</label><pre>'+html.escape(r['truth'])+'</pre><label>模型输出</label><pre>'+html.escape(r['prediction'])+'</pre>'
  content+='<label>差异：红色删除线＝标准答案中缺失或被替换；绿色＝模型新增或替换后的内容</label><pre>'+token_diff(r['truth'],r['prediction'])+'</pre>'
  content+='<details><summary>逐项 Token 错误</summary><pre>'+html.escape('\n'.join(f'{op}: {a!r} → {b!r}' for op,a,b in edits))+'</pre></details>'
  cards.append(dict(correct=r['truth']==r['prediction'],id=r['id'],truth=r['truth'],prediction=r['prediction'],edits=edits,content=content,number_error=r.get('number_error',False),symbol_error=r.get('symbol_error',False),syntax_error=r.get('syntax_error',False)))
 payload=json.dumps(cards,ensure_ascii=False).replace('<','\\u003c')
 page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>笔迹识别结果查看器</title><style>body{font:16px system-ui;margin:30px auto;max-width:1200px;background:#f3f6fa;color:#172b4d;padding:0 20px}header{position:sticky;top:0;background:#f3f6fa;padding:12px 0;z-index:1}input,select,button{font:inherit;padding:8px;margin:4px;border:1px solid #aab7c8;border-radius:6px}article{background:white;border:1px solid #d9e0e8;border-radius:12px;padding:22px;margin:20px 0}svg{width:100%;max-height:310px;min-height:100px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f5f7fa;padding:14px;font:16px ui-monospace,monospace}label{color:#52677f}del{background:#ffe2e2;color:#9c1818}ins{background:#daf6df;color:#14622c;text-decoration:none}h2{font-size:18px}</style><h1>笔迹识别结果查看器</h1><p>来自已保存的预测结果；蓝色笔迹直接由原始 InkML 坐标绘制。验证样本不等于参与训练的样本。Token 差异不自动代表语法错误。</p><header><select id="result"><option value="wrong">识别错误</option><option value="correct">识别正确</option><option value="all">全部样本</option></select><input id="q" placeholder="样本 ID 或 LaTeX 内容"><select id="kind"><option value="">所有类型</option><option value="symbol_error">符号错误</option><option value="number_error">数字错误</option><option value="syntax_error">确认语法错误</option></select><input id="ref" placeholder="标准 Token，例如 x"><input id="pred" placeholder="错误 Token，例如 X"><button id="filter">筛选</button><div id="count"></div></header><main id="cards"></main><button id="prev">上一条</button><button id="more">下一条</button><script>const rows=PAYLOAD;let filtered=rows,shown=0;const $=id=>document.getElementById(id);function render(){let r=filtered[shown];$('cards').innerHTML=r?'<article>'+r.content+'</article>':'<p>没有匹配的样本</p>';$('count').textContent=filtered.length?`匹配 ${filtered.length} 条，当前第 ${shown+1} 条`:'匹配 0 条';$('prev').disabled=shown===0;$('more').disabled=shown>=filtered.length-1}function filter(){let mode=$('result').value,q=$('q').value,kind=$('kind').value,a=$('ref').value,b=$('pred').value;filtered=rows.filter(r=>(mode==='all'||(mode==='correct'?r.correct:!r.correct))&&(!q||r.id.includes(q)||r.truth.includes(q)||r.prediction.includes(q))&&(!kind||r[kind])&&(!(a||b)||r.edits.some(e=>(!a||e[1]===a)&&(!b||e[2]===b))));shown=0;$('cards').innerHTML='';render()}$('filter').onclick=filter;$('result').onchange=()=>{$('kind').value='';filter()};$('more').onclick=()=>{if(shown+1<filtered.length){shown++;render()}};$('prev').onclick=()=>{if(shown>0){shown--;render()}};$('q').addEventListener('keydown',e=>{if(e.key==='Enter')filter()});filter();</script></html>'''.replace('PAYLOAD',payload)
 args.out.mkdir(parents=True,exist_ok=True);(args.out/'index.html').write_text(page)
 (args.out/'source_samples.json').write_text(json.dumps({sid:{k:v for k,v in r.items() if k!='svg'} for sid,r in raw.items()},ensure_ascii=False))
 (args.out/'manifest.json').write_text(json.dumps(dict(input=str(args.input.resolve()),archive=str(args.archive.resolve()),samples=len(rows),ids=sorted(wanted)),indent=2))
 print(f'Created {args.out / "index.html"}: {len(rows)} samples, raw labels verified')
if __name__=='__main__':main()
