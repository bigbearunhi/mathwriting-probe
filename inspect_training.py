"""Fixed stratified sample of eligible training data, evaluated without updates."""
import json,random,shutil,subprocess,sys,hashlib
from pathlib import Path
import torch
from ctc_train import Data,CTCModel,collate,precision,collapse,edit_distance
out=Path('runs/training-viewer');out.mkdir(exist_ok=True)
frozen=out/'frozen.pt'
if not frozen.exists():shutil.copy2('runs/ctc-125k-lr5e5/best.pt',frozen)
state=torch.load(frozen,map_location='cpu',weights_only=False);torch.set_num_threads(4)
data=Data(Path('data/ctc-full'));model=CTCModel(len(data.vocab)).cuda();model.load_state_dict(state['model']);model.eval()
records=[];report={'step':state['step'],'seed':20260919,'checkpoint_sha256':hashlib.sha256(frozen.read_bytes()).hexdigest(),'selection':'512 per split, uniform without replacement from eligible training pool; membership does not prove sampled previously during training','splits':{}}
for split in ['train','synthetic']:
 chosen=random.Random(20260919).sample(data.rows[split],512);ids=[i for i,n in chosen];correct=errors=tokens=0
 with torch.inference_mode():
  for off in range(0,len(ids),16):
   samples=[data.load(i) for i in ids[off:off+16]];x,y,il,ol=collate(samples,'cuda')
   with precision('cuda'):pred=model(x,il).argmax(-1).cpu()
   for j,s in enumerate(samples):
    a=[data.vocab[k] for k in s[1].tolist()];b=[data.vocab[k] for k in collapse(pred[j,:int(il[j])].tolist())];e=edit_distance(a,b)
    correct+=a==b;errors+=e;tokens+=len(a)
    records.append(dict(id=s[2],split=split,truth=s[3],prediction=''.join(b),reference_tokens=a,predicted_tokens=b,correct=a==b,errors=e))
 report['splits'][split]=dict(samples=len(ids),correct=correct,exact_match=correct/len(ids),token_errors=errors,reference_tokens=tokens,token_error_rate=errors/tokens,rowids=ids)
 print(split,correct/len(ids),errors/tokens,flush=True)
(out/'predictions.json').write_text(json.dumps(records,ensure_ascii=False));(out/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
# Release GPU before extracting original ink.
del model,state,data;torch.cuda.empty_cache()
subprocess.run([sys.executable,'inspect_error.py','--input',str(out/'predictions.json'),'--out',str(out)],check=True)
p=out/'index.html';html=p.read_text();html=html.replace('笔迹识别结果查看器','训练集学习情况查看器')
summary=''.join(f"<tr><td>{s}</td><td>{r['samples']}</td><td>{r['correct']}</td><td>{r['exact_match']:.2%}</td><td>{r['token_error_rate']:.2%}</td></tr>" for s,r in report['splits'].items())
intro=f"<p>模型：{report['step']:,} 步最佳存档 · 固定随机种子 20260919 · 真人 train 与合成 synthetic 各 512 条。仅抽取符合当前训练长度条件的样本，不代表全量准确率。训练采用有放回采样，因此属于训练池不等于确认曾抽中过。</p><table cellpadding='8'><tr><th>来源</th><th>样本</th><th>完全正确</th><th>整式正确率</th><th>Token 错误率</th></tr>{summary}</table><p>一次只显示一条。原始笔迹来自 InkML；下方比较标准 LaTeX 与实际输出。此页没有运行 LaTeX 编译语法检查。</p>"
start=html.index('<p>来自已保存');end=html.index('</p>',start)+4;html=html[:start]+intro+html[end:]
html=html.replace('<select id="kind">','<select id="source"><option value="">全部来源</option><option value="train">真人 train</option><option value="synthetic">合成 synthetic</option></select><select id="kind" hidden>')
html=html.replace('<option value="symbol_error">符号错误</option><option value="number_error">数字错误</option><option value="syntax_error">确认语法错误</option>','')
html=html.replace('filtered=rows.filter(r=>','filtered=rows.filter(r=>(!$(\'source\').value||r.content.includes(\'来源：\'+$(\'source\').value+\' ·\'))&&')
html=html.replace("$('filter').onclick=filter;","$('source').onchange=filter;$('filter').onclick=filter;")
html=html.replace('<option value="all">全部样本</option>','<option value="all" selected>全部样本</option>');p.write_text(html)
print('Viewer complete:',p,flush=True)
