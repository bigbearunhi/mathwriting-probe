import json,sqlite3,random,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from ctc_data import curve_features,NS
OUT=Path('runs/preprocess-viewer');OUT.mkdir(exist_ok=True)
db=sqlite3.connect('file:data/ctc-full/features.sqlite?mode=ro',uri=True);rng=random.Random(20260919)
raw=json.loads(Path('runs/error-viewer/source_samples.json').read_text());pred=json.loads(Path('runs/token-error-analysis/predictions.json').read_text());selected={}
for correct in (False,True):
 pool=[r for r in pred if r['correct']==correct]
 for r in rng.sample(pool,min(80,len(pool))):selected[r['id']]={**raw[r['id']],'prediction':r['prediction'],'recognition':'正确' if correct else '错误'}
wanted={}
for split in ('train','synthetic'):
 rows=db.execute('SELECT id,length,min_frames FROM samples WHERE split=? ORDER BY id',(split,)).fetchall()
 for r in rng.sample([r for r in rows if r[2]<=r[1]*2<=512],60):wanted[r[0]]=split
 for r in sorted(rows,key=lambda r:r[1],reverse=True)[:10]:wanted[r[0]]=split
with tarfile.open('data/mathwriting-2024.tgz','r|gz') as tar:
 for m in tar:
  sid=Path(m.name).stem
  if sid not in wanted or not m.name.endswith('.inkml'):continue
  root=ET.fromstring(tar.extractfile(m).read());meta={e.attrib['type']:e.text for e in root.findall(NS+'annotation')}
  strokes=[[[float(v) for v in p.split()] for p in e.text.split(',')] for e in root.findall(NS+'trace')]
  selected[sid]=dict(split=wanted.pop(sid),label=meta['normalizedLabel'],strokes=strokes,recognition='未评估')
  if not wanted:break
assert not wanted

def decode(features,start):
 pos=np.array(start,dtype=float);curves=[]
 for f in features:
  d=f[:2].astype(float);length=max(np.linalg.norm(d),1e-6)
  def turn(v,a):return np.array([v[0]*np.cos(a)-v[1]*np.sin(a),v[0]*np.sin(a)+v[1]*np.cos(a)])
  unit=d/np.linalg.norm(d) if np.linalg.norm(d)>0 else np.array([1.,0.]);end=pos+d
  cp=np.stack([pos,pos+turn(unit,float(f[4]))*f[2]*length,end+turn(-unit,float(f[5]))*f[3]*length,end])
  curves.append(dict(cp=cp.tolist(),down=bool(f[9])));pos=end
 return curves

def distances(points,curves):
 u=np.linspace(0,1,33);b=np.stack([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3],1)
 paths=[b@np.array(c['cp']) for c in curves if c['down']];a=np.concatenate([p[:-1] for p in paths]);d=np.concatenate([np.diff(p,axis=0) for p in paths]);dd=(d*d).sum(1);out=[]
 for chunk in np.array_split(points,max(1,len(points)//128)):
  diff=chunk[:,None,:]-a[None,:,:];t=np.clip((diff*d).sum(2)/np.maximum(dd,1e-20),0,1)
  out.extend(np.sqrt(((diff-t[:,:,None]*d)**2).sum(2).min(1)).tolist())
 return float(np.mean(out)),float(np.max(out))
records=[]
for sid,r in selected.items():
 label,n,minimum,blob=db.execute('SELECT label,length,min_frames,features FROM samples WHERE id=? AND split=?',(sid,r['split'])).fetchone();assert label==r['label']
 f=np.frombuffer(blob,dtype='<f4').reshape(n,10);assert np.array_equal(f,curve_features(r['strokes'])),sid
 xy=np.concatenate(r['strokes'])[:,:2];span=np.ptp(xy,axis=0);scale=span[1] if span[1]>1e-6 else max(span[0],1.);origin=np.array([xy[0,0],xy[:,1].min()])
 strokes=[((np.array(s)[:,:2]-origin)/scale).tolist() for s in r['strokes']];curves=decode(f,strokes[0][0]);mean,maximum=distances(np.concatenate(strokes),curves)
 allxy=np.concatenate([np.concatenate(strokes),np.concatenate([c['cp'] for c in curves])]);lo=allxy.min(0);hi=allxy.max(0)
 records.append(dict(id=sid,split=r['split'],label=label,prediction=r.get('prediction'),recognition=r['recognition'],strokes=strokes,curves=curves,bounds=[*lo.tolist(),*hi.tolist()],points=len(xy),segments=n,frames=n*2,eligible=minimum<=n*2<=512,mean_error=mean,max_error=maximum,cache_equal=True))
tolerances=(.005,.01,.02,.05)
for idx,record in enumerate(records):
 r=selected[record['id']];minimum=db.execute('SELECT min_frames FROM samples WHERE id=? AND split=?',(record['id'],record['split'])).fetchone()[0]
 record['variants']={}
 for tol in tolerances:
  if tol==.01:
   variant={k:record[k] for k in ('curves','segments','frames','eligible','mean_error','max_error')}
  else:
   f=curve_features(r['strokes'],tolerance=tol);curves=decode(f,record['strokes'][0][0]);mean,maximum=distances(np.concatenate(record['strokes']),curves)
   variant=dict(curves=curves,segments=len(f),frames=2*len(f),eligible=minimum<=2*len(f)<=512,mean_error=mean,max_error=maximum)
  record['variants'][str(tol)]=variant
 # Fixed common viewport across all settings: switching compression does not rescale ink.
 coords=np.concatenate([np.concatenate(record['strokes'])]+[np.concatenate([c['cp'] for c in v['curves']]) for v in record['variants'].values()])
 record['bounds']=[*coords.min(0).tolist(),*coords.max(0).tolist()]
 if idx%25==0:print('compression variants',idx+1,'/',len(records),flush=True)
(OUT/'index.html').write_text(Path('viewer_templates/preprocess.html').read_text())
(OUT/'data.js').write_text('window.SAMPLES='+json.dumps(records,ensure_ascii=False,separators=(',',':'))+';',encoding='utf8')
(OUT/'manifest.json').write_text(json.dumps(dict(samples=len(records),tolerances=list(tolerances),baseline_tolerance=.01,seed=20260919,all_cache_features_exactly_reproduced=True,source='data/ctc-full/features.sqlite',geometry='Inverse decode cached float32 vectors; original start anchors translation',metric='One-way approximate raw-point distance to sampled pen-down curves; units normalized formula height, width fallback for flat ink',prediction_checkpoint=json.loads(Path('runs/token-error-analysis/summary.json').read_text())['step']),ensure_ascii=False,indent=2))
print('Created',len(records),'samples; every cache feature verified',flush=True)
