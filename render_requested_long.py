import json,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
p=Path('checkpoint_015000'); r=json.loads(Path('runs/long-formula-results.json').read_text()); e=next(x for x in r['records'] if x['id']=='e6eec86f43b20780')
ns='{http://www.w3.org/2003/InkML}'
with tarfile.open('data/mathwriting-2024.tgz','r|gz') as t:
 for member in t:
  if Path(member.name).parent.name=='synthetic' and Path(member.name).stem==e['id']:
   root=ET.fromstring(t.extractfile(member).read());break
 else:raise RuntimeError('not found')
meta={a.attrib['type']:a.text for a in root.findall(ns+'annotation')};assert meta['normalizedLabel']==e['truth']
strokes=[[[float(v) for v in point.split()] for point in trace.text.split(',')] for trace in root.findall(ns+'trace')]
im=Image.new('RGB',(2200,850),'white');d=ImageDraw.Draw(im)
f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',24)
d.text((35,20),p.stem+' | ID: '+e['id'],font=f,fill='black')
pts=[v for s in strokes for v in s];xmin=min(v[0] for v in pts);ymin=min(v[1] for v in pts);dx=max(v[0] for v in pts)-xmin;dy=max(v[1] for v in pts)-ymin;scale=min(2080/max(dx,1),400/max(dy,1))
for s in strokes:
 pp=[(60+(v[0]-xmin)*scale,90+(v[1]-ymin)*scale) for v in s]
 if len(pp)>1:d.line(pp,fill='#172b4d',width=4)
 else:d.ellipse((pp[0][0]-2,pp[0][1]-2,pp[0][0]+2,pp[0][1]+2),fill='#172b4d')
import textwrap
for j,line in enumerate(textwrap.wrap('Reference:  '+e['truth'],width=140,break_long_words=True,break_on_hyphens=False)):
 d.text((35,540+j*32),line,font=f,fill='#087443')
for j,line in enumerate(textwrap.wrap('Prediction: '+e['prediction'],width=140,break_long_words=True,break_on_hyphens=False)):
 d.text((35,680+j*32),line,font=f,fill='#ba2727')
im.save('runs/long-formula-e6eec86f43b20780.png');Path('runs/long-formula-e6eec86f43b20780.json').write_text(json.dumps(dict(validation=str(p),**e),indent=2))
