import json,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
r=json.loads(Path('runs/correct-evidence.json').read_text())
chosen=[r['correct'][i] for i in [0,2,4,6,10,11]]
wanted={e['id']:e for e in chosen};ns='{http://www.w3.org/2003/InkML}'
with tarfile.open('data/mathwriting-2024.tgz','r|gz') as tar:
 for e in tar:
  if Path(e.name).parent.name!='valid' or Path(e.name).stem not in wanted:continue
  root=ET.fromstring(tar.extractfile(e).read());item=wanted[Path(e.name).stem]
  item['strokes']=[[[float(v) for v in point.split()] for point in trace.text.split(',')] for trace in root.findall(ns+'trace')]
  meta={a.attrib['type']:a.text for a in root.findall(ns+'annotation')}
  assert meta['normalizedLabel']==item['truth']
  if all('strokes' in v for v in chosen):break
im=Image.new('RGB',(1400,1080),'white');draw=ImageDraw.Draw(im)
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',18)
draw.text((30,20),f'Validation checkpoint {r["step"]}: {r["correct_count"]}/{r["samples"]} exact matches',font=font,fill='black')
for i,e in enumerate(chosen):
 x0=30+(i%2)*700;y0=85+(i//2)*325
 draw.text((x0,y0),'ID: '+e['id'],font=small,fill='#555555')
 pts=[p for st in e['strokes'] for p in st];xmin=min(p[0] for p in pts);ymin=min(p[1] for p in pts)
 dx=max(p[0] for p in pts)-xmin;dy=max(p[1] for p in pts)-ymin
 scale=min(600/max(dx,1),170/max(dy,1))
 for st in e['strokes']:
  pp=[(x0+20+(p[0]-xmin)*scale,y0+35+(p[1]-ymin)*scale) for p in st]
  if len(pp)>1:draw.line(pp,fill='#172b4d',width=3)
  else:draw.ellipse((pp[0][0]-2,pp[0][1]-2,pp[0][0]+2,pp[0][1]+2),fill='#172b4d')
 draw.text((x0,y0+225),'Reference:  '+e['truth'],font=small,fill='black')
 draw.text((x0,y0+255),'Prediction: '+e['prediction'],font=small,fill='#087443')
im.save('runs/correct-evidence.png')
