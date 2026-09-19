import random,tempfile,unittest
from pathlib import Path
import torch
from stroke_conv import StrokeConvCTC
from ctc_train import save
from resume_stroke_conv import restore
class ResumeTests(unittest.TestCase):
 def test_optimizer_rng_and_next_update(self):
  torch.manual_seed(42);torch.set_num_threads(1)
  kw=dict(dim=16,layers=1,heads=4,ff=32,dropout=.15)
  model=StrokeConvCTC(9,**kw);opt=torch.optim.Adam(model.parameters(),lr=1e-4);rng=random.Random(42)
  x=torch.randn(2,4,10).repeat_interleave(2,1);x[:,:,9]=1;lengths=torch.tensor([8,8])
  def step(m,o):
   o.zero_grad();loss=m(x,lengths).square().mean();loss.backward();o.step();return loss.detach()
  step(model,opt)
  with tempfile.TemporaryDirectory() as t:
   p=Path(t)/'s.pt';save(p,model,opt,7,dict(architecture='StrokeConvCTC',**kw),rng)
   expected_random=rng.random();expected=step(model,opt)
   m2=StrokeConvCTC(9,**kw);o2=torch.optim.Adam(m2.parameters(),lr=.1);r2=random.Random()
   state=restore(p,m2,o2,r2)
   self.assertEqual(state['step'],7);self.assertEqual(r2.random(),expected_random)
   torch.testing.assert_close(step(m2,o2),expected,rtol=0,atol=0)
   for a,b in zip(model.parameters(),m2.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
if __name__=='__main__':unittest.main()
