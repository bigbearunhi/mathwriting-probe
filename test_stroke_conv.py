import unittest
import torch
from stroke_conv import StrokeConvCTC

class StrokeConvTests(unittest.TestCase):
    def test_no_cross_stroke_or_padding_leakage(self):
        torch.manual_seed(2)
        model=StrokeConvCTC(9,dim=16,layers=1,heads=4,ff=32,dropout=0).eval()
        x=torch.randn(1,7,10);x[:,:,9]=torch.tensor([1,1,0,1,1,0,0]);lengths=torch.tensor([5])
        a=model.local_features(x,lengths)
        changed=x.clone();changed[:,3:5,:9]+=100;changed[:,5:,:9]=500
        b=model.local_features(changed,lengths)
        torch.testing.assert_close(a[:,:3],b[:,:3]);self.assertFalse(torch.allclose(a[:,3:5],b[:,3:5]))
        torch.testing.assert_close(a[:,2],model.input(x[:,2]))
        self.assertTrue(torch.all(a[:,5:]==0))
    def test_same_stroke_context_and_ctc_backward(self):
        torch.manual_seed(3);m=StrokeConvCTC(9,dim=16,layers=1,heads=4,ff=32,dropout=0)
        raw=torch.randn(2,5,10);raw[:,:,9]=1;raw[:,2,9]=0;raw[1,3:,9]=0
        lengths=torch.tensor([5,3]);a=m.local_features(raw,lengths);b=raw.clone();b[:,0,:9]+=1
        self.assertFalse(torch.allclose(a[:,1],m.local_features(b,lengths)[:,1]))
        repeated=raw.repeat_interleave(2,dim=1);logits=m(repeated,lengths*2)
        self.assertEqual(tuple(logits.shape),(2,10,9))
        loss=torch.nn.functional.ctc_loss(logits.log_softmax(-1).transpose(0,1),torch.tensor([1,2,3,4]),lengths*2,torch.tensor([2,2]),blank=0)
        loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(m.conv1.weight.grad).all())

if __name__=='__main__':unittest.main()
