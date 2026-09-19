"""Stroke-local, length-preserving convolution before curve repetition."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from ctc_train import CTCModel

class StrokeConvCTC(CTCModel):
    def __init__(self,vocab_size,**kwargs):
        super().__init__(vocab_size,**kwargs)
        self.conv1=nn.Conv1d(10,64,3,padding=1)
        self.conv2=nn.Conv1d(64,self.dim,3,padding=1)
    @staticmethod
    def masked_conv(x,layer,allowed):
        windows=F.pad(x.transpose(1,2),(1,1)).unfold(2,3,1).permute(0,2,3,1)
        windows=windows*allowed.unsqueeze(-1)
        return torch.einsum('blkc,ock->blo',windows,layer.weight)+layer.bias
    def local_features(self,x,lengths):
        valid=torch.arange(x.size(1),device=x.device)[None,:]<lengths.to(x.device)[:,None]
        down=(x[:,:,9]>.5)&valid
        stroke=(~down).long().cumsum(1)
        neighbors=F.pad(stroke,(1,1),value=-1).unfold(1,3,1)
        neighbor_down=F.pad(down,(1,1),value=False).unfold(1,3,1)
        allowed=(neighbors==stroke.unsqueeze(-1))&neighbor_down&down.unsqueeze(-1)
        h=F.silu(self.masked_conv(x,self.conv1,allowed))
        correction=self.masked_conv(h,self.conv2,allowed)
        # Pen-up connectors keep their original linear embedding.
        return (self.input(x)+correction*down.unsqueeze(-1))*valid.unsqueeze(-1)
    def forward(self,x,lengths):
        # Data uses repeat=2. Convolve unique curve segments, then repeat embeddings.
        x=self.local_features(x[:,::2],lengths//2).repeat_interleave(2,dim=1)
        pos=torch.arange(x.size(1),device=x.device,dtype=torch.float32)[:,None]
        freq=torch.exp(torch.arange(0,self.dim,2,device=x.device)*(-math.log(10000)/self.dim))
        pe=torch.zeros(x.size(1),self.dim,device=x.device)
        pe[:,0::2],pe[:,1::2]=(pos*freq).sin(),(pos*freq).cos()
        x=self.drop(x+pe.to(x.dtype))
        padding=torch.arange(x.size(1),device=x.device)[None,:]>=lengths.to(x.device)[:,None]
        for layer in self.layers:x=layer(x,padding)
        return self.output(self.final_norm(x))
