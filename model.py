import torch 
import math
import torch.nn as nn
import torch.nn.functional as F;
class ScaleAttention(nn.Module):
    def __init__(self,dff):
        super(ScaleAttention,self).__init__()
        self.dff=dff
        
    def forward(self,query,key,value,mask=None):
        atten_score=torch.matmul(query,key.transpose(-1,-2))
        atten_score=atten_score/math.sqrt(self.dff)
        if mask is not None:            
            mask=mask.to(atten_score.device)
            atten_score=atten_score.masked_fill(mask,float('-inf'))
        atten_score=F.softmax(atten_score,dim=-1)
        atten_score=torch.nan_to_num(atten_score,nan=0.0)
        atten_score=torch.matmul(atten_score,value)
        return atten_score


class MultiHeadAttention(nn.Module):
    def __init__(self,dmodel,n_head,dff,max_seq,droprate=0.1):
        super(MultiHeadAttention,self).__init__()
        self.dmodel=dmodel
        self.dff=dff
        self.n_head=n_head
        self.drop=nn.Dropout(droprate)
        self.wquery=nn.Linear(self.dmodel,self.dmodel)
        self.wkey=nn.Linear(self.dmodel,self.dmodel)
        self.wvalue=nn.Linear(self.dmodel,self.dmodel)
        self.wo=nn.Linear(self.dmodel,self.dmodel)
        self.attention=ScaleAttention(self.dff)
        inf_fre=1.0/(10000**(torch.arange(0,self.dff,2).float()/self.dff))
        position=torch.arange(max_seq).float()
        angle=torch.einsum("i,j->ij",position,inf_fre)
        self.register_buffer("sine",torch.sin(angle))
        self.register_buffer("cose",torch.cos(angle))
        
       
        
        

    def Rotate_half(self,x):
        x_even=x[...,::2]
        x_odd=x[...,1::2]
        return torch.stack([-x_odd,x_even],dim=-1).flatten(-2)

    def apply_rope(self,x,start_pos=0):
        seq=x.size(-2)
        sin=self.sine[:start_pos+seq].unsqueeze(0).unsqueeze(0)
        cos=self.cose[:start_pos+seq].unsqueeze(0).unsqueeze(0)
        cos=cos.repeat_interleave(2,dim=-1)
        sin=sin.repeat_interleave(2,dim=-1)
        return x*cos+self.Rotate_half(x)*sin
        
    
    
    def split_head(self,x):
        return x.reshape(x.size(0),x.size(1), self.n_head,self.dff)
    def group_head(self,x):
        return x.reshape(x.size(0),x.size(1), self.n_head*self.dff)

    def forward(self,x,start_pos=0,mask=None):
        # x=x.permute(0, 2, 1)
        
        Q=self.split_head(self.wquery(x))
        v=self.split_head(self.wvalue(x))
        k=self.split_head(self.wkey(x))
        Q=self.apply_rope(Q.permute(0,2,1,3).contiguous(),start_pos=0)
        v=v.permute(0,2,1,3).contiguous()
        k=self.apply_rope(k.permute(0,2,1,3).contiguous(),start_pos=0)
        
        at=self.attention(Q,k,v,mask=mask)
        at=at.permute(0,2,1,3)
        at=self.group_head(at)
        at=self.drop(at)
        at=self.wo(at)
        return at
        
        
        

class Gelu(nn.Module):
    def __init__(self):
        super(Gelu,self).__init__()
        
    def forward(self,x):
        return 0.5* x*(1+torch.tanh((torch.sqrt(torch.tensor(2.0/torch.pi))*(x+0.044715*torch.pow(x,3)))))
    
class FeedForward(nn.Module):
    def __init__(self,dmodel,dff):
        super(FeedForward,self).__init__()
        self.dmodel=dmodel
        self.dff=dff
        self.layer=nn.Sequential(
            nn.Linear(self.dmodel,self.dff),
            Gelu(),
            nn.Linear(self.dff,self.dmodel)
        )
    def forward(self,x):
        return self.layer(x)
    
class Transformer(nn.Module):
    def __init__(self,dmodel,dff,n_head,f_dff,max_seq,droprate=0.1):
        super(Transformer,self).__init__()
        self.dmodel=dmodel
        self.dff=dff
        self.n_head=n_head
        self.f_dff=f_dff
        self.drop1=nn.Dropout(droprate)
        self.drop2=nn.Dropout(droprate)
        self.mha=MultiHeadAttention(dmodel=self.dmodel,dff=self.dff,n_head=self.n_head,droprate=droprate,max_seq=max_seq)
        self.ffn=FeedForward(dmodel=self.dmodel,dff=self.f_dff)
        self.lay1=nn.LayerNorm(self.dmodel)
        self.lay2=nn.LayerNorm(self.dmodel)

    def mask(self,xsize):
        return torch.triu(torch.ones(xsize,xsize),diagonal=1).bool()

    
    def forward(self,x,start_pos=0):
        mask1=self.mask(x.size(1))
        norm_x = self.lay1(x)
        x1=self.mha(norm_x,start_pos,mask=mask1)
        x=x+self.drop1(x1)
        norm=self.lay2(x)
        x1=self.ffn(norm)
        x=x+self.drop2(x1)
        return x
        
        
        
        
        
        
        
class GPR(nn.Module):
    def __init__(self,dmodel,dff,n_head,n_layer,f_dff,max_seq,vocab_size,droprate=0.1):
        super(GPR,self).__init__()
        self.dmodel=dmodel
        self.dff=dff
        self.n_head=n_head
        self.f_dff=f_dff
        self.n_layer=n_layer
        self.vocab_size=vocab_size
        self.norm=nn.LayerNorm(self.dmodel)
        self.transformer=nn.ModuleList([
            Transformer(dmodel=self.dmodel,dff=self.dff,n_head=self.n_head,f_dff=self.f_dff,droprate=droprate,max_seq=max_seq)
                      for _ in range(n_layer) ])
        self.embedd=nn.Embedding(self.vocab_size,self.dmodel)
        self.output=nn.Linear(self.dmodel,self.vocab_size, bias=False)
        self.output.weight = self.embedd.weight

    def forward(self, x,start_pos=0):
       
        x=self.embedd(x)
        for layer in self.transformer:
            x=layer(x,start_pos)
        x=self.norm(x)
        x=self.output(x)
        return x
        
        
        
