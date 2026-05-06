import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# found at https://github.com/lingxitong/MIL_BASELINE


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m,nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class AMDLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, agent_num=512, tem=0, pool=False, thresh=None, thresh_tem='classical', kaiming_init=False):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = AMD_Layer(
            dim = dim,
            agent_num=agent_num,
            heads = 8,          
        )

    def forward(self, x, return_WSI_attn = False):
        forward_return = self.attn(self.norm(x), return_WSI_attn)
        x = x + forward_return['amd_out']
        new_forward_return = {}
        new_forward_return['amd_out'] = x
        if return_WSI_attn:
            new_forward_return['WSI_attn'] = forward_return['WSI_attn']

        return new_forward_return


'''
@article{shao2021transmil,
  title={Transmil: Transformer based correlated multiple instance learning for whole slide image classification},
  author={Shao, Zhuchen and Bian, Hao and Chen, Yang and Wang, Yifeng and Zhang, Jian and Ji, Xiangyang and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2136--2147},
  year={2021}
}
'''
class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x

class AMD_MIL(nn.Module):
    def __init__(self, num_classes,in_dim,embed_dim,dropout,act,agent_num=256):
        super(AMD_MIL, self).__init__()
        self.pos_layer = PPEG(dim=embed_dim) # PPEG from TransMIL

        self._fc1 = [nn.Linear(in_dim, embed_dim)]
        self._fc1 += [act]
        if dropout:
            self._fc1 += [nn.Dropout(dropout)]
        self._fc1 = nn.Sequential(*self._fc1)
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

        nn.init.normal_(self.cls_token, std=1e-6)
        self.num_classes = num_classes
        self.amdlayer1 = AMDLayer(dim=embed_dim, agent_num = agent_num)
        self.amdlayer2 = AMDLayer(dim=embed_dim, agent_num = agent_num)
        self.norm = nn.LayerNorm(embed_dim)         
        self._fc2 = nn.Linear(embed_dim, self.num_classes)
        self.apply(initialize_weights)
        self.lr = 2e-4

    def forward(self, x, return_WSI_attn = False, return_WSI_feature = False):
        forward_return = {}
        B = x.shape[0]
        N = x.shape[1]
        h = self._fc1(x) 
        # fit for PPEG
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]],dim = 1) 
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(h.device)
        h = torch.cat((cls_tokens, h), dim=1)
        h = self.amdlayer1(h)['amd_out']
        h = self.pos_layer(h, _H, _W) 
        amd_return = self.amdlayer2(h, return_WSI_attn)
        h = amd_return['amd_out']
        h = self.norm(h)[:,:1] 
        h = h.squeeze(1)
        logits = F.sigmoid(self._fc2(h))
        forward_return['logits'] = logits.squeeze()
        if return_WSI_feature:
            forward_return['WSI_feature'] = h
        if return_WSI_attn:
            forward_return['A'] = amd_return['WSI_attn'][:,:,0,1:N+1].mean(1).transpose(0,1)
        return forward_return

class AMD_Layer(nn.Module):
    def __init__(
        self,
        dim,
        agent_num=256,
        heads = 8,
    ):
        super().__init__()
        self.dim_head = dim//heads
        self.agent_num = agent_num
        self.denoise = nn.Linear(self.dim_head,self.dim_head)
        self.mask = nn.Linear(self.dim_head,self.dim_head)
        self.get_thresh = nn.Linear(dim,1)
        self.heads = heads
        self.scale = self.dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias = False)
        self.agent = nn.Parameter(torch.randn(heads, agent_num, self.dim_head))

    def forward(self, x , return_WSI_attn = False):
        forward_return = {}
        b, _, _, h = *x.shape, self.heads
        # obtain the qkv matrix
        q, k, v = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))
        agent = self.agent.unsqueeze(0).expand(b,-1,-1,-1)
        # Perform agent calculations
        q = torch.matmul(q,agent.transpose(-1,-2))
        k = torch.matmul(agent,k.transpose(-1,-2))
        softmax = nn.Softmax(dim=-1)
        q *= self.scale
        q = softmax(q)
        k = softmax(k)
        kv = torch.matmul(k,v) 
        kv_c = kv.reshape(b,self.agent_num,-1)
        thresh = self.get_thresh(kv_c).squeeze().mean()
        thresh = F.sigmoid(thresh)
        # Perform mask and denoise operations
        denoise = self.denoise(kv)
        denoise = torch.sigmoid(denoise)
        mask = self.mask(kv)
        mask = torch.sigmoid(mask)
        mask = torch.where(mask > thresh, torch.ones_like(mask), torch.zeros_like(mask))
        kv = kv * mask + denoise
        # Obtain weighted features
        kv = softmax(kv)
        out = torch.matmul(q,kv)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        forward_return['amd_out'] = out
        if return_WSI_attn:
            WSI_attn = torch.matmul(q,k)
            forward_return['WSI_attn'] = WSI_attn 
        return forward_return
    
    
if __name__ == "__main__":
    agent = AMD_Layer(dim=512).cuda()
    x = torch.randn(1, 1000, 512).cuda()
    out = agent(x)
    print(out.shape)


