import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchsummary import summary

class MultiScaleAttentionRouter(nn.Module):

    def __init__(self, in_channels, num_experts, hidden_dim=256, reduction=8):
        super().__init__()
       
        self.conv_reduce = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
        
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim, hidden_dim // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_dim // reduction, hidden_dim, bias=False),
            nn.Sigmoid()
        )
        
        self.pool_scales = [1, 2, 4]  
        self.multi_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(scale) for scale in self.pool_scales
        ])
        
        total_dim = hidden_dim * sum(s * s for s in self.pool_scales)
        self.fc = nn.Linear(total_dim, num_experts)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.conv_reduce(x)               # (B, hidden_dim, H, W)
        
        # channel attention
        attn = self.channel_attention(x)       # (B, hidden_dim)
        attn = attn.unsqueeze(-1).unsqueeze(-1) # (B, hidden_dim, 1, 1)
        x_weighted = x * attn               
        
        # multi-scal pooling
        multi_features = []
        for pool in self.multi_pool:
            pooled = pool(x_weighted)           # (B, hidden_dim, s, s)
            multi_features.append(pooled.flatten(1))  # (B, hidden_dim * s * s)
        
        # concat
        concat = torch.cat(multi_features, dim=1)  # (B, total_dim)
        
        # final logits
        logits = self.fc(concat)                # (B, num_experts)
        return logits
    
class FeatureMixerLayer(nn.Module):
    def __init__(self, in_dim, mlp_ratio=1):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, int(in_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(in_dim * mlp_ratio), in_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)

class MixVPR(nn.Module):
    def __init__(self,
                 in_channels=1024,
                 in_h=20,         
                 in_w=20,        
                 out_channels=512,
                 mix_depth=4,
                 mlp_ratio=1,
                 out_rows=4,
                 ) -> None:
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio

        hw = in_h * in_w
        self.mix = nn.Sequential(*[
            FeatureMixerLayer(in_dim=hw, mlp_ratio=mlp_ratio) 
            for _ in range(self.mix_depth)
        ])
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x):
        # (in_h, in_w)
        if x.shape[2] != self.in_h or x.shape[3] != self.in_w:
            x = F.adaptive_avg_pool2d(x, (self.in_h, self.in_w))

        x = x.flatten(2)                # (B, C, H*W)
        x = self.mix(x)                  # Feature-Mixer
        x = x.permute(0, 2, 1)           # (B, H*W, C)
        x = self.channel_proj(x)          # (B, H*W, out_channels)
        x = x.permute(0, 2, 1)           # (B, out_channels, H*W)
        x = self.row_proj(x)              # (B, out_channels, out_rows)
        x = x.flatten(1)                  # (B, out_channels * out_rows)
        x = F.normalize(x, p=2, dim=-1)
        return x


class MoEMixVPR(nn.Module):
    def __init__(self, in_channels, experts_cfgs, top_k=2):
        super().__init__()
        self.experts = nn.ModuleList([MixVPR(in_channels=in_channels, **cfg) for cfg in experts_cfgs])

        self.router = MultiScaleAttentionRouter(
            in_channels=in_channels,
            num_experts=len(experts_cfgs),
            hidden_dim=256,       
            reduction=8
        )
        self.top_k = top_k
        
    def forward(self, feat_map):

        # B = feat_map.size(0)                           
        # logits = torch.randn(B, 4, device=feat_map.device)   # (B, E)
        logits = self.router(feat_map)          # (B, E)
        probs = torch.softmax(logits, dim=-1)   # (B, E)

        if self.top_k == 1:
            # hard routing
            idx = probs.argmax(dim=-1)          # (B,)
            outs = []
            for b in range(feat_map.size(0)):
        
                expert_out = self.experts[idx[b]](feat_map[b:b+1])  # (1, D)
                outs.append(expert_out)
            out = torch.cat(outs, dim=0)        # (B, D)
            return out, probs
        else:
            # soft mixture
            all_outs = torch.stack([e(feat_map) for e in self.experts], dim=1)  # (B, E, D)
            out = (all_outs * probs.unsqueeze(-1)).sum(dim=1)                   # (B, D)
            # out = F.normalize(out, p=2, dim=-1)
            return out, probs
# -------------------------------------------------------------------------------
def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Trainable parameters: {params / 1e6:.3}M')

