import math
import pdb

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

import ldm
from ldm.modules.diffusionmodules.util import checkpoint
from ldm.modules.attention import (
    exists, default, 
    CrossAttention, MemoryEfficientCrossAttention, BasicTransformerBlock, SpatialTransformer, 
    FeedForward, XFORMERS_IS_AVAILBLE, _ATTN_PRECISION
    )
 

class CrossAttentionWithInjection(CrossAttention):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__(query_dim=query_dim, context_dim=context_dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.attn = None
        self.q = None
        self.k = None
        self.v = None


    def forward(self, x, context=None, mask=None, q_injected=None, k_injected=None):
        print('!!!! CrossAttentionWithInjection forward called')
        self.attn = None
        h = self.heads
        b = x.shape[0] // 2
        if q_injected is None:
            print('!!!! running to_q in CrossAttentionWithInjection')
            q = self.to_q(x)
            print('!!!! q shape', q.shape)
            q = rearrange(q, 'b n (h d) -> (b h) n d', h=h)
        else:
            q_uncond, q_cond = q_injected.chunk(2)
            q = torch.cat([q_uncond]*b + [q_cond]*b, dim=0)
        context = default(context, x)

        if k_injected is None:
            k = self.to_k(context)
            k = rearrange(k, 'b m (h d) -> (b h) m d', h=h)
        else:
            k_uncond, k_cond = k_injected
            k = torch.cat([k_uncond]*b + [k_cond]*b, dim=0)
        
        v = self.to_v(context)
        v = rearrange(v, 'b m (h d) -> (b h) m d', h=h)

        self.q = q
        self.k = k
        self.v = v

        # TODO: check whether this will cause error
        # force cast to fp32 to avoid overflowing
        if _ATTN_PRECISION =="fp32":
            with torch.autocast(enabled=False, device_type = 'cuda'):
                q, k = q.float(), k.float()
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        else:
            sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
        
        del q, k

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)
        
        self.attn = attn

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlockWithInjection(nn.Module):
    ATTENTION_MODES = {
        "softmax": CrossAttentionWithInjection,  # vanilla attention
        "softmax-xformers": MemoryEfficientCrossAttention
    }

    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
                 disable_self_attn=False):
        # super().__init__(dim, n_heads, d_head, 
        #                  dropout=dropout, context_dim=context_dim, gated_ff=gated_ff, 
        #                  checkpoint=checkpoint,
        #                  disable_self_attn=disable_self_attn)

        super().__init__()
        attn_mode = "softmax-xformers" if XFORMERS_IS_AVAILBLE else "softmax"
        assert attn_mode in self.ATTENTION_MODES
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.disable_self_attn = disable_self_attn
        self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                              context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                              heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None, self_attn_q_injected=None, self_attn_k_injected=None):
        print('!!!! BasicTransformerBlockWithInjection forward called')
        return checkpoint(
            self._forward, 
            (x, context, self_attn_q_injected, self_attn_k_injected),
            self.parameters(),
            self.checkpoint
            )

    def _forward(self, x, context=None, self_attn_q_injected=None, self_attn_k_injected=None):
        print('!!!! BasicTransformerBlockWithInjection _forward called')
        x = self.attn1(self.norm1(x),
                       q_injected=self_attn_q_injected,
                       k_injected=self_attn_k_injected,
                       context=context if self.disable_self_attn else None) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformerWithInjection(SpatialTransformer):
    def __init__(
        self, 
        in_channels, 
        n_heads,
        d_head,depth=1, 
        dropout=0., 
        context_dim=None,
        disable_self_attn=False,
        use_linear=False,
        use_checkpoint=False
    ):
        super().__init__(
            in_channels=in_channels, 
            n_heads=n_heads,
            d_head=d_head,
            depth=depth, 
            dropout=dropout, 
            context_dim=context_dim,
            disable_self_attn=disable_self_attn,
            use_linear=use_linear,
            use_checkpoint=use_checkpoint
            )
        print("!!!! SpatialTransformerWithInjection initialized")
    
    def forward(self, x, context=None, self_attn_q_injected=None, self_attn_k_injected=None):
        # pdb.set_trace()
        print('!!!! SpatialTransformerWithInjection forward called')
        # note: if no context is given, cross-attention defaults to self-attention
        if not isinstance(context, list):
            context = [context]
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        if not self.use_linear:
            x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c').contiguous()
        if self.use_linear:
            x = self.proj_in(x)

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, 
                      context=context[i],
                      self_attn_q_injected=self_attn_q_injected,
                      self_attn_k_injected=self_attn_k_injected,
                      )

        if self.use_linear:
            x = self.proj_out(x)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w).contiguous()
        if not self.use_linear:
            x = self.proj_out(x)
        return x + x_in



def apply_attention_replacement():
    print('!!!! Applying replacement')
    ldm.modules.attention.CrossAttention = CrossAttentionWithInjection
    ldm.modules.attention.BasicTransformerBlock = BasicTransformerBlockWithInjection
    ldm.modules.attention.SpatialTransformer = SpatialTransformerWithInjection