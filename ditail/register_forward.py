import os
_ATTN_PRECISION = os.environ.get("ATTN_PRECISION", "fp32")
import pdb

import torch as th
from inspect import isfunction
from torch import nn, einsum
from einops import rearrange, repeat

def exists(val):
    return val is not None

def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -th.finfo(t.dtype).max

def head_to_batch_dim(tensor, head_size, out_dim:int=3) -> th.Tensor:
    if tensor.ndim == 3:
        batch_size, seq_len, dim = tensor.shape
        extra_dim = 1
    else:
        batch_size, extra_dim, seq_len, dim = tensor.shape
    tensor = tensor.reshape(batch_size, seq_len * extra_dim, head_size, dim//head_size)
    tensor = tensor.permute(0, 2, 1, 3)
    if out_dim == 3:
        tensor = tensor.reshape(batch_size*head_size, seq_len*extra_dim, dim//head_size)

    return tensor

def register_time(model, t):
    for block in model.output_blocks:
        if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
            module = block[1].transformer_blocks[0].attn1
            setattr(module, 't', t)

    conv_modules = model.output_blocks[4][0]
    setattr(conv_modules, 't', t)


def register_attn_inj(model, injection_schedule=None):
    def sa_forward(self):
        # pdb.set_trace()
        to_out = self.to_out
        if type(to_out) is nn.modules.container.ModuleList:
            to_out = to_out[0]
        
        def forward(x, context=None, mask=None):
            # print('!!!! sa_forward called')
            # batch_size, seqence_length, dim = x.shape
            h = self.heads

            q = self.to_q(x)
            is_cross = context
            context = context if context is not None else x
            k = self.to_k(context)
            v = self.to_v(context)
            # if not is_cross and self.injection_schedule is not None and (
            #     self.t in self.injection_schedule or self.t == 1000):
            
            if not is_cross and self.injection_schedule is not None and self.t in self.injection_schedule:
                bs = int(q.shape[0] // 3)
                # inject pos chunk
                q[:bs] = q[bs:2*bs]
                k[:bs] = k[bs:2*bs]
                # inject neg chunk
                q[2*bs:] = q[bs:2*bs]
                k[2*bs:] = k[bs:2*bs]
                # print('**** attn injection done')

            # else:
            #     print('!!!! context shape', context.shape)

            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            # force cast to fp32 to avoid overflowing
            if _ATTN_PRECISION =="fp32":
                with th.autocast(enabled=False, device_type = 'cuda'):
                    q, k = q.float(), k.float()
                    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            else:
                sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
            
            del q, k

            if exists(mask):
                mask = rearrange(mask, 'b ... -> b (...)')
                max_neg_value = -th.finfo(sim.dtype).max
                mask = repeat(mask, 'b j -> (b h) () j', h=h)
                sim.masked_fill_(~mask, max_neg_value)
            
            sim = sim.softmax(dim=-1)
            out = einsum('b i j, b j d -> b i d', sim, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return self.to_out(out)
        return forward
    
    for block in model.output_blocks:
        if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
            module = block[1].transformer_blocks[0].attn1
            module._uninjected_forward = module.forward
            module.forward = sa_forward(module)
            setattr(module, 'injection_schedule', injection_schedule)


def unregister_attn_inj(model):
    for block in model.output_blocks:
        if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
            module = block[1].transformer_blocks[0].attn1
            if hasattr(module, '_uninjected_forward'):
                module.forward = module._uninjected_forward
                delattr(module, 'injection_schedule')
                delattr(module, '_uninjected_forward')


def register_conv_inj(model, injection_schedule):
    def conv_forward(self):
        def forward(x, emb):
            # print('!!!! conv_forward called')
            if self.updown:
                in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
                h = in_rest(x)
                h = self.h_upd(h)
                x = self.x_upd(x)
                h = in_conv(h)
            else:
                h = self.in_layers(x)
            emb_out = self.emb_layers(emb).type(h.dtype)
            while len(emb_out.shape) < len(h.shape):
                emb_out = emb_out[..., None]
            if self.use_scale_shift_norm:
                out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
                scale, shift = th.chunk(emb_out, 2, dim=1)
                h = out_norm(h) * (1 + scale) + shift
                h = out_rest(h)
            else:
                h = h + emb_out
                h = self.out_layers(h)

            # if self.injection_schedule is not None and (
            #     self.t in self.injection_schedule or self.t == 1000):
            if self.injection_schedule is not None and self.t in self.injection_schedule:
                bs = int(h.shape[0] // 3)
                h[:bs] = h[bs:2*bs]
                h[2*bs:] = h[bs:2*bs]
                # print('**** conv injection done')

            return self.skip_connection(x) + h
        return forward


    conv_modules = model.output_blocks[4][0]
    conv_modules._uninjected_forward = conv_modules._forward
    conv_modules._forward = conv_forward(conv_modules)
    setattr(conv_modules, 'injection_schedule', injection_schedule)

def unregister_conv_inj(model):

    conv_modules = model.output_blocks[4][0]
    if hasattr(conv_modules, '_uninjected_forward'):
        conv_modules._forward = conv_modules._uninjected_forward
        delattr(conv_modules, 'injection_schedule')
        delattr(conv_modules, '_uninjected_forward')