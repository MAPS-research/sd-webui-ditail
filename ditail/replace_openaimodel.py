import torch as th

from functools import partial

import ldm
from ldm.modules.diffusionmodules.util import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, ResBlock, TimestepBlock, SpatialTransformer

# replace ldm.modules.diffusionmodules.openaimodel.forward with this function
def forward_with_injection(self, x, timesteps=None, context=None, y=None, injected_features=None, **kwargs):
    """
    Apply the model to an input batch.
    :param x: an [N x C x ...] Tensor of inputs.
    :param timesteps: a 1-D batch of timesteps.
    :param context: conditioning plugged in via crossattn
    :param y: an [N] Tensor of labels, if class-conditional.
    :return: an [N x C x ...] Tensor of outputs.
    """

    print('!!!! UNetModel forward with injection called')
    print('!!!! kwargs:', kwargs)
    # print('!!!! other args:', x.shape, timesteps, context, y)

    assert (y is not None) == (
        self.num_classes is not None
    ), "must specify y if and only if the model is class-conditional"
    hs = []
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    emb = self.time_embed(t_emb)

    if self.num_classes is not None:
        assert y.shape[0] == x.shape[0]
        emb = emb + self.label_emb(y)

    h = x.type(self.dtype)
    for module in self.input_blocks:
        h = module(h, emb, context)
        hs.append(h)
    h = self.middle_block(h, emb, context)

    module_i = 0
    for module in self.output_blocks:
        self_attn_q_injected = None
        self_attn_k_injected = None
        out_layers_injected = None
        q_feature_key = f'output_block_{module_i}_self_attn_q'
        k_feature_key = f'output_block_{module_i}_self_attn_k'
        out_layers_feature_key = f'output_block_{module_i}_out_layers'

        if injected_features is not None and q_feature_key in injected_features:
            self_attn_q_injected = injected_features[q_feature_key]
        
        if injected_features is not None and k_feature_key in injected_features:
            self_attn_k_injected = injected_features[k_feature_key]
        
        if injected_features is not None and out_layers_feature_key in injected_features:
            out_layers_injected = injected_features[out_layers_feature_key]


        h = th.cat([h, hs.pop()], dim=1)
        h = module(h, emb, context,
                   self_attn_q_injected=self_attn_q_injected,
                   self_attn_k_injected=self_attn_k_injected,
                   out_layers_injected=out_layers_injected
                   )
    
        module_i += 1

    h = h.type(x.dtype)
    if self.predict_codebook_ids:
        return self.id_predictor(h)
    else:
        return self.out(h)
    

class ResBlockWithInjection(ResBlock):
    def __init__(self, channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False):
        super.__init__(channels, emb_channels, dropout, out_channels=None, use_conv=False, use_scale_shift_norm=False, dims=2, use_checkpoint=False, up=False, down=False)
    
    def forward(self, x, emb, out_layers_injected=None):
        return checkpoint(
            self._forward, 
            (x, emb, out_layers_injected),
            self.parameters(),
            self.use_checkpoint
        )

    def _forward(self, x, emb, out_layers_injected=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
            self.in_layers_features = h
        
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            if out_layers_injected is not None:
                out_layers_injected_uncond, out_layers_injected_cond = out_layers_injected.chunk(2)
                b = x.shape[0] // 2
                h = th.cat([out_layers_injected_uncond]*b + [out_layers_injected_cond]*b)
            else:
                h = h + emb_out
                h = self.out_layers(h)
            self.out_layers_features = h
        return self.skip_connection(x) + h

class TimestepEmbedSequentialWithInjection(TimestepEmbedSequential):
    def forward(self, x, emb, context, self_attn_q_injected=None, self_attn_k_injected=None, out_layers_injected=None):
        print('!!!! TimestepEmbedSequentialWithInjection forward called')
        for layer in self:
            if isinstance(layer, ResBlock):
                x = layer(x, emb, out_layers_injected=out_layers_injected)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformer):
                x = layer(x, context, self_attn_q_injected, self_attn_k_injected)
            else:
                x = layer(x)
        self.stored_output = x
        return x

def replace_openaimodel(injected_features=None):
    
    ldm.modules.diffusionmodules.openaimodel.ResBlock = ResBlockWithInjection
    ldm.modules.diffusionmodules.openaimodel.TimestepEmbedSequential = TimestepEmbedSequentialWithInjection
    ldm.modules.diffusionmodules.openaimodel.forward = partial(forward_with_injection, injected_features=injected_features)
    print('!!!! openaimodel replaced')
    return True
