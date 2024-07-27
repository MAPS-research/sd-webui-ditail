import torch as th
import torch.nn as nn

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
from ldm.modules.diffusionmodules.openaimodel import TimestepEmbedSequential, Downsample, Upsample, ResBlock, AttentionBlock, TimestepBlock, SpatialTransformer, UNetModel
from ldm.util import exists
from .replace_attention import SpatialTransformerWithInjection


    

class ResBlockWithInjection(ResBlock):
    def __init__(
        self, 
        channels, 
        emb_channels, 
        dropout, 
        out_channels=None, 
        use_conv=False, 
        use_scale_shift_norm=False, 
        dims=2, 
        use_checkpoint=False, 
        up=False, 
        down=False
    ):
        super().__init__(
            channels=channels, 
            emb_channels=emb_channels, 
            dropout=dropout, 
            out_channels=out_channels, 
            use_conv=use_conv, 
            use_scale_shift_norm=use_scale_shift_norm, 
            dims=dims, 
            use_checkpoint=use_checkpoint, 
            up=up, 
            down=down
            )
        self.in_layers_features = None
        self.out_layers_features = None
        print("!!!! ResBlockWithInjection initialized")

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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, emb, context, self_attn_q_injected=None, self_attn_k_injected=None, out_layers_injected=None):
        # print('!!!! TimestepEmbedSequentialWithInjection forward called')
        # print('!!!! check layers of TimestepEmbedSequentialWithInjection:', self)
        for layer in self:
            if isinstance(layer, ResBlockWithInjection):
                x = layer(x, emb, out_layers_injected=out_layers_injected)
            elif isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            elif isinstance(layer, SpatialTransformerWithInjection):
                x = layer(x, context, self_attn_q_injected, self_attn_k_injected)
            else:
                print('!!!! unknown layer type called in TimestepEmbedSequentialWithInjection:', layer )
                x = layer(x)
        self.stored_output = x
        return x
    

class UNetModelWithInjection(UNetModel):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        use_bf16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        adm_in_channels=None,
    ):
        super().__init__(
            image_size=image_size,
            in_channels=in_channels,
            model_channels=model_channels,
            out_channels=out_channels,
            num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions,
            dropout=dropout,
            channel_mult=channel_mult,
            conv_resample=conv_resample,
            dims=dims,
            num_classes=num_classes,
            use_checkpoint=use_checkpoint,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
            num_heads=num_heads,
            num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample,
            use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown,
            use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer,
            transformer_depth=transformer_depth,
            context_dim=context_dim,
            n_embed=n_embed,
            legacy=legacy,
            disable_self_attentions=disable_self_attentions,
            num_attention_blocks=num_attention_blocks,
            disable_middle_self_attn=disable_middle_self_attn,
            use_linear_in_transformer=use_linear_in_transformer,
            adm_in_channels=adm_in_channels,
        )

        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.dtype = th.bfloat16 if use_bf16 else self.dtype
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            elif self.num_classes == "sequential":
                assert adm_in_channels is not None
                self.label_emb = nn.Sequential(
                    nn.Sequential(
                        linear(adm_in_channels, time_embed_dim),
                        nn.SiLU(),
                        linear(time_embed_dim, time_embed_dim),
                    )
                )
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequentialWithInjection(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlockWithInjection(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerWithInjection(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequentialWithInjection(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequentialWithInjection(
                        ResBlockWithInjection(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            #num_heads = 1
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequentialWithInjection(
            ResBlockWithInjection(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformerWithInjection(  # always uses a self-attn
                            ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                            disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                            use_checkpoint=use_checkpoint
                        ),
            ResBlockWithInjection(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlockWithInjection(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        #num_heads = 1
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or i < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads_upsample,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformerWithInjection(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                if level and i == self.num_res_blocks[level]:
                    out_ch = ch
                    layers.append(
                        ResBlockWithInjection(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequentialWithInjection(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
        )
        if self.predict_codebook_ids:
            self.id_predictor = nn.Sequential(
            normalization(ch),
            conv_nd(dims, model_channels, n_embed, 1),
            #nn.LogSoftmax(dim=1)  # change to cross_entropy and produce non-normalized logits
        )

    # replace forward function
    def forward(self, x, timesteps=None, context=None, y=None, injected_features=None, **kwargs):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param context: conditioning plugged in via crossattn
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """

        # try:
        #     injected_features = temperal_load_target_features()
        #     print('@@@@ temperal loading features succeeded')
        # except Exception as e:
        #     print('@@@@ temperal loading features failed', e)


        # print('!!!! UNetModel forward with injection called')
        # # print('!!!! kwargs:', kwargs)
        # print('!!!! injected_features:', injected_features[0].keys())
        # # print('!!!! other args:', x.shape, timesteps, context, y)
        # # print('!!!! check self type in UNetModel:', self, type(self))

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
                print('!!!! self_attn_q_injected:', self_attn_q_injected.shape)
            
            if injected_features is not None and k_feature_key in injected_features:
                self_attn_k_injected = injected_features[k_feature_key]
                print('!!!! self_attn_k_injected:', self_attn_k_injected.shape)
            
            if injected_features is not None and out_layers_feature_key in injected_features:
                out_layers_injected = injected_features[out_layers_feature_key]
                print('!!!! out_layers_injected:', out_layers_injected.shape)

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

def apply_openaimodel_replacement(injected_features=None):
    
    ldm.modules.diffusionmodules.openaimodel.ResBlock = ResBlockWithInjection
    ldm.modules.diffusionmodules.openaimodel.TimestepEmbedSequential = TimestepEmbedSequentialWithInjection
    # ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = partial(forward_with_injection, injected_features=injected_features)
    ldm.modules.diffusionmodules.openaimodel.UNetModel = UNetModelWithInjection
    print('!!!! openaimodel replaced')
    return True


def temperal_load_target_features(latent_save_path="./extensions/sd-webui-ditail/features"):

    import torch
    from tqdm import tqdm
    import os
    self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
    out_layers_output_block_indices = [4]
    output_block_self_attn_map_injection_thresholds = [20 // 2] * len(self_attn_output_block_indices)
    feature_injection_thresholds = [int(20 * 1.0)] # TODO: should be one of the arguments, change later
    target_features = []

    T = 999
    c = T // 20

    iterator = tqdm(range(0, T, c), desc="Loading target features", total = 20)
    
    for i, t in enumerate(iterator):
        current_features = {}
        for (output_block_idx, output_block_self_attn_map_injection_threshold) in zip(self_attn_output_block_indices, output_block_self_attn_map_injection_thresholds):
             if i <= int(output_block_self_attn_map_injection_threshold):
                output_q = torch.load(os.path.join(latent_save_path, f"output_block_{output_block_idx}_self_attn_q_time_{t}.pt"))
                output_k = torch.load(os.path.join(latent_save_path, f"output_block_{output_block_idx}_self_attn_k_time_{t}.pt"))
                current_features[f"output_block_{output_block_idx}_self_attn_q"] = output_q
                current_features[f"output_block_{output_block_idx}_self_attn_k"] = output_k
            
        for (output_block_idx, feature_injection_threshold) in zip(out_layers_output_block_indices, feature_injection_thresholds):
            if i <= int(feature_injection_threshold):
                output = torch.load(os.path.join(latent_save_path, f"output_block_{output_block_idx}_out_layers_features_time_{t}.pt"))
                current_features[f"output_block_{output_block_idx}_out_layers"] = output

        target_features.append(current_features)
    return target_features