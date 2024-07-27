import os

import torch
import numpy as np

from tqdm import tqdm
from torch import autocast
from PIL import Image
from pytorch_lightning import seed_everything

from ditail.utils import create_path
from modules.shared import device
from ldm.models.diffusion.ddim import DDIMSampler

class DDIMInverse(DDIMSampler):
    def __init__(self, model):
        super().__init__(model)

    def encode_ddim(self, 
                    img, 
                    num_steps, 
                    conditioning,
                    save_feature_maps_callback=None,
        ):
        latents = {}
        print(f"Running DDIM inversion with {num_steps} steps")
        T = 1000
        c = T // num_steps
        iterator = tqdm(range(1, T, c), desc="DDIM inversion", total = num_steps)
        steps = list(range(1, T+c, c))

        # print('!! steps', steps)
        # print('!! iterator', [i for i in range(0, T, c)])

        for i, t in enumerate(iterator):
            img, _ = self.reverse_ddim(img, t, t_next=steps[i+1], c=conditioning) # TODO: i changed i+1 to i-1 here to avoid error
            latents[t] = img

            # print('!! check model', self.model, self.model.model.__dict__.keys())
            save_feature_maps_callback(self.model.model.diffusion_model, t)

        return latents, img

    @torch.no_grad()
    def reverse_ddim(self, 
                     x, 
                     t, 
                     t_next, 
                     c=None, 
                     quantize_denoised=False, #TODO: check if this is necessary,
                    #  unconditional_guidance_scale=1.,
                    #  unconditional_conditioning=None,
        ):
        b, *_, device = *x.shape, x.device

        t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        # if c is None:
        #     e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        # elif unconditional_conditioning is None or unconditional_guidance_scale==1.:
        #     e_t = self.model.apply_model(x, t_tensor, c)

        # else:
        #     x_in = torch.cat([x] * 2)
        #     t_in = torch.cat([t_tensor] * 2)
        #     c_in = torch.cat([unconditional_conditioning, c])
        #     e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        eps = self.model.apply_model(x, t_tensor, c)

        alphas = self.model.alphas_cumprod
        sigmas = self.model.sqrt_one_minus_alphas_cumprod
        
        # print('!! alphas', alphas, len(alphas))
        # print('!! sigmas', sigmas, len(sigmas))

        mus = torch.sqrt(alphas)

        # TODO: check out the 'next' and 'prev' timestep
        a_t = torch.full((b,1,1,1), alphas[t], device=device)
        if t_next > len(alphas): # TODO: a temporary fix for the last timestep
            t_next = len(alphas) - 1
        a_next = torch.full((b,1,1,1), alphas[t_next], device=device)
        mu_t = mus[t]

        mu_next = mus[t_next]
        sigma_t = torch.full((b,1,1,1), sigmas[t], device=device)
        sigma_next = torch.full((b,1,1,1), sigmas[t_next], device=device)

        pred_x0 = (x - sigma_t * eps) / mu_t
        x_next = a_next * pred_x0 + sigma_next * eps
        return x_next, pred_x0

    
class ExtractLatent:
    def __init__(self, latent_save_path="./extensions/sd-webui-ditail/features"):
        # self.init_image = init_image
        # self.model = model
        # self.positive_prompt = positive_prompt
        # self.negative_prompt = negative_prompt
        # self.sampler = DDIMSampler(self.model)
        # self.seed = seed
        # self.save_feature_timesteps = ddim_steps
        self.outpath = create_path(latent_save_path) #TODO: change this to a parameter
        self.sample_path = create_path(os.path.join(self.outpath, "samples"))
        self.precision_scope = autocast
        self.device = device
        print('!! extract latent device is', self.device)

        # seed_everything(seed)
        pass

    def save_feature_maps_callback(self, model, i):
        self.save_feature_maps(model.output_blocks, i, "output_block") 

    def save_feature_maps(self, blocks, i, feature_type="input_block"):
        block_idx = 0
        for block in tqdm(blocks, desc="Saving input blocks feature maps"):
            # if not opt.save_all_features and block_idx < 4:
            if block_idx < 4:
                block_idx += 1
                continue
            if "ResBlock" in str(type(block[0])):
                # if opt.save_all_features or block_idx == 4:
                if block_idx == 4:
                    print('!! check block 0', type(block[0]), block[0].__dict__)
                    self.save_feature_map(block[0].in_layers_features, f"{feature_type}_{block_idx}_in_layers_features_time_{i}")
                    self.save_feature_map(block[0].out_layers_features, f"{feature_type}_{block_idx}_out_layers_features_time_{i}")

            if len(block) > 1 and "SpatialTransformer" in str(type(block[1])):
                self.save_feature_map(block[1].transformer_blocks[0].attn1.to_k, f"{feature_type}_{block_idx}_self_attn_k_time_{i}")
                self.save_feature_map(block[1].transformer_blocks[0].attn1.to_q, f"{feature_type}_{block_idx}_self_attn_q_time_{i}")
            block_idx += 1
    
    def save_feature_map(self, feature_map, filename):
        save_path = os.path.join(self.outpath, f"{filename}.pt")
        torch.save(feature_map, save_path)

    @torch.inference_mode()
    def extract(self, init_image, model, positive_prompt, negative_prompt, alpha=3.0, beta=0.5, seed=42, ddim_inversion_steps=999, save_feature_timesteps=60, H=512, W=512, C=4, f=8, cfgscale=7.5):
        '''
        C for latent channels
        f for downsampling factor
        '''
        # with self.model.ema_scope() #TODO: check if this is necessary
        seed_everything(seed)
        sampler = DDIMInverse(model)

        # uc = model.get_learned_conditioning([""])
        # TODO: setting prompt inside a list is a hack, check if this is necessary
        pos_c = model.get_learned_conditioning([positive_prompt])
        neg_c = model.get_learned_conditioning([negative_prompt])
        c = alpha * pos_c - beta * neg_c
        print('!! c shape', c.shape)
        # shape = [C, H//f, W//f]

        z_enc = None
        # turn init_image to tensor
        init_image = torch.from_numpy(init_image).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        print('!! init_latent shape', init_latent.shape)
        # ddim_inversion_steps = 999
        latents, z_enc = sampler.encode_ddim(
            init_latent, 
            num_steps=ddim_inversion_steps, 
            conditioning=c, 
            save_feature_maps_callback=self.save_feature_maps_callback,
        ) 

        return latents, z_enc

        # torch.save(z_enc, os.path.join(self.outpath, "z_enc.pt"))
        # print('z_enc saved')


