import os

import torch
import numpy as np

from tqdm import tqdm
from torch import autocast
from PIL import Image
from pytorch_lightning import seed_everything

from ditail.utils import create_path
from ldm.models.diffusion.ddim import DDIMSampler

class DDIMInverse(DDIMSampler):
    def __init__(self, model):
        super().__init__(model)

    def encode_ddim(self, 
                    img, 
                    num_steps, 
                    conditioning, 
                    unconditional_conditioning, 
                    unconditional_guidance_scale
        ):
        print(f"Running DDIM inversion with {num_steps} steps")
        T = 999
        c = T // num_steps
        iterator = tqdm(range(0, T, c), desc="DDIM inversion", total = num_steps)
        steps = list(range(0, T+c, c))

        for i, t in enumerate(iterator):
            img, _ = self.reverse_ddim(img, t, t_next=steps[i+1], c=conditioning)

        return img, _
        # pass # TODO: add reverse_ddim

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
        mus = torch.sqrt(alphas)

        # TODO: check out the 'next' and 'prev' timestep
        a_t = torch.full((b,1,1,1), alphas[t], device=device)
        a_next = torch.full((b,1,1,1), alphas[t_next], device=device)
        mu_t = mus[t]
        mu_next = mus[t_next]
        sigma_t = torch.full((b,1,1,1), sigmas[t], device=device)
        sigma_next = torch.full((b,1,1,1), sigmas[t_next], device=device)

        pred_x0 = (x - sigma_t * eps) / mu_t
        x_next = a_next * pred_x0 + sigma_next * eps
        return x_next, pred_x0

    
class ExtractLatent:
    def __init__(self):
        # self.init_image = init_image
        # self.model = model
        # self.positive_prompt = positive_prompt
        # self.negative_prompt = negative_prompt
        # self.sampler = DDIMSampler(self.model)
        # self.seed = seed
        # self.save_feature_timesteps = ddim_steps
        self.outpath = create_path("features") #TODO: change this to a parameter
        self.sample_path = create_path(os.path.join(self.outpath, "samples"))
        self.precision_scope = autocast

        # seed_everything(seed)
        pass

    @torch.inference_mode()
    def extract(self, init_image, model, positive_prompt, negative_prompt, alpha=3.0, beta=0.5, seed=42, ddim_steps=999, save_feature_timesteps=60, H=512, W=512, C=4, f=8, cfgscale=7.5):
        '''
        C for latent channels
        f for downsampling factor
        '''
        # with self.model.ema_scope() #TODO: check if this is necessary
        seed_everything(seed)
        sampler = DDIMInverse(model)

        # uc = model.get_learned_conditioning([""])
        pos_c = model.get_learned_conditioning(positive_prompt)
        print('!! pos_c', pos_c.shape)
        neg_c = model.get_learned_conditioning(negative_prompt)
        print('!! neg_c', neg_c.shape)
        c = alpha * pos_c - beta * neg_c
        print('!! c', c)
        # shape = [C, H//f, W//f]

        z_enc = None
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        ddim_inversion_steps = 999
        z_enc, _ = sampler.encode_ddim(
            init_latent, 
            num_steps=ddim_inversion_steps, 
            conditoning=c, 
            # unconditional_conditioning=uc, 
            # unconditional_guidance_scale=cfgscale, #TODO: check cfg scale
        ) 

        torch.save(z_enc, os.path.join(self.outpath, "latent.pt"))
        print('latent saved')


