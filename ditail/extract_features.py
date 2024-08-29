import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import autocast
from PIL import Image
from pytorch_lightning import seed_everything

from modules.shared import device
from ldm.models.diffusion.ddim import DDIMSampler

class DDIMInverse(DDIMSampler):
    def __init__(self, model, timesteps_sched=None, sigmas_sched=None):
        super().__init__(model)
        # turn timesteps_sched from tensor to list and reverse it
        self.timesteps_sched = [float(t) for t in timesteps_sched]
        if sigmas_sched is not None:
            sigmas_sched = sigmas_sched.tolist()
        else:
            T = 1000
            c = T // len(self.timesteps_sched)
            iterator = range(1, T, c)
            sigmas_sched = [self.model.sqrt_one_minus_alphas_cumprod[t] for t in iterator]
        self.sigmas_sched = [float(s) for s in reversed(sigmas_sched)]
        # print('!! timesteps_sched', self.timesteps_sched)
        # print('!! sigmas_sched', self.sigmas_sched)

    def encode_ddim(self, 
                    img, 
                    num_steps, 
                    conditioning,
        ):
        latents = {}
        # print(f"Running DDIM inversion with {num_steps} steps")

        latents[self.timesteps_sched[0]] = img
        for i in tqdm(range(1, len(self.timesteps_sched)), desc="DDIM inversion"):
            img, _, t = self.reverse_ddim(img, step_idx=i, c=conditioning)
            latents[t[0].item()] = img
        return latents, img

    @torch.no_grad()
    def reverse_ddim(self, 
                     x, 
                     step_idx,
                     c=None, 
                     quantize_denoised=False, #TODO: not implemented yet
        ):
        b, *_, device = *x.shape, x.device

        t = torch.full((b,), self.timesteps_sched[step_idx], device=device)
        t_prev = torch.full((b,), self.timesteps_sched[step_idx-1], device=device)

        sigma = torch.full((b,1,1,1), self.sigmas_sched[step_idx], device=device)
        sigma_prev = torch.full((b,1,1,1), self.sigmas_sched[step_idx-1], device=device)

        # alpha = 1-sigma**2
        mu = torch.sqrt(1-sigma**2)
        # alpha_prev = 1-sigma_prev**2

        eps = self.model.apply_model(x, t, c)

        pred_x0 = (x - sigma * eps) / mu
        x_prev = mu * pred_x0 + sigma * eps

        # x_prev should not be Nan
        assert not torch.isnan(x_prev).any()

        return x_prev, pred_x0, t

class ExtractLatent:
    def __init__(self):
        self.precision_scope = autocast
        self.device = device

    @torch.inference_mode()
    def extract(self, init_image, model, positive_prompt, negative_prompt, timesteps_sched, sigmas_sched, alpha=3.0, beta=0.5, seed=42, ddim_inversion_steps=999):

        seed_everything(seed)
        sampler = DDIMInverse(model, timesteps_sched=timesteps_sched, sigmas_sched=sigmas_sched)

        pos_c = model.get_learned_conditioning([positive_prompt])
        neg_c = model.get_learned_conditioning([negative_prompt])
        c = alpha * pos_c - beta * neg_c
        # print('!! c shape', c.shape)

        z_enc = None
        # turn init_image to tensor
        init_image = np.array(init_image).astype(np.float32) / 255.0
        init_image = np.moveaxis(init_image, 2, 0)
        init_image = 2.0 * init_image - 1.0
        init_image = torch.from_numpy(init_image).unsqueeze(0).to(self.device)

        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))

        latent_check = model.decode_first_stage(init_latent)
        # latent_check = torch.clamp((latent_check+1.0)/2.0, 0.0, 1.0)
        latent_check = latent_check[0].permute(1, 2, 0).cpu().numpy()

        latents, z_enc = sampler.encode_ddim(
            init_latent, 
            num_steps=ddim_inversion_steps, 
            conditioning=c, 
        ) 

        return latents, z_enc


