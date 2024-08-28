import os

import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import autocast
from PIL import Image
from pytorch_lightning import seed_everything

from ditail.utils import create_path
from modules.shared import device
from ldm.models.diffusion.ddim import DDIMSampler

class DDIMInverse(DDIMSampler):
    def __init__(self, model, timesteps_sched=None, sigmas_sched=None):
        super().__init__(model)
        # turn timesteps_sched from tensor to list and reverse it
        # print('check timesteps_sched type', type(timesteps_sched))

        self.timesteps_sched = [float(t) for t in timesteps_sched]
        if sigmas_sched is not None:
            sigmas_sched = sigmas_sched.tolist()
        else:
            T = 1000
            c = T // len(self.timesteps_sched)
            iterator = range(1, T, c)
            sigmas_sched = [self.model.sqrt_one_minus_alphas_cumprod[t] for t in iterator]
            # sigmas_sched = self.model.get_sigmas(len(self.timesteps_sched))
        self.sigmas_sched = [float(s) for s in reversed(sigmas_sched)]

        # print('!! timesteps_sched', self.timesteps_sched)
        # print('!! sigmas_sched', self.sigmas_sched)

    def encode_ddim(self, 
                    img, 
                    num_steps, 
                    conditioning,
                    # save_feature_maps_callback=None,
        ):
        latents = {}
        # print(f"Running DDIM inversion with {num_steps} steps")


        latents[self.timesteps_sched[0]] = img
        for i in tqdm(range(1, len(self.timesteps_sched)), desc="DDIM inversion"):
            img, _, t = self.reverse_ddim(img, step_idx=i, c=conditioning)
            # print('!! t at idx', i, t[0], t.shape)
            latents[t[0].item()] = img

        # print('!! latents keys', latents.keys())

        return latents, img

    @torch.no_grad()
    def reverse_ddim(self, 
                     x, 
                     step_idx,
                     c=None, 
                     quantize_denoised=False, #TODO: check if this is necessary,
                    #  unconditional_guidance_scale=1.,
                    #  unconditional_conditioning=None,
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

        # # sigma =  self.sigmas_sched[step_idx]
        # # sigma_prev = self.sigmas_sched[step_idx-1]


        # t_tensor = torch.full((b,), t, device=device, dtype=torch.long)
        # # if c is None:
        # #     e_t = self.model.apply_model(x, t_tensor, unconditional_conditioning)
        # # elif unconditional_conditioning is None or unconditional_guidance_scale==1.:
        # #     e_t = self.model.apply_model(x, t_tensor, c)

        # # else:
        # #     x_in = torch.cat([x] * 2)
        # #     t_in = torch.cat([t_tensor] * 2)
        # #     c_in = torch.cat([unconditional_conditioning, c])
        # #     e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
        # #     e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
        # eps = self.model.apply_model(x, t_tensor, c)

        # alphas = self.model.alphas_cumprod
        # sigmas = self.model.sqrt_one_minus_alphas_cumprod

        # # TODO: check out the 'next' and 'prev' timestep
        # a_t = torch.full((b,1,1,1), alphas[t], device=device)
        # a_next = torch.full((b,1,1,1), alphas[t_next], device=device)

        # sigma_t = torch.full((b,1,1,1), sigmas[t], device=device)

        # pred_x0 = (x - sigma_t * eps) / a_t.sqrt()
        # dir_xt = (1. - a_next).sqrt() * eps
        # x_next = a_next.sqrt() * pred_x0 + dir_xt

        # return x_next, pred_x0

    
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
        # print('!! extract latent device is', self.device)

    @torch.inference_mode()
    def extract(self, init_image, model, positive_prompt, negative_prompt, timesteps_sched, sigmas_sched, alpha=3.0, beta=0.5, seed=42, ddim_inversion_steps=999, save_feature_timesteps=60, H=512, W=512, C=4, f=8, cfgscale=7.5):
        '''
        C for latent channels
        f for downsampling factor
        '''
        # with self.model.ema_scope() #TODO: check if this is necessary
        seed_everything(seed)
        sampler = DDIMInverse(model, timesteps_sched=timesteps_sched, sigmas_sched=sigmas_sched)

        # uc = model.get_learned_conditioning([""])
        # TODO: setting prompt inside a list is a hack, check if this is necessary
        pos_c = model.get_learned_conditioning([positive_prompt])
        neg_c = model.get_learned_conditioning([negative_prompt])
        c = alpha * pos_c - beta * neg_c
        # print('!! c shape', c.shape)
        # shape = [C, H//f, W//f]

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

        # print('!! latent_check, min, max', latent_check.min(), latent_check.max())
        # plt.figure()
        # plt.imshow(latent_check)
        # plt.savefig(f'./extensions/sd-webui-ditail/features/samples/latent_init.png')

        # print('!! init_latent shape', init_latent.shape)
        # ddim_inversion_steps = 999
        latents, z_enc = sampler.encode_ddim(
            init_latent, 
            num_steps=ddim_inversion_steps, 
            conditioning=c, 
            # save_feature_maps_callback=self.save_feature_maps_callback,
        ) 

        return latents, z_enc

        # torch.save(z_enc, os.path.join(self.outpath, "z_enc.pt"))
        # print('z_enc saved')


