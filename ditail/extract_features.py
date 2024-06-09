import os

import torch
import numpy as np

from torch import autocast
from PIL import Image
from pytorch_lightning import seed_everything

from ditail.utils import create_path
from ldm.models.diffusion.ddim import DDIMSampler

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
    def extract(self, init_image, model, positive_prompt, negative_prompt, seed=42, ddim_steps=999, save_feature_timesteps=60, H=512, W=512, C=4, f=8, cfgscale=7.5):
        '''
        C for latent channels
        f for downsampling factor
        '''
        # with self.model.ema_scope() #TODO: check if this is necessary
        seed_everything(seed)
        sampler = DDIMSampler(model)

        uc = model.get_learned_conditioning([""])
        c = model.get_learned_conditioning(positive_prompt)
        shape = [C, H//f, W//f]

        z_enc = None
        init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))
        ddim_inversion_steps = 999
        z_enc, _ = sampler.encode_ddim(
            init_latent, 
            num_steps=ddim_inversion_steps, 
            conditoning=c, 
            unconditional_conditioning=uc, 
            unconditional_guidance_scale=cfgscale, #TODO: check cfg scale
        ) 

        torch.save(z_enc, os.path.join(self.outpath, "latent.pt"))


