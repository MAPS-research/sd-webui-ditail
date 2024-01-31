from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.transforms as T

from pytorch_lightning import seed_everything

from modules import shared, sd_models

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler


@torch.no_grad()
def encode_image(self, img_path):
    image_pil = T.Resize(512)(Image.open(img_path).convert('RGB'))
    image = T.ToTensor()(image_pil).unsqueeze(0).to(self.device)
    with torch.autocast(device_type=self.device, dtype=torch.float32):
        image = 2 * image - 1
        posterior = self.vae.encode(image).latent_dist
        latent = posterior.mean * 0.18215
    return latent

class DitailInverse():
    def __init__(self, prompt, negative_prompt, alpha, beta, seed=0):
        self.model = shared.sd_model
        self.device = shared.device
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.alpha = alpha
        self.beta = beta
        
        seed_everything(seed)



