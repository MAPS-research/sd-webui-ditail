import contextlib
import copy
import os
import sys
import platform
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, NamedTuple

import gradio as gr
import pickle
import torch
from tqdm import tqdm
from PIL import Image

import modules
import ldm.modules.diffusionmodules.openaimodel
from modules import scripts, script_callbacks, shared, sd_models, sd_samplers, sd_unet, safe
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button
from modules.sd_samplers import all_samplers, get_sampler_and_scheduler, all_samplers_map
from modules.sd_schedulers import schedulers_map
from modules.shared import cmd_opts, opts, state
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing
from modules.sd_samplers_kdiffusion import samplers_k_diffusion, k_diffusion_samplers_map
from modules.sd_samplers_timesteps import CompVisSampler

from ditail import (
    DITAIL,
    __version__,
)
from ditail.args import ALL_ARGS, DitailArgs
from ditail.ui import WebuiInfo, ditailui
from ditail.extract_features import ExtractLatent
from ditail.register_forward import register_attn_inj, register_conv_inj, register_time
from ditail.utils import create_path


txt2img_submit_button = img2img_submit_button = None

print(
    f"[-] Ditail script loaded. version: {__version__}, device: {shared.device}"
)

def send_text_to_prompt(new_text, old_text, new_neg_text, old_neg_text):
    # if old_text == "":  # if text on the textbox text2img or img2img is empty, return new text
    #     return new_text
    # return old_text + " " + new_text  # else join them together and send it to the textbox
    return new_text, new_neg_text




class DitailScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.ultralytics_device = self.get_ultralytics_device()
        self.latents = None
        self.z_enc = None
    
    def __repr__(self):
        return f"{self.__class__.__name__}(version={__version__})"
        
    @staticmethod
    def get_ultralytics_device() -> str:
        if "adetailer" in shared.cmd_opts.use_cpu:
            return "cpu"

        if platform.system() == "Darwin":
            return ""

        vram_args = ["lowvram", "medvram", "medvram_sdxl"]
        if any(getattr(cmd_opts, vram, False) for vram in vram_args):
            return "cpu"

        return ""
    
    def get_i(self, p) -> int:
        iteration = p.iteration
        batch_size = p.batch_size
        if batch_index := getattr(p, "batch_index", None):
            return iteration * batch_size + batch_index
        else:
            return iteration * batch_size + 0


    def title(self):
        return DITAIL

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        try:
            checkpoint_list = sd_models.checkpoint_tiles(use_shorts=True)
        except TypeError:
            checkpoint_list = modules.sd_models.checkpoint_tiles()
        vae_list = modules.shared_items.sd_vae_items()

        webui_info = WebuiInfo(
            checkpoints_list=checkpoint_list,
            vae_list=vae_list,
            t2i_button=txt2img_submit_button,
            i2i_button=img2img_submit_button,
        )

        components, infotext_fields = ditailui(is_img2img, webui_info)
        if is_img2img:
            components[0] = self.img2img_image
        # components = self.replace_components(components, is_img2img)
        self.infotext_fields = infotext_fields 
        print("!! check components", components)
        return components

    
    def is_ditail_enabled(self, ditail_args: DitailArgs) -> bool: 
        if ditail_args.src_img is None:
            message = """
            !! Ditail: No source image detected.
               Ditail disabled.
            """
            print(dedent(message), file=sys.stderr)
            return False

        return ditail_args.enable_ditail
    

    def replace_empty_args(self, p, ditail_args: DitailArgs) -> DitailArgs:
        i = self.get_i(p)
        # TODO: check whether prompt should be str or list
        ditail_args.inv_prompt = p.all_prompts[i] if ditail_args.inv_prompt == '' else ditail_args.inv_prompt
        ditail_args.inv_negative_prompt = p.all_negative_prompts[i] if ditail_args.inv_negative_prompt == '' else ditail_args.inv_negative_prompt
        ditail_args.inv_steps = p.steps

        # replace sampler and scheduler as p for now, maybe allow user to change it later
        ditail_args.inv_sampler_name = p.sampler_name
        ditail_args.inv_scheduler_name = p.scheduler

        return ditail_args
    
    
    def get_scheduler_timesteps(self, p, sampler_name, scheduler_name):
        sampler_name, scheduler_name = get_sampler_and_scheduler(sampler_name, scheduler_name)

        # print('!! sampler name', p.sampler, self.sampler_name, sampler_name, scheduler_name)

        sampler = all_samplers_map.get(sampler_name)
        sampler_class = sampler.constructor(shared.sd_model)
        sampler_class.config = sampler

        if isinstance(sampler_class, CompVisSampler):
            timesteps_sched = sampler_class.get_timesteps(p, p.steps)
            sigmas_sched = None
        else:
            sigmas_sched = sampler_class.get_sigmas(p, p.steps)
            timesteps_sched = [sampler_class.model_wrap.sigma_to_t(s) for s in sigmas_sched]

        return timesteps_sched, sigmas_sched
        # print('!! sampler', sampler)

        # print('!! sampler and scheduler name', sampler_name, scheduler_name)
        # if scheduler_name == 'Automatic':
        #     scheduler = schedulers_map.get('uniform')
        #     sampler_row = [row for row in samplers_k_diffusion if row[0] == sampler_name]
        #     if len(sampler_row) > 0:
        #         options = sampler_row[0][3]
        #         if options.get('scheduler', False):
        #             scheduler = schedulers_map.get(options['scheduler'])
        # scheduler = scheduler

        # print('!! device check', sigmas.device)

        # print('!! scheduler', scheduler, type(scheduler))
        # sigmas_sched = scheduler.function(p.steps, sigma_min=min(sigmas), sigma_max=max(sigmas), device=shared.device)
        # timestep_sched = [sampler_class.model_wrap.sigma_to_t(s) for s in sigmas_sched]
        # print('!! timestep_sched', timestep_sched)
        # # print('!! timesteps', scheduler.function(p.steps, sigma_min=min(sigmas), sigma_max=max(sigmas), device=shared.device))

        # return scheduler.function(p.steps, sigma_min=min(sigmas), sigma_max=max(sigmas), device=shared.device)


    def process(self, p, *args):
        # model_structure = str(shared.sd_model)
        # with open('./extensions/sd-webui-ditail/scripts/model_structure.txt', 'w') as f:
        #     f.write(model_structure)

        # print('!! p', type(p))'
        # print(!!'opts', opts.scheduler, type(opts))

        # force change sampler to DDIM
        p.sampler_name = "DDIM"

        print('!! get i', self.get_i(p))
        shared.sd_model.cond_stage_key = "edit"

        sampler_config = sd_samplers.find_sampler_config(p.sampler_name)
        total_steps = sampler_config.total_steps(p.steps)
        print('!! total sampler steps', total_steps)

        # map args to ditail_args
        ditail_args = DitailArgs()
        ditail_args.src_img, ditail_args.enable_ditail = args[0], args[1]
        for k, v in args[2].items():
            setattr(ditail_args, k, v)
        
        ditail_args = self.replace_empty_args(p, ditail_args)

        self.swap_txt2img_pipeline(p, init_images=[Image.fromarray(ditail_args.src_img)])
        # create an PIL image of random gaussian noise as init_images input
        # random_noise_img = Image.fromarray((torch.randn(3, 256, 256) * 255).byte().cpu().numpy().transpose(1, 2, 0))
        # self.swap_txt2img_pipeline(p, init_images=[random_noise_img])
        # script_callbacks.on_cfg_denoiser(self.sampling_check_callback)


        if self.is_ditail_enabled(ditail_args):
            print('!! ditail enabled')
            # overwrite empty args

            # self.load_inv_model(ditail_args.src_model_name)

            self.timesteps_sched, self.sigmas_sched = self.get_scheduler_timesteps(p, ditail_args.inv_sampler_name, ditail_args.inv_scheduler_name)
            self.timesteps_sched_sampling = reversed(self.timesteps_sched)
            latent_save_path = create_path("./extensions/sd-webui-ditail/features")
            self.extract_latents(p, ditail_args, shared.sd_model, latent_save_path)

            conv_threshold = int(0.8 * len(self.timesteps_sched_sampling))
            attn_threshold = int(0.5 * len(self.timesteps_sched_sampling))
        
            register_conv_inj(shared.sd_model.model.diffusion_model, injection_schedule=self.timesteps_sched_sampling[:conv_threshold])
            register_attn_inj(shared.sd_model.model.diffusion_model, injection_schedule=self.timesteps_sched_sampling[:attn_threshold])
        
            script_callbacks.on_cfg_denoiser(self.sampling_loop_start_callback)

    def swap_txt2img_pipeline(self, p: StableDiffusionProcessingTxt2Img, init_images: list):
        p.__class__ = StableDiffusionProcessingImg2Img
        dummy = StableDiffusionProcessingImg2Img()
        for k, v in dummy.__dict__.items():
            if hasattr(p, k):
                continue
            setattr(p, k, v)
        p.init_images = init_images
        p.initial_noise_multiplier = 1.0
        p.image_cfg_scale = p.cfg_scale
        p.denoising_strength = 1.0

    def load_inv_model(self, checkpoint_name):

        print('!! shared model', shared.opts.sd_model_checkpoint)
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
        print("!! loading inv model", checkpoint_info, type(checkpoint_info))
        self.old_model = shared.sd_model
        inv_model = sd_models.reload_model_weights(info = checkpoint_info)
        print("!! inv model loaded", type(inv_model), inv_model.sd_model_checkpoint)
        print("!! new shared model", shared.opts.sd_model_checkpoint)

        # print("!! current sampler name", sampler_name)

        # sd_models.reload_model_weights(sd_model = self.old_model)
        # print("!! Reloaded old model", shared.opts.sd_model_checkpoint)
        # with contextlib.suppress(AttributeError):  # Ignore the error if the attribute is not present
        #     if is_img2img:
        #         # Bind the click event of the button to the send_text_to_prompt function
        #         # Inputs: text_to_be_sent (textbox), self.boxxIMG (textbox)
        #         # Outputs: self.boxxIMG (textbox)
        #         # send_text_button.click(fn=send_text_to_prompt, inputs=[text_to_be_sent, self.boxxIMG], outputs=[self.boxxIMG])
        #         pass
        #     else:
        #         # Bind the click event of the button to the send_text_to_prompt function
        #         # Inputs: text_to_be_sent (textbox), self.boxx (textbox)
        #         # Outputs: self.boxx (textbox)
        #         pass
        #         # send_text_button.click(fn=send_text_to_prompt, inputs=[text_to_be_sent, self.boxx, negative_text_to_be_sent, self.neg_prompt_boxTXT], outputs=[self.boxx, self.neg_prompt_boxTXT])

        # return [text_to_be_sent, send_text_button]
    
    def extract_latents(self, p, ditail_args: DitailArgs, model, latent_save_path, seed=42):
        extracter = ExtractLatent(latent_save_path)

        assert self.timesteps_sched is not None, "Timesteps scheduler is not set"

        self.latents, self.z_enc = extracter.extract(
                init_image=ditail_args.src_img,
                model=model,
                positive_prompt=ditail_args.inv_prompt,
                negative_prompt=ditail_args.inv_negative_prompt,
                timesteps_sched=self.timesteps_sched,
                sigmas_sched=self.sigmas_sched,
                alpha=ditail_args.ditail_alpha,
                beta=ditail_args.ditail_beta,
                seed=seed,
                ddim_inversion_steps=ditail_args.inv_steps,
            )

    def sampling_check_callback(self, params):
        # plot the latent during generation
        import matplotlib.pyplot as plt
        latent_check = params.x[0].permute(1, 2, 0).cpu().numpy()

        plt.figure()
        plt.imshow(latent_check)
        plt.savefig(f'./extensions/sd-webui-ditail/features/samples/gen_{params.sampling_step}.png')

    
    def sampling_loop_start_callback(self, params):
        print('x size', params.x.shape)
        print('sampling step', params.sampling_step)
        print('sampling steps', params.total_sampling_steps)

        # replace the image condition chunk with the extracted latent for injection
        params.x[1] = self.latents[params.sigma[0].item()]
        params.image_cond = torch.zeros_like(params.image_cond)
        register_time(shared.sd_model.model.diffusion_model, params.sigma[0].item())
        print('registered time', params.sigma[0].item())
        return params



    # def after_component(self, component, **kwargs):
    #     #https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7456#issuecomment-1414465888 helpfull link
    #     # Find the text2img textbox component
    #     if kwargs.get("elem_id") == "txt2img_prompt": #postive prompt textbox
    #         self.boxx = component
    #         print('!! boxx', self.boxx)
    #     # Find the img2img textbox component
    #     if kwargs.get("elem_id") == "img2img_prompt":  #postive prompt textbox
    #         self.boxxIMG = component

    #     #this code below  works aswell, you can send negative prompt text box,provided you change the code a little
    #     #switch  self.boxx with  self.neg_prompt_boxTXT  and self.boxxIMG with self.neg_prompt_boxIMG

    #     if kwargs.get("elem_id") == "txt2img_neg_prompt":
    #         self.neg_prompt_boxTXT = component
    #     if kwargs.get("elem_id") == "img2img_neg_prompt":
    #         self.neg_prompt_boxIMG = component

    #     if kwargs.get("elem_id") == "txt2img_sampling":
    #         self.sampler_name = component
    #         print('!! sampler name', self.sampler_name)
    #         # self.sampler_name.change(fn=self.show_sampler, inputs=[self.sampler_name])

    def after_component(self, component, **kwargs):
        if kwargs.get("elem_id") == "img2img_image":
            self.img2img_image = component
        
        # if kwargs.get("elem_id") == "txt2img_prompt":
        #     self.txt2img_prompt = component
        # if kwargs.get("elem_id") == "img2img_prompt":
        #     self.img2img_prompt = component
        # if kwargs.get("elem_id") == "txt2img_neg_prompt":
        #     self.txt2img_neg_prompt = component
        # if kwargs.get("elem_id") == "img2img_neg_prompt":
        #     self.img2img_neg_prompt = component

    def post_sample(self, p, ps: scripts.PostSampleArgs, *args):
        # clear up callbacks
        script_callbacks.remove_current_script_callbacks()

