import contextlib
import copy
import os
import sys
import platform
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import gradio as gr
import pickle
import torch
from tqdm import tqdm
from PIL import Image

import modules
import ldm.modules.diffusionmodules.openaimodel
from modules import scripts, script_callbacks, shared, sd_models, sd_vae, sd_samplers, sd_unet, safe
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
from ditail.register_forward import register_attn_inj, unregister_attn_inj, register_conv_inj, unregister_conv_inj, register_time

txt2img_submit_button = img2img_submit_button = None

print(
    f"[-] Ditail script loaded. version: {__version__}, device: {shared.device}"
)

class DitailScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.ultralytics_device = self.get_ultralytics_device()
        self.latents = None
        self.z_enc = None
        self.original_model_cond_stage_key = None
        self.original_processing_pipeline = None
        self.original_checkpoint_name = None
        self.original_vae_name = None
        self.is_ditail_enabled = False
        self.ditail_process_callback = self.disable_ditail_callback # set to disable by default
    
    def __repr__(self):
        return f"{self.__class__.__name__}(version={__version__})"
        
    @staticmethod
    def get_ultralytics_device() -> str:
        if "ditail" in shared.cmd_opts.use_cpu:
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
        self.infotext_fields = infotext_fields 
        
        enable_ditail_checkbox = components[1]
        enable_ditail_checkbox.select(fn=self.switch_ditail_onoff, inputs=[enable_ditail_checkbox], outputs=[self.sampler_component, self.scheduler_component])

        return components


    def switch_ditail_onoff(self, checkbox_value):
        if checkbox_value:
            self.is_ditail_enabled = True
            self.ditail_process_callback = self.enable_ditail_callback
            return (
                gr.Dropdown.update(interactive=False, value="DDIM"),
                gr.Dropdown.update(interactive=False, value="Automatic"),
            )

        else:
            self.is_ditail_enabled = False
            self.ditail_process_callback = self.disable_ditail_callback
            return (
                gr.Dropdown.update(interactive=True),
                gr.Dropdown.update(interactive=True)
            )

    def enable_ditail_callback(self, p, ditail_args: DitailArgs):
        print('[-] Ditail enabled')

        # preprocess ditail_args.src_img
        # TODO: make resize mode configurable in UI
        ditail_args.src_img = modules.images.resize_image(resize_mode="0", im=ditail_args.src_img, width=p.width, height=p.height, upscaler_name=None)
        # other src_img preprocessing are done in extrac_features.py

        # replace pipeline to img2img
        self.original_model_cond_stage_key = shared.sd_model.cond_stage_key
        self.swap_xxx2img_pipeline(p, init_images=[ditail_args.src_img])

        self.load_model(ditail_args.src_model_name, ditail_args.src_vae_name, for_inv=True)

        self.timesteps_sched, self.sigmas_sched = self.get_scheduler_timesteps(p) # timesteps_sched example: [1.0, 51.0, 101.0, 151.0 ... ]
        self.latents, self.z_enc = self.extract_latents(p, ditail_args, shared.sd_model, self.timesteps_sched, self.sigmas_sched)

        self.load_model(self.original_checkpoint_name, self.original_vae_name, for_inv=False)
        shared.sd_model.cond_stage_key = "edit"

        conv_threshold = int(ditail_args.conv_ratio * len(self.timesteps_sched))
        attn_threshold = int(ditail_args.attn_ratio * len(self.timesteps_sched))

        register_conv_inj(shared.sd_model.model.diffusion_model, injection_schedule=reversed(self.timesteps_sched)[:conv_threshold])
        register_attn_inj(shared.sd_model.model.diffusion_model, injection_schedule=reversed(self.timesteps_sched)[:attn_threshold])

        # add sampling loop callback for feature injection
        script_callbacks.on_cfg_denoiser(self.sampling_loop_start_callback)

    def disable_ditail_callback(self, p, ditail_args: DitailArgs):
        # print('[-] Ditail disabled')

        # reset the model condition stage key
        if self.original_model_cond_stage_key:
            shared.sd_model.cond_stage_key = self.original_model_cond_stage_key

        # reset the processing pipeline
        if self.original_processing_pipeline:
            self.swap_xxx2orig_pipeline(p)

        self.timesteps_sched, self.sigmas_sched = None, None

        self.latents, self.z_enc = None, None

        # make injection not happen
        unregister_conv_inj(shared.sd_model.model.diffusion_model)
        unregister_attn_inj(shared.sd_model.model.diffusion_model)
        
        # remove sampling loop callback
        script_callbacks.remove_current_script_callbacks()

    
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
    
    
    def get_scheduler_timesteps(self, p):
        sampler_name, scheduler_name = get_sampler_and_scheduler(p.sampler_name, p.scheduler)
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

    def process(self, p, *args):
        # map args to ditail_args
        ditail_args = DitailArgs()
        ditail_args.src_img = args[0]
        for k, v in args[2].items():
            setattr(ditail_args, k, v)
        
        ditail_args = self.replace_empty_args(p, ditail_args)

        if ditail_args.src_img is None:
            self.is_ditail_enabled = False
            self.ditail_process_callback = self.disable_ditail_callback
            print('[!] ditail is not working because no source image is provided')
        
        if self.ditail_process_callback is not None:
            self.ditail_process_callback(p, ditail_args)

    def swap_xxx2img_pipeline(self, p, init_images: list):
        self.original_processing_pipeline = p.__class__
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
    
    def swap_xxx2orig_pipeline(self, p):
        dummy = self.original_processing_pipeline()
        for k, v in p.__dict__.items():
            if hasattr(dummy, k):
                setattr(dummy, k, v)
        p = dummy

    def func_trial(self, func, max_tries=2, *args, **kwargs):
        for i in range(max_tries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if i == max_tries - 1:
                    print(f"[!] Error occured when executing {func.__name__}, ditail will not work")
                else:
                    print(f"[!] Error occured when executing {func.__name__}, retrying... {i+1}/{max_tries}")
        return None

    def load_model(self, checkpoint_name, vae_name=None, for_inv=True):
        shared.opts.sd_vae_overrides_per_model_preferences = True
        if shared.opts.sd_model_checkpoint != checkpoint_name:
            if for_inv:
                # we need to keep the original model to be able to switch back
                self.original_checkpoint_name = shared.opts.sd_model_checkpoint
                self.original_vae_name = shared.opts.sd_vae

            if vae_name is not None and shared.opts.sd_vae != vae_name:
                shared.opts.sd_vae = vae_name
            checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
            self.func_trial(sd_models.reload_model_weights, 2, info = checkpoint_info)
    
    def extract_latents(self, p, ditail_args: DitailArgs, model, timesteps_sched, sigmas_sched, seed=42):
        extracter = ExtractLatent()
        assert timesteps_sched is not None, "Timesteps scheduler is not set"

        latents, z_enc = extracter.extract(
                init_image=ditail_args.src_img,
                model=model,
                positive_prompt=ditail_args.inv_prompt,
                negative_prompt=ditail_args.inv_negative_prompt,
                timesteps_sched=timesteps_sched,
                sigmas_sched=sigmas_sched,
                alpha=ditail_args.ditail_alpha,
                beta=ditail_args.ditail_beta,
                seed=seed,
                ddim_inversion_steps=ditail_args.inv_steps,
            )
    
        return latents, z_enc
    
    def sampling_loop_start_callback(self, params):
        # replace the image condition chunk with the extracted latent for injection
        params.x[1] = self.latents[params.sigma[0].item()]
        params.image_cond = torch.zeros_like(params.image_cond)
        register_time(shared.sd_model.model.diffusion_model, params.sigma[0].item())
        return params


    def after_component(self, component, **kwargs):
        if kwargs.get("elem_id") == "img2img_image":
            self.img2img_image = component
        
        if kwargs.get("elem_id") == "img2img_sampling" or kwargs.get("elem_id") == "txt2img_sampling":
            self.sampler_component = component
        
        if kwargs.get("elem_id") == "img2img_scheduler" or kwargs.get("elem_id") == "txt2img_scheduler":
            self.scheduler_component = component

    def post_sample(self, p, ps: scripts.PostSampleArgs, *args):
        # clear up callbacks
        script_callbacks.remove_current_script_callbacks()

