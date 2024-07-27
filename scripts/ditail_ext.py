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

import modules
import ldm.modules.diffusionmodules.openaimodel
from modules import scripts, script_callbacks, shared, sd_models, sd_samplers, sd_unet, safe
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button
# from modules.sd_samplers import all_samplers
from modules.shared import cmd_opts, opts, state


from ditail import (
    DITAIL,
    __version__,
)
from ditail.args import ALL_ARGS, DitailArgs
from ditail.ui import WebuiInfo, ditailui
from ditail.extract_features import ExtractLatent
from ditail.replace_openaimodel import apply_openaimodel_replacement, UNetModelWithInjection
from ditail.replace_attention import apply_attention_replacement
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
        # self.img2img_image = None
        # self.txt2img_prompt = None
        # self.txt2img_neg_prompt = None
        # self.img2img_prompt = None
        # self.img2img_neg_prompt = None

        # apply attention replacement
        apply_attention_replacement()
        apply_openaimodel_replacement()

        # replace_openaimodel(injected_features=None)
    
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

        # with gr.Accordion("Ditail", open=False):
        #     with gr.Group(visible=not is_img2img) as self.image_upload_panel:
        #         image = gr.Image(
        #                         label="content image",
        #                         source="upload",
        #                         brush_radius=20,
        #                         mirror_webcam=False,
        #                         type="numpy",
        #                         tool="sketch",
        #                         elem_id="input_image",
        #                         elem_classes=["ditail-image"],
        #                         brush_color=shared.opts.img2img_inpaint_mask_brush_color
        #                         if hasattr(
        #                             shared.opts, "img2img_inpaint_mask_brush_color"
        #                         )
        #                         else None,
        #                     )
        #     self.enabled = gr.Checkbox(label="Enable", default=False)
        #     with FormRow(elem_id="ditail_src_model"):
        #         self.src_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="ditail_src_model_name", label="Source Checkpoint")
        #         create_refresh_button(self.src_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_src_checkpoint")
        #         print("!! sampler when creating ditail ui", self.sampler_name)
        #         print("!! src model name when creating ditail ui", self.src_model_name)
        #         # self.src_model_name.change(fn=self.load_inv_model, inputs=[self.src_model_name, self.sampler_name])
                
 
        #     with gr.Row(elem_id="ditail_prompt_weight"):
        #         self.ditail_alpha = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, step=0.1, label="positive prompt scaling weight (alpha)", elem_id="ditail_alpha", interactive=True)
        #         self.ditail_beta = gr.Slider(minimum=0.0, maximum=10.0, value=0.5, step=0.1, label="negative prompt scaling weight (beta)", elem_id="ditail_beta", interactive=True)
        #     # text_to_be_sent = gr.Textbox(label="drop text")
        #     # negative_text_to_be_sent = gr.Textbox(label="drop negative text")
        #     # send_text_button = gr.Button(value='send text', variant='primary')
        #     print('!! device', shared.device)
        #     return image, 

    
    def is_ditail_enabled(self, ditail_args: DitailArgs) -> bool: 
        # src_image = args[0]
        # ditail_enabled = args[1]
        # ditail_args = [a for a in args if isinstance(a, dict)]

        # if not args or not ditail_args:
        #     message = f"""
        #     !! Ditail: Invalid arguments detected.
        #        input: {args!r}
        #        Ditail disabled.
        #     """
        #     print(dedent(message), file=sys.stderr)
        #     return False

        if ditail_args.src_img is None:
            message = f"""
            !! Ditail: No source image detected.
               Ditail disabled.
            """
            print(dedent(message), file=sys.stderr)
            return False

        return ditail_args.enable_ditail
    

    def replace_empty_args(self, p, ditail_args: DitailArgs) -> DitailArgs:
        i = self.get_i(p)
        # if ditail_args['inv_prompt'] == '':
        #     ditail_args['inv_prompt'] = p.all_prompts[i]
        # if ditail_args['inv_negative_prompt'] == '':
        #     ditail_args['inv_negative_prompt'] = p.all_negative_prompts[i]
        # return ditail_args
        # TODO: check whether prompt should be str or list
        ditail_args.inv_prompt = p.all_prompts[i] if ditail_args.inv_prompt == '' else ditail_args.inv_prompt
        ditail_args.inv_negative_prompt = p.all_negative_prompts[i] if ditail_args.inv_negative_prompt == '' else ditail_args.inv_negative_prompt
        ditail_args.inv_steps = p.steps
        return ditail_args


    def process(self, p, *args):
        print('!! get i', self.get_i(p))

        sampler_config = sd_samplers.find_sampler_config(p.sampler_name)
        total_steps = sampler_config.total_steps(p.steps)
        print('!! total sampler steps', total_steps)


        # print('!! check out p', dir(p))
        # print('!! check out p.steps', p.steps)
        # print('!! check out p')
        # for k in dir(p):
        #     if k != 'sd_model':
        #         print(k, getattr(p, k))
        # print('!! check out p.all_prompts', p.all_prompts)

        # print('!! check out args in process', args, type(args))
        # src_img, enable_ditail, ditail_args = args

        # map args to ditail_args
        ditail_args = DitailArgs()
        ditail_args.src_img, ditail_args.enable_ditail = args[0], args[1]
        for k, v in args[2].items():
            setattr(ditail_args, k, v)
        
        ditail_args = self.replace_empty_args(p, ditail_args)


        # # try replacing forward
        # ldm.modules.diffusionmodules.openaimodel.UNetModel.forward = forward_with_injection
        

        if self.is_ditail_enabled(ditail_args):
            print('!! ditail enabled')
            # overwrite empty args
            
            self.load_inv_model(ditail_args.src_model_name)
            model = shared.sd_model

            latent_save_path = create_path("./extensions/sd-webui-ditail/features")
            self.extract_feature_maps(ditail_args, model, latent_save_path)

            # injected_features = self.load_target_features(ditail_args, latent_save_path)
            # print('!!loaded injectd features', injected_features)
            # print('!! loaded injected features', len(injected_features))
            # p.sd_model.forward = partial(p.sd_model.forward, injected_features=injected_features)
            # UNetModelWithInjection.forward = partial(copy.deepcopy(UNetModelWithInjection.forward), injected_features=injected_features)
            

            # simple_tensor = torch.randn(10, 10)
            # torch.save(simple_tensor, os.path.join(latent_save_path, 'simple_tensor.pt'))
            # with safe.Extra(self.extra_handler):
            # # x = torch.load('model.pt')
            #     loaded_tensor = torch.load(os.path.join(latent_save_path, 'simple_tensor.pt'))
            #     print('!! loaded tensor', loaded_tensor)

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
    
    def extract_feature_maps(self, ditail_args: DitailArgs, model, latent_save_path, seed=42):
        extracter = ExtractLatent(latent_save_path)
        latents, z_enc = extracter.extract(
                init_image=ditail_args.src_img,
                model=model,
                positive_prompt=ditail_args.inv_prompt,
                negative_prompt=ditail_args.inv_negative_prompt,
                alpha=ditail_args.ditail_alpha,
                beta=ditail_args.ditail_beta,
                seed=42,
                ddim_inversion_steps=ditail_args.inv_steps,
            )
        
        # with open(os.path.join(latent_save_path, 'latents_tmp.pkl'), 'wb') as f:
        #     pickle.dump(latents, f)
        # save z_enc
        torch.save(z_enc, os.path.join(latent_save_path, 'z_enc.pt'))

    def load_target_features(self, ditail_args: DitailArgs, latent_save_path):
        self_attn_output_block_indices = [4,5,6,7,8,9,10,11]
        out_layers_output_block_indices = [4]
        output_block_self_attn_map_injection_thresholds = [ditail_args.inv_steps // 2] * len(self_attn_output_block_indices)
        feature_injection_thresholds = [int(ditail_args.inv_steps * 1.0)] # TODO: should be one of the arguments, change later
        target_features = []

        T = 999
        c = T // ditail_args.inv_steps

        iterator = tqdm(reversed(range(0, T, c)), desc="DDIM inversion", total = ditail_args.inv_steps)
        
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
    
    def extra_handler(module, name):
        print(f'!!extra handler for {name} in {module} is called')
        if module == 'torch' and name == 'Tensor':
            return torch.Tensor
        return None


        


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


# def new_forward(self, x, timesteps=None, context=None, y=None, injected_features=None, **kwargs):
#     return UNetModelWithInjection.forward(self, x=x, timesteps=timesteps, context=context, y=y, injected_features=injected_features, **kwargs)