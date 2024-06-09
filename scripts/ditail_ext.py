import contextlib
import os
import sys
import platform
from functools import partial
from textwrap import dedent
from typing import TYPE_CHECKING, Any, NamedTuple

import gradio as gr

import modules
from modules import scripts
from modules import script_callbacks
from modules import sd_models
from modules import shared
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button
from modules.sd_samplers import all_samplers
from modules.shared import cmd_opts, opts, state

from ditail import (
    DITAIL,
    __version__,
)
from ditail.args import ALL_ARGS, DitailArgs
from ditail.ui import WebuiInfo, ditailui

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
        self.img2img_image = None
    
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

    
    def is_ditail_enabled(self, *args) -> bool: 
        src_image = args[0]
        ditail_enabled = args[1]
        ditail_args = [a for a in args if isinstance(a, dict)]

        if not args or not ditail_args:
            message = f"""
            !! Ditail: Invalid arguments detected.
               input: {args!r}
               Ditail disabled.
            """
            print(dedent(message), file=sys.stderr)
            return False

        if not src_image:
            message = f"""
            !! Ditail: No source image detected.
               Ditail disabled.
            """
            print(dedent(message), file=sys.stderr)
            return False

        return ditail_enabled

    def process(self, p, *args):

        print('!! check out p', dir(p))

        # print('!! current checkpoint', p.sd_model)
        # print('!! check out p in process', p, type(p))
        # # print(p.prompt)
        print('!! check out args in process', args, type(args))
        src_img, enable_ditail, ditail_args = args
        if self.is_ditail_enabled(*args):
            print('!! ditail enabled')
            self.load_inv_model(ditail_args['src_model_name'])
        else:
            print('!! ditail disabled')
        


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


 



