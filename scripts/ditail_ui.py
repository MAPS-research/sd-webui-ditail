import contextlib

import gradio as gr
from modules import scripts
from modules import script_callbacks
from modules import sd_models
from modules import shared
from modules.ui_components import FormRow
from modules.ui_common import create_refresh_button


def send_text_to_prompt(new_text, old_text, new_neg_text, old_neg_text):
    # if old_text == "":  # if text on the textbox text2img or img2img is empty, return new text
    #     return new_text
    # return old_text + " " + new_text  # else join them together and send it to the textbox
    return new_text, new_neg_text


class ExampleScript(scripts.Script):
    def __init__(self) -> None:
        super().__init__()
        self.enabled = None
        self.image_upload_panel = None
        self.image = None
        self.ditail_alpha = None
        self.ditail_beta = None
        self.sampler_name = None

    def title(self):
        return "Ditail"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        with gr.Accordion("Ditail", open=False):
            with gr.Group(visible=not is_img2img) as self.image_upload_panel:
                self.image = gr.Image(
                                label="content image",
                                source="upload",
                                brush_radius=20,
                                mirror_webcam=False,
                                type="numpy",
                                tool="sketch",
                                elem_id="input_image",
                                elem_classes=["ditail-image"],
                                brush_color=shared.opts.img2img_inpaint_mask_brush_color
                                if hasattr(
                                    shared.opts, "img2img_inpaint_mask_brush_color"
                                )
                                else None,
                            )
            self.enabled = gr.Checkbox(label="Enable", default=False)
            with FormRow(elem_id="ditail_src_model"):
                self.src_model_name = gr.Dropdown(sd_models.checkpoint_tiles(), elem_id="ditail_src_model_name", label="Source Checkpoint")
                create_refresh_button(self.src_model_name, sd_models.list_models, lambda: {"choices": sd_models.checkpoint_tiles()}, "refresh_src_checkpoint")
                print("!! sampler when creating ditail ui", self.sampler_name)
                print("!! src model name when creating ditail ui", self.src_model_name)
                # self.src_model_name.change(fn=self.load_inv_model, inputs=[self.src_model_name, self.sampler_name])
 
            with gr.Row(elem_id="ditail_prompt_weight"):
                self.ditail_alpha = gr.Slider(minimum=0.0, maximum=10.0, value=3.0, step=0.1, label="positive prompt scaling weight (alpha)", elem_id="ditail_alpha", interactive=True)
                self.ditail_beta = gr.Slider(minimum=0.0, maximum=10.0, value=0.5, step=0.1, label="negative prompt scaling weight (beta)", elem_id="ditail_beta", interactive=True)
            # text_to_be_sent = gr.Textbox(label="drop text")
            # negative_text_to_be_sent = gr.Textbox(label="drop negative text")
            # send_text_button = gr.Button(value='send text', variant='primary')
            print('!! device', shared.device)

    def load_inv_model(self, checkpoint_name, sampler_name):

        print('!! shared model', shared.opts.sd_model_checkpoint)
        checkpoint_info = sd_models.get_closet_checkpoint_match(checkpoint_name)
        print("!! loading inv model", checkpoint_info, type(checkpoint_info))
        self.old_model = shared.sd_model
        inv_model = sd_models.reload_model_weights(info = checkpoint_info)
        print("!! inv model loaded", type(inv_model), inv_model.sd_model_checkpoint)
        print("!! new shared model", shared.opts.sd_model_checkpoint)

        print("!! current sampler name", sampler_name)
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


    def after_component(self, component, **kwargs):
        #https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/7456#issuecomment-1414465888 helpfull link
        # Find the text2img textbox component
        if kwargs.get("elem_id") == "txt2img_prompt": #postive prompt textbox
            self.boxx = component
            print('!! boxx', self.boxx)
        # Find the img2img textbox component
        if kwargs.get("elem_id") == "img2img_prompt":  #postive prompt textbox
            self.boxxIMG = component

        #this code below  works aswell, you can send negative prompt text box,provided you change the code a little
        #switch  self.boxx with  self.neg_prompt_boxTXT  and self.boxxIMG with self.neg_prompt_boxIMG

        if kwargs.get("elem_id") == "txt2img_neg_prompt":
            self.neg_prompt_boxTXT = component
        if kwargs.get("elem_id") == "img2img_neg_prompt":
            self.neg_prompt_boxIMG = component

        if kwargs.get("elem_id") == "txt2img_sampling":
            self.sampler_name = component
            print('!! sampler name', self.sampler_name)








