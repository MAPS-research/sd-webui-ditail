import os

from dataclasses import dataclass
from functools import partial
from types import SimpleNamespace
from typing import Any

import gradio as gr

from ditail import DITAIL, __version__
from ditail.args import ALL_ARGS

class Widgets(SimpleNamespace):
    def tolist(self):
        return [getattr(self, attr) for attr in ALL_ARGS.keys()]

    def attr2name(self, attr):
        idx = ALL_ARGS.attrs.index(attr)
        return ALL_ARGS.names[idx]

    def name2attr(self, name):
        idx = ALL_ARGS.names.index(name)
        return ALL_ARGS.attrs[idx]

@dataclass
class WebuiInfo:
    # sampler_names: list[str]
    checkpoints_list: list[str]
    vae_list: list[str]
    t2i_button: gr.Button
    i2i_button: gr.Button

def elem_id(item_id:str, is_img2img:bool) -> str:
    tab = "img2img" if is_img2img else "txt2img"
    return f"{tab}_ditail_{item_id}"

def state_init(w: Widgets) -> dict[str, Any]:
    return {attr: getattr(w, attr).value for attr in ALL_ARGS.attrs}

def on_widget_change(state: dict, value: Any, *, attr: str):
    if "is_api" in state:
        state = state.copy()
        state.pop("is_api")
    state[attr] = value
    return state

def on_generate_click(state: dict, *values: Any):
    for attr, value in zip(ALL_ARGS.attrs, values):
        state[attr] = value
    state["is_api"] = ()
    return state

def ditailui(
    is_img2img: bool,
    webui_info: WebuiInfo,
):
    
    states = []
    infotext_fields = []
    w = Widgets()
    eid = partial(elem_id, is_img2img=is_img2img)

    with gr.Accordion(DITAIL, open=False, elem_id=eid("main_accordion")):
        cont_image = gr.Image(
                            label="content image",
                            value='./extensions/sd-webui-ditail/placeholder_imgs/Cocktail.jpg',
                            source="upload",
                            # brush_radius=20,
                            mirror_webcam=False,
                            type="pil",
                            # tool="sketch",
                            elem_id=eid("cont_image"),
                            visible=not is_img2img
                            # brush_color=shared.opts.img2img_inpaint_mask_brush_color
                            # if hasattr(
                            #     shared.opts, "img2img_inpaint_mask_brush_color"
                            # )
                            # else None,
                        ) 

        with gr.Row():
            with gr.Column():
                ditail_enable = gr.Checkbox(
                    label = "Enable Ditail",
                    value=False,
                    visible=True,
                    elem_id=eid("enable_ditail")
                )

            with gr.Column():
                gr.Markdown(
                    f"v{__version__}",
                    elem_id=eid("version"),
                )
        
        infotext_fields.append((ditail_enable, "Ditail enabled"))

        with gr.Row():
            w.src_model_name = gr.Dropdown(
                choices=webui_info.checkpoints_list,
                label=w.attr2name("src_model_name"),
                elem_id=eid("src_model_name"),
                visible=True
            )

            w.src_vae_name = gr.Dropdown(
                choices=webui_info.vae_list,
                label=w.attr2name("src_vae_name"),
                elem_id=eid("src_vae_name"),
                visible=True
            )

        with gr.Group():

            with gr.Row():
                w.ditail_alpha = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=5.0,
                    step=0.1,
                    label=w.attr2name("ditail_alpha"),
                    elem_id=eid("ditail_alpha"),
                    interactive=True,
                    visible=True
                )
            
                w.ditail_beta = gr.Slider(
                    minimum=0.0,
                    maximum=10.0,
                    value=0.5,
                    step=0.1,
                    label=w.attr2name("ditail_beta"),
                    elem_id=eid("ditail_beta"),
                    interactive=True,
                    visible=True
                )


        with gr.Accordion("Extra Options", open=False, elem_id=eid("extra_options_accordion")):
            w.inv_prompt = gr.Textbox(
                label=w.attr2name("inv_prompt"),
                show_label=False,
                lines=3,
                placeholder=w.attr2name("inv_prompt") + "\nMain prompt will be used if left empty",
                elem_id=eid("inv_prompt"),
            )

            w.inv_negative_prompt = gr.Textbox(
                label=w.attr2name("inv_negative_prompt"),
                show_label=False,
                lines=2,
                placeholder=w.attr2name("inv_negative_prompt") + "\nMain negative prompt will be used if left empty",
                elem_id=eid("inv_negative_prompt"),
            )

            with gr.Row():
                w.conv_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.8,
                    step=0.1,
                    label=w.attr2name("conv_ratio"),
                    elem_id=eid("conv_ratio"),
                    interactive=True,
                    visible=True
                )

                w.attn_ratio = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.5,
                    step=0.1,
                    label=w.attr2name("attn_ratio"),
                    elem_id=eid("attn_ratio"),
                    interactive=True,
                    visible=True
                )

        state = gr.State(lambda: state_init(w))

        for attr in ALL_ARGS.attrs:
            widget = getattr(w, attr)
            on_change = partial(on_widget_change, attr=attr)
            widget.change(on_change, inputs=[state, widget], outputs=state, queue=False)

    infotext_fields.extend([
        (getattr(w, attr), name) for attr, name in ALL_ARGS
    ])

    states.append(state)
    
    components = [cont_image, ditail_enable, *states]
    return components, infotext_fields
        

        


    