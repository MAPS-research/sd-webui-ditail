from __future__ import annotations

from collections import UserList
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Literal, NamedTuple, Optional

from PIL import Image

import pydantic
from pydantic import (
    BaseModel,
    Extra,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    confloat,
    conint,
    constr,
    validator,
)

class Arg(NamedTuple):
    attr: str
    name: str

class ArgsList(UserList):
    @cached_property
    def attrs(self) -> tuple[str]:
        return tuple(attr for attr, _ in self)
    
    @cached_property
    def names(self) -> tuple[str]:
        return tuple(name for _, name in self)

class DitailArgs(BaseModel, extra=Extra.forbid, arbitrary_types_allowed = True):
    # enable_ditail: bool = False
    src_img: Image.Image = None
    # src_sd_model = None
    src_model_name: str = ""
    src_vae_name: str = ""
    inv_prompt: str = ""
    inv_negative_prompt: str = ""
    inv_steps: int = 1000 # will match the generation steps in main ui later
    inv_sampler_name: str = "DDIM"
    inv_scheduler_name: str = "Automatic"
    ditail_alpha: confloat(ge=0.0, le=10.0) = 3.0
    ditail_beta: confloat(ge=0.0, le=10.0) = 0.5
    conv_ratio: confloat(ge=0.0, le=1.0) = 0.8 # TODO: enable custom input in UI
    attn_ratio: confloat(ge=0.0, le=1.0) = 0.5
    is_api: bool = True

    @validator("is_api", pre=True)
    def is_api_validator(cls, v: Any):  # TODO: check what this is doing
        "tuple is json serializable but cannot be made with json deserialize."
        return type(v) is not tuple
    

_all_args = [
    ("src_model_name", "Source Checkpoint"),
    ("src_vae_name", "Source VAE"),
    ("inv_prompt", "Positive Inversion Prompt"),
    ("inv_negative_prompt", "Negative Inversion Prompt"),
    ("ditail_alpha", "Positive Prompt Scaling (Alpha)"),
    ("ditail_beta", "Negative Prompt Scaling (Beta)"),
    ("conv_ratio", "Convolutional Ratio"),
    ("attn_ratio", "Attention Ratio"),
    ("inv_steps", "Inversion Steps"),
    ("inv_sampler_name", "Inversion Sampler"),
    ("inv_scheduler_name", "Inversion Scheduler"),
]

_args = [Arg(*args) for args in _all_args]
ALL_ARGS = ArgsList(_args)