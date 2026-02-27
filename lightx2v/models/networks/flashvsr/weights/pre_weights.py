import os

import torch
from loguru import logger

from lightx2v.models.networks.flashvsr.modules import CausalLQ4xProj


class FlashVSRPreWeights:
    def __init__(self, config, device, dtype):
        self.config = config
        self.model_path = config["model_path"]
        self.device = device
        self.dtype = dtype

        self.lq_proj = CausalLQ4xProj(
            in_dim=config.get("flashvsr_lq_proj_in_dim", 3),
            out_dim=config.get("flashvsr_lq_proj_out_dim", 1536),
            layer_num=config.get("flashvsr_lq_proj_layer_num", 1),
        ).to(device=self.device, dtype=self.dtype)

        ckpt_name = config.get("flashvsr_lq_proj_ckpt", "LQ_proj_in.ckpt")
        ckpt_path = os.path.join(self.model_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"FlashVSR LQ projector checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu")
        self.lq_proj.load_state_dict(state, strict=True)
        self.lq_proj.to(device=self.device, dtype=self.dtype)
        logger.info(f"[FlashVSR] Loaded LQ projector from {ckpt_path}")
