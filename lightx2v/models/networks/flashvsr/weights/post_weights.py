import os

import torch
from loguru import logger

from lightx2v.models.runners.vsr.utils.TCDecoder import build_tcdecoder


class FlashVSRPostWeights:
    def __init__(self, config, device, dtype):
        self.config = config
        self.model_path = config["model_path"]
        self.device = device
        self.dtype = dtype

        self.tc_decoder = build_tcdecoder(
            new_channels=config.get("flashvsr_tcdecoder_channels", [512, 256, 128, 128]),
            new_latent_channels=config.get("flashvsr_tcdecoder_latent_channels", 16 + 768),
            device=self.device,
            dtype=self.dtype,
        )

        ckpt_name = config.get("flashvsr_tcdecoder_ckpt", "TCDecoder.ckpt")
        ckpt_path = os.path.join(self.model_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"FlashVSR TCDecoder checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location="cpu")
        missing = self.tc_decoder.load_state_dict(state, strict=False)
        self.tc_decoder.to(device=self.device, dtype=self.dtype)
        logger.info(f"[FlashVSR] Loaded TCDecoder from {ckpt_path}, missing_keys={len(missing.missing_keys)}")

