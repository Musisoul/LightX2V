import numpy as np
import torch
from PIL import Image

from lightx2v.models.networks.flashvsr import FlashVSRModel
from lightx2v.models.runners.default_runner import DefaultRunner
from lightx2v.models.schedulers.flashvsr.scheduler import FlashVSRScheduler
from lightx2v.utils.registry_factory import RUNNER_REGISTER
from lightx2v_platform.base.global_var import AI_DEVICE


@RUNNER_REGISTER("flashvsr")
class FlashVSRRunner(DefaultRunner):
    def set_init_device(self):
        # FlashVSR tiny currently relies on CUDA-side diffsynth pipeline.
        self.init_device = torch.device(AI_DEVICE)

    def __init__(self, config):
        super().__init__(config)
        self.run_input_encoder = self._run_input_encoder_local_sr
        self._ori_length = 0

    def init_scheduler(self):
        self.scheduler = FlashVSRScheduler(self.config)

    def load_transformer(self):
        return FlashVSRModel(config=self.config, device=torch.device(AI_DEVICE))

    def load_text_encoder(self):
        return []

    def load_image_encoder(self):
        return None

    def load_vae(self):
        return None, None

    def load_vae_decoder(self):
        return None

    def run_text_encoder(self, input_info):
        return None

    def run_image_encoder(self, img):
        return None

    def _read_sr_input(self):
        if "video_path" in self.input_info.__dataclass_fields__ and self.input_info.video_path:
            from torchvision.io import read_video

            video, _, _ = read_video(self.input_info.video_path, output_format="TCHW")
            if video.numel() == 0:
                raise ValueError(f"Failed to read video from {self.input_info.video_path}")
            video = (video.float() / 255.0).permute(0, 2, 3, 1).to(self.init_device)  # [T,H,W,C]
            return video

        if "image_path" in self.input_info.__dataclass_fields__ and self.input_info.image_path:
            image = Image.open(self.input_info.image_path).convert("RGB")
            image = torch.from_numpy(np.array(image)).float() / 255.0  # [H,W,C]
            image = image.unsqueeze(0).to(self.init_device)  # [1,H,W,C]
            return image

        raise ValueError("SR task requires image_path or video_path")

    def _run_input_encoder_local_sr(self):
        lq_video = self._read_sr_input()
        self._ori_length = lq_video.shape[0]
        self.input_info.latent_shape = [1, lq_video.shape[-1], lq_video.shape[0], lq_video.shape[1], lq_video.shape[2]]

        return {
            "lq_video": lq_video,
            "sr_ratio": float(getattr(self.input_info, "sr_ratio", self.config.get("sr_ratio", 2.0))),
            "seed": int(self.input_info.seed),
            "image_encoder_output": None,
            "text_encoder_output": None,
            "vae_encoder_out": None,
        }

    def run_vae_decoder(self, latents):
        # FlashVSR already outputs RGB video tensor in [-1, 1].
        if self._ori_length > 0 and latents.shape[2] > self._ori_length:
            latents = latents[:, :, : self._ori_length]
        return latents
