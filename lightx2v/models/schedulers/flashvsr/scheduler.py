import torch

from lightx2v.models.schedulers.scheduler import BaseScheduler
from lightx2v_platform.base.global_var import AI_DEVICE


class FlashVSRScheduler(BaseScheduler):
    def __init__(self, config):
        super().__init__(config)
        # FlashVSR tiny follows one denoise step in the original inference script.
        self.infer_steps = 1
        self.caching_records = [True]
        self.generator = None

    def prepare(self, seed, latent_shape, image_encoder_output=None):
        self.latents = None
        self.step_index = 0
        self.generator = torch.Generator(device=AI_DEVICE)
        self.generator.manual_seed(int(seed))

    def step_pre(self, step_index):
        self.step_index = step_index

    def step_post(self):
        pass
