import torch

from lightx2v.models.networks.flashvsr.infer.post_infer import FlashVSRPostInfer
from lightx2v.models.networks.flashvsr.infer.pre_infer import FlashVSRPreInfer
from lightx2v.models.networks.flashvsr.infer.transformer_infer import FlashVSRTransformerInfer
from lightx2v.models.networks.flashvsr.weights.post_weights import FlashVSRPostWeights
from lightx2v.models.networks.flashvsr.weights.pre_weights import FlashVSRPreWeights
from lightx2v.models.networks.flashvsr.weights.transformer_weights import FlashVSRTransformerWeights
from lightx2v.utils.envs import GET_DTYPE


class FlashVSRModel:
    """FlashVSR model with split pre/transformer/post infer and weights."""

    pre_weight_class = FlashVSRPreWeights
    transformer_weight_class = FlashVSRTransformerWeights
    post_weight_class = FlashVSRPostWeights

    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.dtype = GET_DTYPE()

        self._init_infer_class()
        self._init_weights()
        self._init_infer()
        self.scheduler = None

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler
        self.pre_infer.set_scheduler(scheduler)
        self.transformer_infer.set_scheduler(scheduler)
        self.post_infer.set_scheduler(scheduler)

    def _init_infer_class(self):
        self.pre_infer_class = FlashVSRPreInfer
        self.transformer_infer_class = FlashVSRTransformerInfer
        self.post_infer_class = FlashVSRPostInfer

    def _init_weights(self):
        self.pre_weight = self.pre_weight_class(self.config, self.device, self.dtype)
        self.transformer_weights = self.transformer_weight_class(self.config, self.device, self.dtype)
        self.post_weight = self.post_weight_class(self.config, self.device, self.dtype)
        self.transformer_weights.bind_modules(self.pre_weight, self.post_weight)

    def _init_infer(self):
        self.pre_infer = self.pre_infer_class(self.config)
        self.transformer_infer = self.transformer_infer_class(self.config)
        self.post_infer = self.post_infer_class(self.config)

    @torch.no_grad()
    def infer(self, inputs):
        lq_video = inputs["lq_video"]
        scale = inputs.get("sr_ratio", self.config.get("sr_ratio", 2.0))
        seed = inputs.get("seed", 42)

        pre_out = self.pre_infer.infer(lq_video=lq_video, scale=scale)
        pred = self.transformer_infer.infer(
            weights=self.transformer_weights,
            pre_infer_out=pre_out,
            seed=seed,
        )
        latents = self.post_infer.infer(pred)

        if self.scheduler is not None:
            self.scheduler.latents = latents
        return latents
