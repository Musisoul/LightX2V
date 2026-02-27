class FlashVSRPostInfer:
    def __init__(self, config):
        self.config = config

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, video):
        # FlashVSR output: [C, T, H, W] in [-1, 1]
        # LightX2V VAE output convention: [B, C, T, H, W] in [-1, 1]
        if video.ndim != 4:
            raise ValueError(f"Expected FlashVSR output shape [C,T,H,W], got: {tuple(video.shape)}")
        return video.unsqueeze(0)
