class FlashVSRTransformerInfer:
    def __init__(self, config):
        self.config = config
        self.cfg_scale = config.get("flashvsr_cfg_scale", 1.0)
        self.num_inference_steps = config.get("flashvsr_num_inference_steps", 1)
        self.sparse_ratio = config.get("flashvsr_sparse_ratio", 2.0)
        self.kv_ratio = config.get("flashvsr_kv_ratio", 3.0)
        self.local_range = config.get("flashvsr_local_range", 11)
        self.color_fix = config.get("flashvsr_color_fix", True)
        self.is_full_block = config.get("flashvsr_is_full_block", False)
        self.if_buffer = config.get("flashvsr_if_buffer", True)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, weights, pre_infer_out, seed):
        lq_video = pre_infer_out.lq_video
        lq_proj = weights.pipeline.denoising_model().LQ_proj_in
        proj_param = next(lq_proj.parameters(), None)
        if proj_param is not None:
            lq_video = lq_video.to(device=proj_param.device, dtype=proj_param.dtype)

        topk_ratio = self.sparse_ratio * 768 * 1280 / (pre_infer_out.height * pre_infer_out.width)
        return weights.pipeline(
            prompt="",
            negative_prompt="",
            cfg_scale=self.cfg_scale,
            num_inference_steps=self.num_inference_steps,
            seed=int(seed),
            LQ_video=lq_video,
            num_frames=pre_infer_out.num_frames,
            height=pre_infer_out.height,
            width=pre_infer_out.width,
            is_full_block=self.is_full_block,
            if_buffer=self.if_buffer,
            topk_ratio=topk_ratio,
            kv_ratio=self.kv_ratio,
            local_range=self.local_range,
            color_fix=self.color_fix,
        )
