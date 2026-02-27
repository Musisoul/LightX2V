import os

from loguru import logger

try:
    from diffsynth import FlashVSRTinyPipeline, ModelManager
except ImportError:
    FlashVSRTinyPipeline = None
    ModelManager = None


class FlashVSRTransformerWeights:
    def __init__(self, config, device, dtype):
        self.config = config
        self.model_path = config["model_path"]
        self.device = device
        self.dtype = dtype
        self.pipeline = None

        if ModelManager is None or FlashVSRTinyPipeline is None:
            raise ImportError("FlashVSR requires `diffsynth`. Please install it before using model_cls=flashvsr.")

        ckpt_name = config.get("flashvsr_diffusion_ckpt", "diffusion_pytorch_model_streaming_dmd.safetensors")
        ckpt_path = os.path.join(self.model_path, ckpt_name)
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"FlashVSR diffusion checkpoint not found: {ckpt_path}")

        model_manager = ModelManager(torch_dtype=self.dtype, device="cpu")
        model_manager.load_models([ckpt_path])
        self.pipeline = FlashVSRTinyPipeline.from_model_manager(model_manager, device=str(self.device))
        logger.info(f"[FlashVSR] Loaded diffusion backbone from {ckpt_path}")

    def bind_modules(self, pre_weights, post_weights):
        self.pipeline.denoising_model().LQ_proj_in = pre_weights.lq_proj
        self.pipeline.TCDecoder = post_weights.tc_decoder

        self.pipeline.to(str(self.device))
        if hasattr(self.pipeline, "enable_vram_management"):
            self.pipeline.enable_vram_management(num_persistent_param_in_dit=None)
        if hasattr(self.pipeline, "init_cross_kv"):
            self.pipeline.init_cross_kv()
        if hasattr(self.pipeline, "load_models_to_device"):
            self.pipeline.load_models_to_device(["dit", "vae"])

