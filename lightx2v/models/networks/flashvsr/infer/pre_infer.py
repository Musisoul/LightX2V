import torch
import torch.nn.functional as F

from .module_io import FlashVSRPreInferOutput


def largest_8n1_leq(n):
    return 0 if n < 1 else ((n - 1) // 8) * 8 + 1


def compute_scaled_and_target_dims(w0: int, h0: int, scale: float = 4.0, multiple: int = 128):
    if w0 <= 0 or h0 <= 0:
        raise ValueError("Invalid original size")
    if scale <= 0:
        raise ValueError("scale must be > 0")

    scaled_w = int(round(w0 * scale))
    scaled_h = int(round(h0 * scale))
    target_w = (scaled_w // multiple) * multiple
    target_h = (scaled_h // multiple) * multiple

    if target_w == 0 or target_h == 0:
        raise ValueError(f"Scaled size too small ({scaled_w}x{scaled_h}) for multiple={multiple}. Increase scale (got {scale}).")

    return target_w, target_h


class FlashVSRPreInfer:
    def __init__(self, config):
        self.config = config
        self.default_scale = config.get("sr_ratio", 2.0)
        self.multiple = config.get("flashvsr_multiple", 128)

    def set_scheduler(self, scheduler):
        self.scheduler = scheduler

    def infer(self, lq_video: torch.Tensor, scale: float | None = None):
        # input: [T, H, W, C], value range [0, 1]
        scale = self.default_scale if scale is None else scale
        if lq_video.ndim != 4:
            raise ValueError(f"Expected lq_video shape [T,H,W,C], got: {tuple(lq_video.shape)}")

        input_video = lq_video.to(dtype=torch.float32)
        total, h0, w0, _ = input_video.shape

        target_w, target_h = compute_scaled_and_target_dims(w0, h0, scale=scale, multiple=self.multiple)

        idx = list(range(total)) + [total - 1] * 4
        num_frames = largest_8n1_leq(len(idx))
        if num_frames == 0:
            raise RuntimeError(f"Not enough frames after padding. Got {len(idx)}.")
        idx = idx[:num_frames]

        frames = input_video[idx].permute(0, 3, 1, 2) * 2.0 - 1.0  # [F,3,H,W], [-1,1]
        frames = F.interpolate(frames, scale_factor=scale, mode="bicubic", align_corners=False)

        scaled_h, scaled_w = frames.shape[-2], frames.shape[-1]
        left = (scaled_w - target_w) // 2
        top = (scaled_h - target_h) // 2
        frames = frames[:, :, top : top + target_h, left : left + target_w]

        # [1, C, F, H, W]
        lq_video = frames.permute(1, 0, 2, 3).unsqueeze(0)

        return FlashVSRPreInferOutput(
            lq_video=lq_video,
            height=target_h,
            width=target_w,
            num_frames=num_frames,
        )

