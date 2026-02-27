from dataclasses import dataclass

import torch


@dataclass
class FlashVSRPreInferOutput:
    lq_video: torch.Tensor
    height: int
    width: int
    num_frames: int

