import torch
import torch.nn as nn
from einops import rearrange

from lightx2v.models.runners.vsr.utils.utils import CACHE_T, CausalConv3d, PixelShuffle3d, RMS_norm


class CausalLQ4xProj(nn.Module):
    """Causal low-quality video projector used by FlashVSR v1.1 tiny."""

    def __init__(self, in_dim, out_dim, layer_num=1):
        super().__init__()
        self.ff = 1
        self.hh = 16
        self.ww = 16
        self.hidden_dim1 = 2048
        self.hidden_dim2 = 3072
        self.layer_num = layer_num

        self.pixel_shuffle = PixelShuffle3d(self.ff, self.hh, self.ww)
        self.conv1 = CausalConv3d(
            in_dim * self.ff * self.hh * self.ww,
            self.hidden_dim1,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm1 = RMS_norm(self.hidden_dim1, images=False)
        self.act1 = nn.SiLU()

        self.conv2 = CausalConv3d(
            self.hidden_dim1,
            self.hidden_dim2,
            (4, 3, 3),
            stride=(2, 1, 1),
            padding=(1, 1, 1),
        )
        self.norm2 = RMS_norm(self.hidden_dim2, images=False)
        self.act2 = nn.SiLU()

        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_dim2, out_dim) for _ in range(layer_num)])
        self.clip_idx = 0
        self.clear_cache()

    def clear_cache(self):
        self.cache = {
            "conv1": None,
            "conv2": None,
        }
        self.clip_idx = 0

    def forward(self, video):
        self.clear_cache()

        t = video.shape[2]
        iter_n = 1 + (t - 1) // 4
        first_frame = video[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
        video = torch.cat([first_frame, video], dim=2)

        out_x = []
        for i in range(iter_n):
            x = self.pixel_shuffle(video[:, :, i * 4 : (i + 1) * 4, :, :])

            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)

            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            if i == 0:
                self.cache["conv2"] = cache2_x
                continue

            x = self.conv2(x, self.cache["conv2"])
            self.cache["conv2"] = cache2_x
            x = self.norm2(x)
            x = self.act2(x)
            out_x.append(x)

        out_x = torch.cat(out_x, dim=2)
        out_x = rearrange(out_x, "b c f h w -> b (f h w) c")
        return [linear(out_x) for linear in self.linear_layers]

    def stream_forward(self, video_clip):
        if self.clip_idx == 0:
            first_frame = video_clip[:, :, :1, :, :].repeat(1, 1, 3, 1, 1)
            video_clip = torch.cat([first_frame, video_clip], dim=2)
            x = self.pixel_shuffle(video_clip)

            cache1_x = x[:, :, -CACHE_T:, :, :].clone()
            x = self.conv1(x, self.cache["conv1"])
            self.cache["conv1"] = cache1_x
            x = self.norm1(x)
            x = self.act1(x)

            cache2_x = x[:, :, -CACHE_T:, :, :].clone()
            self.cache["conv2"] = cache2_x
            self.clip_idx += 1
            return None

        x = self.pixel_shuffle(video_clip)
        cache1_x = x[:, :, -CACHE_T:, :, :].clone()
        x = self.conv1(x, self.cache["conv1"])
        self.cache["conv1"] = cache1_x
        x = self.norm1(x)
        x = self.act1(x)

        cache2_x = x[:, :, -CACHE_T:, :, :].clone()
        x = self.conv2(x, self.cache["conv2"])
        self.cache["conv2"] = cache2_x
        x = self.norm2(x)
        x = self.act2(x)
        out_x = rearrange(x, "b c f h w -> b (f h w) c")
        self.clip_idx += 1
        return [linear(out_x) for linear in self.linear_layers]

