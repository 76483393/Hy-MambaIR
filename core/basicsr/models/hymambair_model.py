"""Hy-MambaIR model wrapper for evaluation and inference.

This module provides the BasicSR-compatible model wrapper used to run the
paper-line checkpoint for patch-wise reconstruction and evaluation.
"""

import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.models.sr_model import SRModel


@MODEL_REGISTRY.register()
class HyMambaIRModel(SRModel):
    """Hy-MambaIR model wrapper for image restoration."""

    def test(self):
        batch_size, C, h, w = self.lq.size()
        split_token_h = h // 200 + 1
        split_token_w = w // 200 + 1
        mod_pad_h, mod_pad_w = 0, 0
        if h % split_token_h != 0:
            mod_pad_h = split_token_h - h % split_token_h
        if w % split_token_w != 0:
            mod_pad_w = split_token_w - w % split_token_w
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), "reflect")
        _, _, H, W = img.size()
        split_h = H // split_token_h
        split_w = W // split_token_w
        shave_h = split_h // 10
        shave_w = split_w // 10
        scale = self.opt.get("scale", 1)
        ral = H // split_h
        row = W // split_w
        slices = []
        for i in range(ral):
            for j in range(row):
                if i == 0 and i == ral - 1:
                    top = slice(i * split_h, (i + 1) * split_h)
                elif i == 0:
                    top = slice(i * split_h, (i + 1) * split_h + shave_h)
                elif i == ral - 1:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h)
                else:
                    top = slice(i * split_h - shave_h, (i + 1) * split_h + shave_h)
                if j == 0 and j == row - 1:
                    left = slice(j * split_w, (j + 1) * split_w)
                elif j == 0:
                    left = slice(j * split_w, (j + 1) * split_w + shave_w)
                elif j == row - 1:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w)
                else:
                    left = slice(j * split_w - shave_w, (j + 1) * split_w + shave_w)
                slices.append((top, left))

        img_chops = [img[..., top, left] for top, left in slices]

        if hasattr(self, "net_g_ema"):
            self.net_g_ema.eval()
            with torch.no_grad():
                outputs = [self.net_g_ema(chop) for chop in img_chops]
        else:
            self.net_g.eval()
            with torch.no_grad():
                outputs = [self.net_g(chop) for chop in img_chops]
            self.net_g.train()

        merged = outputs[0].new_zeros((batch_size, C, H * scale, W * scale))
        for i in range(ral):
            for j in range(row):
                top = slice(i * split_h * scale, (i + 1) * split_h * scale)
                left = slice(j * split_w * scale, (j + 1) * split_w * scale)
                if i == 0:
                    crop_top = slice(0, split_h * scale)
                else:
                    crop_top = slice(shave_h * scale, (shave_h + split_h) * scale)
                if j == 0:
                    crop_left = slice(0, split_w * scale)
                else:
                    crop_left = slice(shave_w * scale, (shave_w + split_w) * scale)
                merged[..., top, left] = outputs[i * row + j][..., crop_top, crop_left]

        self.output = merged[
            :, :, 0 : H * scale - mod_pad_h * scale, 0 : W * scale - mod_pad_w * scale
        ]
