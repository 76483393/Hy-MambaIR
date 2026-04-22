"""LPIPS metric implementation used by the staged Hy-MambaIR repository."""

import numpy as np
import torch

from basicsr.utils.registry import METRIC_REGISTRY

_lpips_model = None
_lpips_available = None


def _check_lpips_availability():
    global _lpips_available
    if _lpips_available is None:
        try:
            import lpips  # noqa: F401

            _lpips_available = True
        except ImportError:
            _lpips_available = False
    return _lpips_available


def _get_lpips_model():
    global _lpips_model
    if not _check_lpips_availability():
        return None
    if _lpips_model is None:
        import lpips

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _lpips_model = lpips.LPIPS(net="alex").to(device)
    return _lpips_model


@METRIC_REGISTRY.register()
def calculate_lpips_hymambair(img, img2, crop_border=0, test_y_channel=False, **kwargs):
    if not _check_lpips_availability():
        raise RuntimeError(
            "LPIPS is required for calculate_lpips_hymambair but is not installed."
        )
    try:
        lpips_model = _get_lpips_model()
        if lpips_model is None:
            raise RuntimeError("Failed to initialize LPIPS model.")

        img = img.astype(np.float32)
        img2 = img2.astype(np.float32)

        if img.max() > 1.0:
            img = img / 255.0
        if img2.max() > 1.0:
            img2 = img2 / 255.0

        if img.ndim == 2:
            img = img[:, :, None]
        if img2.ndim == 2:
            img2 = img2[:, :, None]

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

        if test_y_channel and img.shape[2] == 3:
            img = rgb2ycbcr(img, y_only=True)[:, :, None]
            img2 = rgb2ycbcr(img2, y_only=True)[:, :, None]

        img = np.clip(img, 0, 1)
        img2 = np.clip(img2, 0, 1)
        img_tensor = (
            torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        )
        img2_tensor = (
            torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() * 2.0 - 1.0
        )

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat(1, 3, 1, 1)
            img2_tensor = img2_tensor.repeat(1, 3, 1, 1)

        device = next(lpips_model.parameters()).device
        img_tensor = img_tensor.to(device)
        img2_tensor = img2_tensor.to(device)

        with torch.no_grad():
            return float(lpips_model(img_tensor, img2_tensor).item())
    except Exception as exc:
        raise RuntimeError(f"LPIPS calculation failed: {exc}") from exc


def rgb2ycbcr(img, y_only=False):
    img_type = img.dtype
    img = img.astype(np.float32)
    if y_only:
        out_img = np.dot(img, [0.299, 0.587, 0.114])
    else:
        out_img = np.matmul(
            img,
            [
                [0.299, 0.587, 0.114],
                [-0.168736, -0.331264, 0.5],
                [0.5, -0.418688, -0.081312],
            ],
        ) + [0, 0.5, 0.5]
    return out_img.astype(img_type)


# Backward-compatible alias for local use. This alias is intentionally not
# registry-registered, so that the public repo does not collide with any
# externally installed BasicSR metric of the same name.
calculate_lpips = calculate_lpips_hymambair
