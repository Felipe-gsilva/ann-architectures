from enum import Enum
from typing import Type


Hyperparams: Type = Type[dict[str, int | float | str | bool]]

class AvailableMetrics(Enum): 
    FID = "fid"
    PSNR = "psnr"
    SSIM = "ssim"
    LPIPS = "lpips"
