# Imports:
import os
import logging
import comfy.model_management as mm
from comfy.utils import ProgressBar
from .inference import OIDN

import torch
import numpy as np
from PIL import Image

# Logging configuration:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)
d = OIDN()


def batch_apply(img_batch,f,progress,permute=False):
    """ ARGS:
            img_batch = tensor [batch,x,y,c], 
            permute = False 
                f = lambda tensor[c,x,y]) -> tensor[c,x,y]
            permute = True
                f = lambda tensor[x,y,c]) -> tensor[x,y,c]
        RETURN:
           tensor[batch,x,y,c]
    """
    results = []
    for img in img_batch:
        if permute: results.append( f(img.permute(2, 0, 1) ).permute(1,2,0) )
        else: results.append( f(img) )
    return torch.stack(results)



# ComnfyUI: Node definitions
class denoiseOIDN:
    @classmethod
    def INPUT_TYPES(s):
        return {
        "required": {
            "img_batch": ("IMAGE", {"tooltip": "Provide an image to be denoised"}),
        }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("denoised_image",)
    FUNCTION = "infer_batch"
    CATEGORY = "denoising"

    def infer_batch(self, img_batch ):
        # Empty cache
        mm.soft_empty_cache()

        pbar = ProgressBar(len(img_batch))

        out = batch_apply(img_batch,self.f, pbar)

        return (out,)

    def f(self,imgtensor):
        """tensor[c,x,y]) -> tensor[c,x,y]"""
        imgarray = imgtensor.cpu().numpy()
        d.__init__()
        d.img = imgarray.copy()
        d.apply(n=1)
        out = d.img
        return torch.tensor(out)

