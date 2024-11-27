from pathlib import Path
import sys
import numpy as np
from PIL import Image

inf="noisy.jpg"
outf="denoised.png"

float2int = lambda img: np.array(np.clip(img* 255, 0, 255), dtype=np.uint8).copy()

import oidn

def denoise(img):
    result = np.zeros_like(img, dtype=np.float32)
    device = oidn.NewDevice()
    oidn.CommitDevice(device)
    filter = oidn.NewFilter(device, "RT")
    oidn.SetSharedFilterImage(filter, "color", img, oidn.FORMAT_FLOAT3, img.shape[1], img.shape[0])
    oidn.SetSharedFilterImage(filter, "output", result, oidn.FORMAT_FLOAT3, img.shape[1], img.shape[0])
    oidn.CommitFilter(filter)
    oidn.ExecuteFilter(filter)
    result = np.array(np.clip(result * 255, 0, 255), dtype=np.uint8)
    oidn.ReleaseFilter(filter)
    oidn.ReleaseDevice(device)
    return result

class OIDN():
    def __init__(self):
        self.device = oidn.NewDevice()
        oidn.CommitDevice(self.device)
        self.filter = oidn.NewFilter(self.device, "RT")
        self.active = True

    def load_file(self,imgpath):
        if not self.active: self.__init__()
        self.img = np.array(Image.open(imgpath), dtype=np.float32) / 255.0

    def load_pil(self,pilimg):
        if not self.active: self.__init__()
        self.img = np.array(pilimg, dtype=np.float32) / 255.0

    def load_array(self,nparray):
        if not self.active: self.__init__()
        self.img = nparray.astype(np.float32) / 255.0

    def filter_once(self):
        self.result = np.zeros_like(self.img, dtype=np.float32)
        oidn.SetSharedFilterImage(self.filter, "output", self.result, oidn.FORMAT_FLOAT3, self.img.shape[1], self.img.shape[0])
        oidn.SetSharedFilterImage(self.filter, "color", self.img, oidn.FORMAT_FLOAT3, self.img.shape[1], self.img.shape[0])
        oidn.CommitFilter(self.filter)
        oidn.ExecuteFilter(self.filter)
        self.img = self.result.copy()

    def apply(self,n=1):
        for i in range(n):
            self.filter_once()
        self.quit()

    def get_array(self):
        return float2int(self.img)

    def get_pil(self):
        return Image.fromarray(self.get_array())

    def save(self,outf):
        self.get_pil().save(outf)

    def quit(self):
        oidn.ReleaseFilter(self.filter)
        oidn.ReleaseDevice(self.device)
        self.active=False

if __name__ == "__main__":
    # OIDN() file to file
    d = OIDN()
    #d.load_file(inf)
    #d.apply(n=3)
    #d.save(outf)
    # OIDN() pipeline
    img = Image.open(inf)
    d.load_array(np.array(img))
    d.apply()
    import pdb; pdb.set_trace()
    print(d.img)
