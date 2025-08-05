import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data
import torchvision.transforms.functional
from PIL import Image

class RegData_COLOR(torch.utils.data.Dataset):
    """
    Load dataset with infrared folder path and visible folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, it_folder: pathlib.Path):
        super().__init__()

        # gain images list
        self.ir_list   = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.it_list   = [x for x in sorted(it_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        it_path = self.it_list[index]

        assert (ir_path.name)[:9] == (it_path.name)[:9], f"Mismatch ir:{ir_path.name} vi:{it_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_COLOR, unsqueeze=False)
        it = self.imread(path=it_path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False)

        return (ir, it), (str(ir_path), str(it_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE, unsqueeze=False):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts.unsqueeze(0) if unsqueeze else im_ts
