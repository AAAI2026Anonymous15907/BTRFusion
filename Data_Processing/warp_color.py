import sys

sys.path.append("..")

import argparse
import pathlib
import warnings
import statistics
import time

import cv2
import numpy as np
import kornia
import torch.backends.cudnn
import torch.cuda
import torch.utils.data
from torch import Tensor
from tqdm import tqdm
import os

from data import RegData_COLOR

import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters.kernels import get_gaussian_kernel2d
import torch.nn.functional as nnf
from PIL import Image

class AffineTransform(nn.Module):
    """
    Add random affine transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, degrees=0, translate=0.1):
        super(AffineTransform, self).__init__()
        self.trs = kornia.augmentation.RandomAffine(degrees, (translate, translate), return_transform=True, p=1)

    def forward(self, input):
        # image shape
        batch_size, _, height, weight = input.shape
        # affine transform
        warped, affine_param = self.trs(input)  # [batch_size, 3, 3]
        affine_theta = self.param_to_theta(affine_param, weight, height)  # [batch_size, 2, 3]
        # base + disp = grid -> disp = grid - base
        base = kornia.utils.create_meshgrid(height, weight, device=input.device).to(input.dtype)
        grid = F.affine_grid(affine_theta, size=input.size(), align_corners=False)  # [batch_size, height, weight, 2]
        disp = grid - base
        return warped, -disp

    @staticmethod
    def param_to_theta(param, weight, height):
        """
        Convert affine transform matrix to theta in F.affine_grid
        :param param: affine transform matrix [batch_size, 3, 3]
        :param weight: image weight
        :param height: image height
        :return: theta in F.affine_grid [batch_size, 2, 3]
        """

        theta = torch.zeros(size=(param.shape[0], 2, 3)).to(param.device)  # [batch_size, 2, 3]

        theta[:, 0, 0] = param[:, 0, 0]
        theta[:, 0, 1] = param[:, 0, 1] * height / weight
        theta[:, 0, 2] = param[:, 0, 2] * 2 / weight + param[:, 0, 0] + param[:, 0, 1] - 1
        theta[:, 1, 0] = param[:, 1, 0] * weight / height
        theta[:, 1, 1] = param[:, 1, 1]
        theta[:, 1, 2] = param[:, 1, 2] * 2 / height + param[:, 1, 0] + param[:, 1, 1] - 1

        return theta
class ElasticTransform(nn.Module):
    """
    Add random elastic transforms to a tensor image.
    Most functions are obtained from Kornia, difference:
    - gain the disp grid
    - no p and same_on_batch
    """

    def __init__(self, kernel_size: int = 63, sigma: float = 32, align_corners: bool = False, mode: str = "bilinear"):
        super(ElasticTransform, self).__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.align_corners = align_corners
        self.mode = mode

    def forward(self, input):
        # generate noise
        batch_size, _, height, weight = input.shape
        noise = torch.rand(batch_size, 2, height, weight) * 2 - 1
        # elastic transform
        warped, disp = self.elastic_transform2d(input, noise)
        return warped, disp

    def elastic_transform2d(self, image: torch.Tensor, noise: torch.Tensor):
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Input image is not torch.Tensor. Got {type(image)}")

        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"Input noise is not torch.Tensor. Got {type(noise)}")

        if not len(image.shape) == 4:
            raise ValueError(f"Invalid image shape, we expect BxCxHxW. Got: {image.shape}")

        if not len(noise.shape) == 4 or noise.shape[1] != 2:
            raise ValueError(f"Invalid noise shape, we expect Bx2xHxW. Got: {noise.shape}")

        # unpack hyper parameters
        kernel_size = self.kernel_size
        sigma = self.sigma
        align_corners = self.align_corners
        mode = self.mode
        device = image.device

        # Get Gaussian kernel for 'y' and 'x' displacement
        kernel_x: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]
        kernel_y: torch.Tensor = get_gaussian_kernel2d((kernel_size, kernel_size), (sigma, sigma))[None]

        # Convolve over a random displacement matrix and scale them with 'alpha'
        disp_x: torch.Tensor = noise[:, :1].to(device)
        disp_y: torch.Tensor = noise[:, 1:].to(device)

        disp_x = kornia.filters.filter2d(disp_x, kernel=kernel_y, border_type="constant")
        disp_y = kornia.filters.filter2d(disp_y, kernel=kernel_x, border_type="constant")

        # stack and normalize displacement
        disp = torch.cat([disp_x, disp_y], dim=1).permute(0, 2, 3, 1)

        # Warp image based on displacement matrix
        b, c, h, w = image.shape
        grid = kornia.utils.create_meshgrid(h, w, device=image.device).to(image.dtype)
        warped = F.grid_sample(image, (grid + disp).clamp(-1, 1), align_corners=align_corners, mode=mode)

        return warped, disp

def hyper_args():
    """
    get hyper parameters from args
    """
    parser = argparse.ArgumentParser(description='RegNet Net eval process')
    parser.add_argument('--ir', default='./datasets/PET-MRI/PET', type=pathlib.Path)
    parser.add_argument('--it', default='./datasets/PET-MRI/MRI', type=pathlib.Path)
    parser.add_argument('--dst',  default='./results/PET-MRI/warp/', help='fuse image save folder', type=pathlib.Path)
    parser.add_argument("--cuda", action="store_false", help="Use cuda?")
    args = parser.parse_args()
    return args

def main(args):

    cuda = args.cuda
    if cuda and torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        raise Exception("No GPU found...")
    torch.backends.cudnn.benchmark = True

    print("===> Loading datasets")
    data = RegData_COLOR(args.ir, args.it)
    test_data_loader = torch.utils.data.DataLoader(data, 1, True, pin_memory=True)

    print("===> Building deformation")
    affine = AffineTransform(degrees=(-10, 10), translate=0.05)
    elastic = ElasticTransform(kernel_size=101, sigma=16)
    print("===> Starting Testing")
    test(test_data_loader, args.dst, elastic, affine)

def test(test_data_loader, dst, elastic, affine):

    tqdm_loader = tqdm(test_data_loader, disable=True)

    for (ir, it), (ir_path, it_path) in tqdm_loader:
        name, ext = os.path.splitext(os.path.basename(ir_path[0]))
        file_name = name + ext
        ir = ir.cuda()
        it = it.cuda()

        print('------------')
        ir_affine, affine_disp = affine(ir)
        ir_elastic, elastic_disp = elastic(ir_affine)
        disp = affine_disp + elastic_disp
        ir_warp = ir_elastic

        ir_warp.detach_()
        print('------------')
        # TODO: save registrated images
        imsave(ir_warp, dst, file_name)

    pass

def _draw_grid(im_cv, grid_size: int = 24):
    im_gd_cv = np.full_like(im_cv, 255.0)
    im_gd_cv = cv2.cvtColor(im_gd_cv, cv2.COLOR_GRAY2BGR)

    height, width = im_cv.shape
    color = (0, 0, 255)
    for x in range(0, width - 1, grid_size):
        cv2.line(im_gd_cv, (x, 0), (x, height), color, 1, 1) # (0, 0, 0)
    for y in range(0, height - 1, grid_size):
        cv2.line(im_gd_cv, (0, y), (width, y), color, 1, 1)
    im_gd_ts = kornia.utils.image_to_tensor(im_gd_cv / 255.).type(torch.FloatTensor).cuda()
    return im_gd_ts

def imsave(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    save images to path
    :param im_s: image(s)
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """

    im_s = im_s if type(im_s) == list else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = kornia.utils.tensor_to_image(im_ts) * 255.
        cv2.imwrite(str(p), im_cv)

def imsave_gray(im_s: [Tensor], dst: pathlib.Path, im_name: str = ''):
    """
    Save grayscale images to path
    :param im_s: image(s), shape should be [1, H, W] or [H, W]
    :param dst: if one image: path; if multiple images: folder path
    :param im_name: name of image
    """
    im_s = im_s if isinstance(im_s, list) else [im_s]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]

    for im_ts, p in zip(im_s, dst):
        im_ts = im_ts.squeeze().cpu()  # shape: [H, W]
        p.parent.mkdir(parents=True, exist_ok=True)
        im_np = im_ts.numpy()
        im_np = (im_np * 255).clip(0, 255).astype('uint8')  # ensure proper scaling and dtype
        cv2.imwrite(str(p), im_np)

def save_flow(flow: [Tensor], dst: pathlib.Path, im_name: str = ''):
    rgb_flow = flow2rgb(flow, max_value=None) # (3, 512, 512) type; numpy.ndarray
    im_s = rgb_flow if type(rgb_flow) == list else [rgb_flow]
    dst = [dst / str(i + 1).zfill(3) / im_name for i in range(len(im_s))] if len(im_s) != 1 else [dst / im_name]
    for im_ts, p in zip(im_s, dst):
        p.parent.mkdir(parents=True, exist_ok=True)
        im_cv = (im_ts * 255).astype(np.uint8).transpose(1, 2, 0)
        cv2.imwrite(str(p), im_cv)

def flow2rgb(flow_map: [Tensor], max_value: None):
    flow_map_np = flow_map.squeeze().detach().cpu().numpy()
    _, h, w = flow_map_np.shape
    rgb_map = np.ones((3, h, w)).astype(np.float32)
    if max_value is not None:
        normalized_flow_map = flow_map_np / max_value
    else:
        normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    rgb_map[0] += normalized_flow_map[0]
    rgb_map[1] -= 0.5 * (normalized_flow_map[0] + normalized_flow_map[1])
    rgb_map[2] += normalized_flow_map[1]
    rgb_flow = rgb_map.clip(0, 1)
    return rgb_flow

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    args = hyper_args()
    main(args)
