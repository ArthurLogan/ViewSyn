import numpy as np
import os
import torch
from time import time
import cv2
import matplotlib.pyplot as plt


def bilinear_wrapper(images: torch.Tensor, pixels: torch.Tensor):
    """Bilinear Interpolation"""
    _, d, h, w = pixels.shape
    p, _, _, c = images.shape

    x = pixels[0].view(-1)
    y = pixels[1].view(-1)

    x0 = torch.floor(x)
    x1 = x0 + 1
    y0 = torch.floor(y)
    y1 = y0 + 1

    diff_x0 = x - x0
    diff_y0 = y - y0
    diff_x1 = x1 - x
    diff_y1 = y1 - y

    x0 = x0.long()
    x1 = x1.long()
    y0 = y0.long()
    y1 = y1.long()

    x0 = (x0 + w) % w
    y0 = (y0 + h) % h
    x1 = (x1 + w) % w
    y1 = (y1 + h) % h

    if p == d:
        dp = torch.arange(d).view([d, 1]).repeat([1, h * w]).view([-1]).cuda()
    else:
        dp = torch.zeros([d * h * w], dtype=torch.long).cuda()

    value_a = images[dp, y0, x0]
    value_b = images[dp, y0, x1]
    value_c = images[dp, y1, x0]
    value_d = images[dp, y1, x1]

    weight_a = (diff_y1 * diff_x1).view([-1, 1])
    weight_b = (diff_y1 * diff_x0).view([-1, 1])
    weight_c = (diff_y0 * diff_x1).view([-1, 1])
    weight_d = (diff_y0 * diff_x0).view([-1, 1])

    res = value_a * weight_a + value_b * weight_b + value_c * weight_c + value_d * weight_d
    return res.view([d, h, w, c])


def composite(images: torch.Tensor, alphas: torch.Tensor):
    """Composite rgba images to output"""
    d, b, c, h, w = images.shape
    out = 0
    for i in range(d):
        rgb = images[i]
        alp = alphas[i]
        out = rgb * alp + out * (1.0 - alp)
    return out


def remove_output(path: str):
    """Remove last train output"""
    for dir in os.listdir(path):
        dir = os.path.join(path, dir)
        if os.path.isdir(dir):
            for fname in os.listdir(dir):
                os.remove(os.path.join(dir, fname))


def save_image(im: np.ndarray, path: str, norm: bool = False):
    """Output test images"""
    if norm:
        im = (im * 255.0).astype(np.uint8)
    else:
        im = ((im + 1.) / 2. * 255.0).astype(np.uint8)
    if len(im.shape) == 4:
        im = im[0]
    im = np.transpose(im, [1, 2, 0])
    cv2.imwrite(path, im)


def save_plot(im: np.ndarray, path: str, norm: bool = False):
    """Output subplots"""
    im = np.transpose(im, [0, 2, 3, 1])
    if norm:
        im = (im * 255.0).astype(np.uint8)
    else:
        im = ((im + 1.) / 2. * 255.0).astype(np.uint8)
    b, h, w, c = im.shape
    s = 0
    out = np.zeros((b * h + (b - 1) * 10, w, c), dtype=np.uint8)
    for i in range(b):
        out[s:s+h] = im[i]
        s += h + 10
    cv2.imwrite(path, out)


def save_msi(rgb: np.ndarray, wgt: np.ndarray, alp: np.ndarray, path: str):
    """Output msi images"""
    save_plot(rgb, path[:-5] + '_rgb.jpeg')
    save_plot(wgt, path[:-5] + '_wgt.jpeg', True)
    save_plot(alp, path[:-5] + '_alp.jpeg', True)


def save_ssw(im: np.ndarray, path: str):
    """Output sphere sweep images"""
    b, d2, _, h, w = im.shape
    d = d2 // 2
    ref = im[0, :d]
    src = im[0, d:]
    save_plot(np.stack([ref[0], src[0]]), path[:-5] + '_far.jpeg')
    save_plot(np.stack([ref[-1], src[-1]]), path[:-5] + '_ner.jpeg')


def save_input(inp: torch.Tensor, dep: torch.Tensor, path: str):
    """Output image after ods projection"""
    inp = inp.cpu().numpy()
    dep = dep.cpu().numpy()
    b, _, h, w = inp.shape
    d = dep.shape[0]
    inp = inp.reshape([b, 2 * d, 3, h, w])
    save_ssw(inp, path)


def save_origin(ref: torch.Tensor, src: torch.Tensor, path: str):
    """Output original image"""
    ref = ref.cpu().numpy()
    src = src.cpu().numpy()
    ori = np.concatenate([ref, src], 0).transpose([0, 3, 1, 2])
    save_plot(ori, path)


def save_loss(loss: list, path: str):
    """Output train loss & test loss"""
    iter = [i + 1 for i in range(len(loss))]
    plt.plot(iter, loss)
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.savefig(path)
    plt.close()


def calc_psnr(im: np.ndarray, gt: np.ndarray):
    """Output PSNR"""
    b, c, h, w = im.shape
    im = np.transpose(im, [0, 2, 3, 1])
    gt = np.transpose(gt, [0, 2, 3, 1])
    im = ((im + 1.) / 2. * 255.0).astype(np.uint8)
    gt = ((gt + 1.) / 2. * 255.0).astype(np.uint8)
    df = gt.astype(np.float32) - im.astype(np.float32)
    df = np.mean(np.sum(df * df, axis=(1, 2, 3)) / (c * h * w))
    psnr = 10 * np.log10(255 * 255 / df)
    return psnr
