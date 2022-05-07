import numpy as np
import torch
import cv2
import os
import json
import shutil
from matplotlib import pyplot as plt


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
    out = images[0]
    for i in range(1, d):
        rgb = images[i]
        alp = alphas[i]
        out = rgb * alp + out * (1.0 - alp)
    return out


def save_image(im: np.ndarray, path: str, norm: bool = False):
    """Output subplots"""
    if len(im.shape) == 3:
        im = np.expand_dims(im, 0)
    im = np.transpose(im, [0, 2, 3, 1])
    if norm:
        im = (im * 255.0).astype(np.uint8)
    else:
        im = ((im + 1.) / 2. * 255.0).astype(np.uint8)
    b, h, w, c = im.shape
    out = np.zeros((b * h + (b - 1) * 10, w, c), dtype=np.uint8)
    s = 0
    for i in range(b):
        out[s:s+h] = im[i]
        s += h + 10
    cv2.imwrite(path, out)


def save_output(output: tuple, path: str):
    """Output valid output"""
    rgb, wgt, alp = output
    save_image(rgb, path[:-5] + '_rgb.jpeg')
    save_image(wgt, path[:-5] + '_wgt.jpeg', norm=True)
    save_image(alp, path[:-5] + '_alp.jpeg', norm=True)


def save_tensor(out: list, path: str):
    """Output tensor"""
    for id, im in enumerate(out):
        im = im.cpu().detach().numpy()[0]
        minv = im.min()
        maxv = im.max()
        im = (im - minv) / (maxv - minv)
        im = np.expand_dims(im[:3], 1)
        save_image(im, path[:-5] + '%d.jpeg' % (id + 1), norm=True)


def save_curve(curves: list, pos: int, path: str):
    """Output train loss & Test loss & Test PSNR"""
    cnames = ['Train', 'Valid', 'PSNR']
    for cname, curve in zip(cnames, curves):
        iter = [pos + i + 1 for i in range(len(curve))]
        plt.plot(iter, curve)
        plt.xlabel('iteration')
        plt.ylabel('value')
        plt.savefig(os.path.join(path, cname + '.jpeg'))
        plt.close()


def prepare_for_train(args):
    """Prepare folder for training"""
    if not os.path.exists(args.checkpoint_dir):
        os.makedirs(args.checkpoint_dir)
    if os.path.exists(args.result_dir):
        shutil.rmtree(args.result_dir)
    os.makedirs(args.infer_dir)
    os.makedirs(args.sphre_dir)
    os.makedirs(args.demon_dir)
    os.makedirs(args.curve_dir)
    json.dump(args.__dict__, open(os.path.join(args.result_dir, 'Args.txt'), 'w'))


def save_batch(batch: dict, path: str):
    """Save Batch Data"""
    ref = batch['ref'].cpu().detach().numpy().transpose([0, 3, 1, 2])
    src = batch['src'].cpu().detach().numpy().transpose([0, 3, 1, 2])
    tgt = batch['tgt'].cpu().detach().numpy().transpose([0, 3, 1, 2])
    dep = batch['depth'].cpu().detach().numpy()
    inp = batch['input'].cpu().detach().numpy()
    gt = batch['gt'].cpu().detach().numpy()
    save_image(ref, os.path.join(path, 'Ref.jpeg'))
    save_image(src, os.path.join(path, 'Src.jpeg'))
    save_image(tgt, os.path.join(path, 'Tgt.jpeg'))
    save_image(gt, os.path.join(path, 'GT.jpeg'))
    b, _, h, w = inp.shape
    d = dep.shape[0]
    inp = inp.reshape([b, d*2, 3, h, w])
    for i in range(d * 2):
        save_image(inp[0, i], os.path.join(path, 'Sphere_%d.jpeg' % (i + 1)))


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


def calc_ssim(im: np.ndarray, gt: np.ndarray):
    """Output SSIM"""
    pass


def calc_feature(data: list):
    """Calc Mean & Var"""
    data = np.array(data)
    size = data.shape[0]
    mean = np.mean(data)
    vari = 1.0 / size * np.power(data - mean, 2.0).sum()
    return mean, vari

