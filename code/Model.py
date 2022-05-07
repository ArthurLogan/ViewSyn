import torch
from torch import nn
import numpy as np
import lpips
from Projector import project_spherical, lat_lon_grid
from Utils import bilinear_wrapper, composite, save_tensor


class CoordLayer(nn.Module):
    """Coord Vec to convolutional network"""
    def __init__(self, inpc: int, outc: int, h: int, w: int,
                 kernel: tuple = (3, 3), stride: tuple = (1, 1), padding: tuple = (1, 1), dilation: tuple = (1, 1),
                 coord: bool = True, output_layer: bool = False):
        """Init Layer"""
        super().__init__()
        self.coord = coord
        self.conv = nn.Conv2d(inpc + coord, outc, kernel, stride, padding, dilation, padding_mode='zeros')
        if output_layer:
            self.activate = nn.Tanh()
        else:
            self.activate = nn.Sequential(nn.LayerNorm([outc, h, w]), nn.ReLU())

    def forward(self, input: torch.Tensor):
        """Forward"""
        if self.coord:
            input = self.coordnet(input)
        out = self.conv(input)
        out = self.activate(out)
        return out

    def coordnet(self, input: torch.Tensor):
        """Add coord channel"""
        b, c, h, w = input.shape
        coord = torch.linspace(-np.pi / 2.0, np.pi / 2.0, h).view([h, 1]).repeat((1, w)).cuda()
        coord = torch.abs(torch.sin(coord)).view([1, 1, h, w]).repeat((b, 1, 1, 1))
        coord = coord - input[:, :1, :, :]
        return torch.cat([input, coord], dim=1)


class CoordTransposeLayer(nn.Module):
    """Coord Vec to convolutional transpose network"""
    def __init__(self, inpc: int, outc: int, h: int, w: int,
                 kernel: tuple = (4, 4), stride: tuple = (2, 2), padding: tuple = (1, 1), coord: bool = True):
        """Init Layer"""
        super().__init__()
        self.coord = coord
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(inpc + coord, outc, kernel, stride, padding, padding_mode='zeros'),
            nn.LayerNorm([outc, h, w]),
            nn.ReLU()
        )

    def forward(self, input: torch.Tensor):
        """Forward"""
        if self.coord:
            input = self.coordnet(input)
        out = self.conv(input)
        return out

    def coordnet(self, input: torch.Tensor):
        """Add coord channel"""
        b, c, h, w = input.shape
        coord = torch.linspace(-np.pi / 2.0, np.pi / 2.0, h).view([h, 1]).repeat((1, w)).cuda()
        coord = torch.abs(torch.sin(coord)).view([1, 1, h, w]).repeat((b, 1, 1, 1))
        coord = coord - input[:, :1, :, :]
        return torch.cat([input, coord], dim=1)


class CubeLoss(nn.Module):
    """Project to cube and calc loss"""
    def __init__(self, h: int, w: int):
        super().__init__()
        self.pix = []
        r = w // 4
        dif = torch.linspace(-1.0, 1.0, r + 1)[:-1] + 1.0 / r
        dh, dw = torch.meshgrid(dif, dif)
        one = torch.ones_like(dh)
        for i in range(3):
            sign = [1, -1]
            for sig in sign:
                coord = [dh, dh, dh]
                coord[i] = sig * one
                coord[(i+1)%3] = dw
                x, y, z = coord
                t = torch.sqrt(x * x + y * y + z * z)
                x = x / t
                y = y / t
                z = z / t
                u = -torch.atan2(z, x)
                v = torch.atan2(y, torch.sqrt(x * x + z * z))
                u = ((u + np.pi - np.pi / w) / (2 * np.pi - 2 * np.pi / w)) * (w - 1)
                v = ((v + 0.5 * np.pi - 0.5 * np.pi / h) / (np.pi - np.pi / h)) * (h - 1)
                uv = torch.stack([u, v])
                self.pix.append(uv)
        self.pix = torch.stack(self.pix, dim=1).cuda()
        self.loss_fn = nn.ModuleList([lpips.LPIPS(net='vgg'), nn.L1Loss()])
        self.loss_wt = [1, 5]

    def get(self, im: torch.Tensor, gt: torch.Tensor):
        """Loss"""
        los = 0
        for wt, fn in zip(self.loss_wt, self.loss_fn):
            los += wt * fn(im, gt)
        los = torch.mean(los)
        return los

    def forward(self, im: torch.Tensor, gt: torch.Tensor):
        """Forward"""
        im = bilinear_wrapper(torch.permute(im, [0, 2, 3, 1]), self.pix)
        gt = bilinear_wrapper(torch.permute(gt, [0, 2, 3, 1]), self.pix)
        im = torch.permute(im, [0, 3, 1, 2])
        gt = torch.permute(gt, [0, 3, 1, 2])
        los = self.get(im, gt)
        return los


class SphereLoss(nn.Module):
    """Use sphere attention to calc loss"""
    def __init__(self, h: int, w: int):
        super().__init__()
        self.h = h
        self.w = w
        self.att = self.get_att_matry()
        self.loss_fn = nn.ModuleList([lpips.LPIPS(net='vgg')])
        self.loss_wt = [1]

    def get_att_matry(self, eps: float = 1e-12):
        """Attention matryodshka"""
        h = self.h
        w = self.w
        d = np.pi / h

        lat = torch.linspace(-np.pi + eps, np.pi + eps, w).cuda()
        lon = torch.linspace(-np.pi / 2.0 + eps, np.pi / 2.0 + eps, h).cuda()
        oy, ox = torch.meshgrid(lon, lat)

        lat = torch.linspace(-np.pi + d, np.pi + d, w).cuda()
        lon = torch.linspace(-np.pi / 2.0 + d / 2.0, np.pi / 2.0 + d / 2.0, h).cuda()
        sy, sx = torch.meshgrid(lon, lat)
        att = abs(sx - ox) * abs(torch.cos(sy) - torch.cos(oy))
        return att.view([1, 1, h, w])

    def get_att(self, eps: float = 1e-12):
        """Attention"""
        h = self.h
        w = self.w
        cy, _ = lat_lon_grid(h, w)
        att = 2 * np.pi * np.pi / torch.cos(cy + eps)
        return att.view([1, 1, h, w])

    def forward(self, im: torch.Tensor, gt: torch.Tensor):
        """Forward"""
        los = 0
        for wt, fn in zip(self.loss_wt, self.loss_fn):
            los += wt * fn(im * self.att, gt * self.att)
        los = torch.mean(los)
        return los


class MatryNet(nn.Module):
    """MatryODShka for view synthesis for ods"""
    def __init__(self, num_depth: int, h: int, w: int, coord: bool, transform: bool, lr: float, beta: tuple, loss_type: str):
        """Init Network"""
        super().__init__()

        self.num_plane = num_depth
        self.height = h
        self.width = w
        self.transform = transform
        self.rot = torch.tensor(np.identity(4), dtype=torch.float).cuda()
        self.one = torch.tensor(np.identity(4), dtype=torch.float).cuda()

        channel = 2 * num_depth
        self.conv1 = nn.Sequential(
            CoordLayer(3 * channel, channel, h, w, coord=coord),
            CoordLayer(channel, 2 * channel, h // 2, w // 2, stride=(2, 2), coord=coord)
        )
        self.conv2 = nn.Sequential(
            CoordLayer(2 * channel, 2 * channel, h // 2, w // 2, coord=coord),
            CoordLayer(2 * channel, 4 * channel, h // 4, w // 4, stride=(2, 2), coord=coord)
        )
        self.conv3 = nn.Sequential(
            CoordLayer(4 * channel, 4 * channel, h // 4, w // 4, coord=coord),
            CoordLayer(4 * channel, 4 * channel, h // 4, w // 4, coord=coord),
            CoordLayer(4 * channel, 8 * channel, h // 8, w // 8, stride=(2, 2), coord=coord)
        )
        self.conv4 = nn.Sequential(
            CoordLayer(8 * channel, 8 * channel, h // 8, w // 8, padding=(2, 2), dilation=(2, 2), coord=coord),
            CoordLayer(8 * channel, 8 * channel, h // 8, w // 8, padding=(2, 2), dilation=(2, 2), coord=coord),
            CoordLayer(8 * channel, 8 * channel, h // 8, w // 8, padding=(2, 2), dilation=(2, 2), coord=coord)
        )

        self.conv5 = nn.Sequential(
            CoordTransposeLayer(16 * channel, 4 * channel, h // 4, w // 4, coord=coord),
            CoordLayer(4 * channel, 4 * channel, h // 4, w // 4, coord=coord),
            CoordLayer(4 * channel, 4 * channel, h // 4, w // 4, coord=coord)
        )
        self.conv6 = nn.Sequential(
            CoordTransposeLayer(8 * channel, 2 * channel, h // 2, w // 2, coord=coord),
            CoordLayer(2 * channel, 2 * channel, h // 2, w // 2, coord=coord)
        )
        self.conv7 = nn.Sequential(
            CoordTransposeLayer(4 * channel, channel, h, w, coord=coord),
            CoordLayer(channel, channel, h, w, coord=coord),
            CoordLayer(channel, channel, h, w, output_layer=True, coord=coord)
        )
        self.ind = 0
        self.optim = torch.optim.Adam(self.parameters(), lr=lr, betas=beta)
        self.loss_type = loss_type
        if loss_type == 'cube':
            self.loss_fn = CubeLoss(h, w)
        else:
            self.loss_fn = SphereLoss(h, w)

    def load_rotate_matrix(self, rot: torch.Tensor):
        """Load rotation matrix"""
        self.rot = rot

    def forward(self, input: torch.Tensor):
        """UNet Forward"""
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.conv5(torch.cat([out4, out3], dim=1))
        out6 = self.conv6(torch.cat([out5, out2], dim=1))
        out7 = self.conv7(torch.cat([out6, out1], dim=1))
        if not self.training:
            save_tensor([out1, out2, out3, out4, out5, out6, out7], './result/demo/Out.jpeg')
        return out7

    def render(self, rgb: torch.Tensor, alp: torch.Tensor, pose: torch.Tensor, dep: torch.Tensor, rot: torch.Tensor):
        """Render Target View"""
        d, b, _, h, w = rgb.shape
        pix = project_spherical(rot, pose, dep, h, w)
        rgb = rgb.permute([1, 0, 3, 4, 2])
        out = []
        for i in range(b):
            img = bilinear_wrapper(rgb[i], pix[i])
            out.append(img)
        out = torch.stack(out).permute([1, 0, 4, 2, 3])
        out = composite(out, alp)
        return out

    def step(self, inp: torch.Tensor, pose: torch.Tensor, depth: torch.Tensor, rot: torch.Tensor):
        """Network Forward"""
        b, _, h, w = inp.shape
        d = self.num_plane
        out = self.forward(inp)
        wgt = ((out[:, :d] + 1.) / 2.).view([b, d, 1, h, w])
        alp = ((out[:, d:] + 1.) / 2.).view([b, d, 1, h, w])
        ref = inp[:, :(d * 3)].view([b, d, 3, h, w])
        src = inp[:, (d * 3):].view([b, d, 3, h, w])
        rgb = wgt * ref + (1 - wgt) * src

        wgt = torch.permute(wgt, [1, 0, 2, 3, 4])
        alp = torch.permute(alp, [1, 0, 2, 3, 4])
        rgb = torch.permute(rgb, [1, 0, 2, 3, 4])
        out = self.render(rgb, alp, pose, depth, rot)
        return out, rgb, wgt, alp

    def learn(self, batch: dict):
        """Train"""
        self.train()
        inp = batch['input']
        tri = batch['trans']
        gt = batch['gt']
        pos = batch['pos']
        dep = batch['depth']
        out, _, wgt, alp = self.step(inp, pos, dep, self.one)
        los = self.loss_fn(out, gt)
        if self.transform:
            tro, _, _, _ = self.step(tri, pos, dep, self.rot)
            los += 10 * self.loss_fn(tro, out)
        los.backward()
        self.ind += 1
        if self.ind % 32 == 0:
            self.optim.step()
            self.optim.zero_grad()
        return los.item()

    def test(self, batch: dict):
        """Validate"""
        self.eval()
        inp = batch['input']
        gt = batch['gt']
        pos = batch['pos']
        dep = batch['depth']
        im, rgb, wgt, alp = self.step(inp, pos, dep, self.one)
        los = self.loss_fn(im, gt)

        im = im.cpu().detach().numpy()
        gt = gt.cpu().detach().numpy()
        d = dep.shape[0]
        ind = torch.tensor([0, d // 2, d - 1], dtype=torch.long).cuda()
        rgb = rgb[ind, 0]
        wgt = wgt[ind, 0]
        alp = alp[ind, 0]
        rgb = rgb.cpu().detach().numpy()
        wgt = wgt.cpu().detach().numpy()
        alp = alp.cpu().detach().numpy()
        return im, gt, rgb, wgt, alp, los.item()
