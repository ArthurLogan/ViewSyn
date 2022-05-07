import numpy as np
import torch
from Utils import bilinear_wrapper


def apply_rotation(rot: torch.Tensor, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor):
    """Apply rotation to coordinate"""
    shape = x.shape
    x = x.view([1, -1])
    y = y.view([1, -1])
    z = z.view([1, -1])

    dim = rot.shape[1]
    if dim == 3:
        pos = torch.cat([x, y, z])
        pos = torch.matmul(rot, pos)
    else:
        pos = torch.cat([x, y, z, torch.ones_like(x)])
        pos = torch.matmul(rot, pos)
    x = pos[0].view(shape)
    y = pos[1].view(shape)
    z = pos[2].view(shape)
    return x, y, z


def lat_lon_grid(h: int, w: int):
    """Generate polar coordinates"""
    lat = torch.linspace(-np.pi + np.pi / w, np.pi - np.pi / w, w).cuda()
    lon = torch.linspace(-np.pi / 2.0 + np.pi / (2.0 * h), np.pi / 2.0 - np.pi / (2.0 * h), h).cuda()
    return torch.meshgrid(lon, lat)


def backproject_spherical(lat: torch.Tensor, lon: torch.Tensor, depth: torch.Tensor):
    """Equirectangular projection cartesian coordinates"""
    h, w = lat.shape
    num_planes = depth.shape[0]
    rx = lat.view([1, h, w])
    ry = lon.view([1, h, w])
    depth = depth.view([num_planes, 1, 1])

    cosT = torch.cos(ry)
    x = depth * (torch.cos(rx) * cosT)
    y = depth * torch.sin(ry)
    z = -depth * (torch.sin(rx) * cosT)
    return x, y, z


def project_ods(points: torch.Tensor, order: int, intrinsic: torch.Tensor, h: int, w: int):
    """Project points to ods with camera position circular(Matry)"""
    x = points[0]
    y = points[1]
    z = points[2]
    r = intrinsic[0][0]

    f = r * r - x * x - z * z
    comp = z * z >= x * x
    domp = ~comp

    px = comp * x + domp * z
    pz = comp * z + domp * x

    pz2 = pz * pz
    a = 1 + px * px / pz2
    b = -2 * f * px / pz2
    c = f + f * f / pz2
    delt = b * b - 4 * a * c

    s = -order * torch.sign(pz) * torch.sqrt(delt)
    s = comp * s + domp * (-s)

    dx = (-b + s) / (2 * a)
    dz = (f - px * dx) / pz

    dx_final = comp * (-dx) + domp * (-dz)
    dz_final = comp * (-dz) + domp * (-dx)
    dx = dx_final
    dy = y
    dz = dz_final

    theta = -torch.atan2(dz, dx)
    phi = torch.atan2(dy, torch.sqrt(dx * dx + dz * dz))
    msk = torch.isnan(phi)
    phi[msk] = 1.0
    phi = torch.clip(phi, -np.pi/2.0, np.pi/2.0)

    u = ((theta + np.pi - np.pi / w) / (2 * np.pi - 2 * np.pi / w)) * (w - 1)
    v = ((phi + 0.5 * np.pi - 0.5 * np.pi / h) / (np.pi - np.pi / h)) * (h - 1)
    msk = delt < 0.
    u[msk] = 1.0
    v[msk] = 1.0
    uv = torch.stack([u, v], 0)
    return uv


def sphere_sweep(ref: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor,
                 rot: torch.Tensor, dep: torch.Tensor, intrinsic: torch.Tensor):
    """
    Sphere sweep to generate training planes

    Args:
        ref: Reference images [b, h, w, c]
        src: Source images [b, h, w, c]
        tgt: Target images [b, h, w, c]
        rot: Target pose [4, 4]
        dep: Depth of msi
        intrinsic: PD/2, pupillary distance
    """
    b, h, w, c = ref.shape
    d = dep.shape[0]
    lon, lat = lat_lon_grid(h, w)
    x, y, z = backproject_spherical(lat, lon, dep)
    x, y, z = apply_rotation(rot, x, y, z)
    pnt = torch.stack([x, y, z])

    inp = []
    for i in range(b):
        pix = project_ods(pnt, -1, intrinsic[i], h, w)
        im1 = bilinear_wrapper(ref[i:(i+1)], pix)
        im1 = torch.permute(im1, [0, 3, 1, 2])
        im1 = im1.contiguous().view([1, d * c, h, w])

        pix = project_ods(pnt, 1, intrinsic[i], h, w)
        im2 = bilinear_wrapper(src[i:(i+1)], pix)
        im2 = torch.permute(im2, [0, 3, 1, 2])
        im2 = im2.contiguous().view([1, d * c, h, w])

        inp.append(torch.cat([im1, im2], 1))

    inp = torch.cat(inp, 0)
    out = torch.permute(tgt, [0, 3, 1, 2])
    return inp, out


def project_spherical(rot: torch.Tensor, pos: torch.Tensor, dep: torch.Tensor, h: int, w: int):
    """Project target image"""
    d = dep.shape[0]

    cx, cy, cz = apply_rotation(rot, pos[:, 0], pos[:, 1], -pos[:, 2])
    cx = cx.view([-1, 1, 1])
    cy = cy.view([-1, 1, 1])
    cz = cz.view([-1, 1, 1])

    lon, lat = lat_lon_grid(h, w)
    rx, ry, rz = backproject_spherical(lat, lon, dep)
    rx, ry, rz = apply_rotation(rot[:3, :3], rx, ry, rz)
    rx = rx.view([1, d, -1])
    ry = ry.view([1, d, -1])
    rz = rz.view([1, d, -1])
    dep = dep.view([1, d, 1])

    a = rx * rx + ry * ry + rz * rz
    b = 2 * (cx * rx + cy * ry + cz * rz)
    c = cx * cx + cy * cy + cz * cz - dep * dep
    delt = b * b - 4 * a * c

    t = (-b + torch.sqrt(delt)) / (2 * a)
    sx = cx + t * rx
    sy = cy + t * ry
    sz = cz + t * rz
    theta = -torch.atan2(sz, sx)
    phi = torch.atan2(sy, torch.sqrt(sx * sx + sz * sz))

    u = ((theta + np.pi - np.pi / w) / (2 * np.pi - 2 * np.pi / w)) * (w - 1)
    v = ((phi + 0.5 * np.pi - 0.5 * np.pi / h) / (np.pi - np.pi / h)) * (h - 1)
    pix = torch.stack([u, v], 1).view([-1, 2, d, h, w])
    return pix


def random_rotate_matrix(rf: float, tf: float):
    angle_range = [-0.03 * rf, 0.03 * rf]
    offst_range = [-0.01 * tf, 0.01 * tf]

    angles = torch.rand([3]) * (angle_range[1] - angle_range[0]) + angle_range[0]
    cos_a, sin_a = torch.cos(angles[0]), torch.sin(angles[0])
    cos_b, sin_b = torch.cos(angles[1]), torch.sin(angles[1])
    cos_y, sin_y = torch.cos(angles[2]), torch.sin(angles[2])
    rot = torch.tensor([
        [ cos_a * cos_y - cos_b * sin_a * sin_y,  sin_a * cos_y + cos_b * cos_a * sin_y, sin_b * sin_y],
        [-cos_a * sin_y - cos_b * sin_a * cos_y, -sin_a * sin_y + cos_b * cos_a * cos_y, sin_b * cos_y],
        [ sin_b * sin_a,                         -sin_b * cos_a,                         cos_b        ]
    ])
    off = torch.rand([3, 1]) * (offst_range[1] - offst_range[0]) + offst_range[0]
    mat = torch.cat([rot, off], dim=1)
    mat = torch.cat([mat, torch.tensor([0.0, 0.0, 0.0, 1.0]).view([1, 4])], dim=0).cuda()
    return mat, torch.inverse(mat)
