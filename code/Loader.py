import os
import numpy as np
import random
import cv2
import torch

from Dataset import OdsSequence
from Parser import parse_glob
from Projector import sphere_sweep


class Loader:
    """Load Camera glob and Images, Sample batch and return"""
    def __init__(self, glob_dir: str, image_dir: str, min_depth: int, max_depth: int, num_planes: int, batch_size: int):
        """
        Args:
            glob_dir: Camera setting directory
            image_dir: Training Images directory
            min_depth: Minimum depth of concentric sphere
            max_depth: Maximum depth of concentric sphere
            num_planes: Total number of concentric sphere
            batch_size: Fixed to 1
        """
        self.glob_dir = glob_dir
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.num_plane = num_planes
        self.depth = self.inv_depth(min_depth, max_depth, num_planes)
        self.rot = torch.tensor(np.identity(4), dtype=torch.float).cuda()
        self.one = torch.tensor(np.identity(4), dtype=torch.float).cuda()
        sequences = []
        for filename in os.listdir(glob_dir):
            sequences += parse_glob(os.path.join(glob_dir, filename))

        self.valid_data = sequences[:3]
        self.train_data = sequences[3:]
        random.shuffle(self.train_data)
        print('[Loader] Number of sequence: %d' % len(sequences))

        image = self.load_image_data(sequences[0])
        h, w, _ = image[0].shape
        self.image_h = h
        self.image_w = w
        print('[Loader] Image size (%d, %d)' % (h, w))

        # sample batch
        self.index = 0
        self.num_batch = len(self.train_data) // batch_size

    def __len__(self):
        """Return total number of training data"""
        return len(self.train_data)

    def load_rotate_matrix(self, rot: torch.Tensor):
        """Load rotation matrix"""
        self.rot = rot

    def load_train_batch(self):
        """Sample a batch, if run out of data, then shuffle and resample"""
        if self.index == self.num_batch:
            self.index = 0
            random.shuffle(self.train_data)

        name = []
        ref, src, tgt = [], [], []
        pos = []
        intrinsic = []
        for sequence in self.train_data[self.index*self.batch_size:(self.index+1)*self.batch_size]:
            ref_, src_, tgt_ = self.load_image_data(sequence)
            pos_ = torch.tensor(sequence.tgt_pose, dtype=torch.float).cuda()
            int_ = torch.tensor(sequence.intrinsic, dtype=torch.float).cuda()
            name.append(sequence.name())
            ref.append(ref_)
            src.append(src_)
            tgt.append(tgt_)
            pos.append(pos_)
            intrinsic.append(int_)

        batch = dict()
        batch['name'] = name
        batch['ref'] = torch.stack(ref)
        batch['src'] = torch.stack(src)
        batch['tgt'] = torch.stack(tgt)
        batch['pos'] = torch.stack(pos)
        batch['depth'] = torch.tensor(self.depth, dtype=torch.float).cuda()
        batch['intrinsic'] = torch.stack(intrinsic)
        inp, out = sphere_sweep(batch['ref'], batch['src'], batch['tgt'], self.one, batch['depth'], batch['intrinsic'])
        tri, ___ = sphere_sweep(batch['ref'], batch['src'], batch['tgt'], self.rot, batch['depth'], batch['intrinsic'])
        batch['input'] = inp
        batch['gt'] = out
        batch['trans'] = tri
        self.index += 1
        return batch

    def load_valid_batch(self):
        """Sample test data for visualization"""
        data = self.valid_data[0]
        ref, src, tgt = self.load_image_data(data)
        pos = torch.tensor(data.tgt_pose, dtype=torch.float).cuda()
        intri = torch.tensor(data.intrinsic, dtype=torch.float).cuda()
        h, w, c = ref.shape

        batch = dict()
        batch['name'] = [data.name()]
        batch['ref'] = ref.view([1, h, w, c])
        batch['src'] = src.view([1, h, w, c])
        batch['tgt'] = tgt.view([1, h, w, c])
        batch['pos'] = pos.view([1, 3])
        batch['depth'] = torch.tensor(self.depth, dtype=torch.float).cuda()
        batch['intrinsic'] = intri.view([1, 3, 3])
        inp, out = sphere_sweep(batch['ref'], batch['src'], batch['tgt'], self.one, batch['depth'], batch['intrinsic'])
        tri, ___ = sphere_sweep(batch['ref'], batch['src'], batch['tgt'], self.rot, batch['depth'], batch['intrinsic'])
        batch['input'] = inp
        batch['gt'] = out
        batch['trans'] = tri
        return batch

    def load_image_data(self, sequence: OdsSequence):
        """
        To prevent memory explosion, only load images when needed, seq: [src, ref, tgt]
        Args:
            sequence: Ods sequence(scene, image_id)
        """
        images = []
        for ind in sequence.image_id:
            image = cv2.imread(self.image_dir + '/' + sequence.scene_id + '_pos%s.jpeg' % ind)
            image = image[:, :, :3]
            image = torch.tensor(image, dtype=torch.float).cuda()
            image = 2. * (image / 255.) - 1.
            images.append(image)
        return images

    def inv_depth(self, min_depth: int, max_depth: int, num_planes: int):
        """Interpolate depth"""
        depth = []
        dep_s = 1. / min_depth
        dep_e = 1. / max_depth
        for i in range(num_planes):
            fract = float(i) / float(num_planes - 1)
            dep_i = dep_s + (dep_e - dep_s) * fract
            depth.append(1.0 / dep_i)
        depth = sorted(depth)
        return depth[::-1]

