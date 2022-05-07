import os
import shutil

import torch
from tqdm import tqdm
import json
from Parser import parse_args
from Utils import save_image, save_output, calc_psnr, calc_feature
from Loader import Loader
from Model import MatryNet


def eval():
    args = parse_args()
    loader = Loader(args.glob_dir, args.test_image_dir, args.min_depth, args.max_depth, args.num_plane, args.batch_size)
    matrynet = MatryNet(args.num_plane, loader.image_h, loader.image_w, args.coord_net, args.transform_inv,
                        args.lr, args.betas, args.which_loss).cuda()

    if os.path.exists(args.test_demon_dir):
        shutil.rmtree(args.test_demon_dir)
    fname = os.listdir(args.checkpoint_dir)[0]
    matrynet.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, fname)))

    num_batch = loader.num_batch
    loss, psnr = [], []
    for _ in tqdm(range(num_batch)):
        batch = loader.load_train_batch()
        im, gt, rgb, wgt, alp, los = matrynet.test(batch)
        path = os.path.join(args.test_demon_dir, batch['name'][0])
        if not os.path.exists(path):
            os.makedirs(path)
        save_image(im, os.path.join(path, 'Out.jpeg'))
        save_image(gt, os.path.join(path, 'GT.jpeg'))
        save_output((rgb, wgt, alp), os.path.join(path, 'MSI.jpeg'))

        loss.append(los)
        psnr.append(calc_psnr(im, gt))

    avg_loss, var_loss = calc_feature(loss)
    avg_psnr, var_psnr = calc_feature(psnr)
    res = dict()
    res['model'] = args.experiment_name
    res['avg-loss'] = avg_loss
    res['var-loss'] = var_loss
    res['avg-psnr'] = avg_psnr
    res['var-psnr'] = var_psnr
    json.dump(res, open(os.path.join(args.test_demon_dir, 'feature.json'), 'w'))


if __name__ == '__main__':
    eval()
