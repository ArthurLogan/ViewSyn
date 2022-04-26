import os
import torch
import numpy as np
from time import time

from Parser import parse_args
from Utils import save_image, save_msi, save_loss, save_input, save_origin, calc_psnr, remove_output
from Loader import Loader
from Model import MatryNet
from Projector import random_rotate_matrix


def train(args):
    loader = Loader(args.glob_dir, args.train_dir, args.min_depth, args.max_depth, args.num_plane, args.batch_size)
    matrynet = MatryNet(args.num_plane, loader.image_h, loader.image_w,
                        args.coord_net, args.transform_inv,
                        args.learning_rate, args.betas
                        ).cuda()
    rot_mat, rot_inv = random_rotate_matrix(args.rot_factor, args.trs_factor)
    loader.load_rotate_matrix(rot_inv)
    matrynet.load_rotate_matrix(rot_mat)

    remove_output(args.result_dir)
    batch = loader.load_valid_batch()
    save_image(batch['gt'].cpu().detach().numpy(), os.path.join(args.demo_dir, 'Target.jpeg'))
    save_origin(batch['ref'], batch['src'], os.path.join(args.demo_dir, 'Origin.jpeg'))
    save_input(batch['input'], batch['depth'], os.path.join(args.demo_dir, 'Sphere.jpeg'))

    train_loss, valid_loss = [], []
    avg_loss = 0
    step = 0
    stime = time()
    while step < args.max_step:
        batch = loader.load_train_batch()
        loss = matrynet.learn(batch)
        train_loss.append(loss)
        avg_loss += loss

        step += 1
        if step % args.rot_per_iter == 0:
            rot_mat, rot_inv = random_rotate_matrix(args.rot_factor, args.trs_factor)
            loader.load_rotate_matrix(rot_inv)
            matrynet.load_rotate_matrix(rot_mat)

        if step % args.valid_per_iter == 0:
            batch = loader.load_valid_batch()
            im, gt, rgb, wgt, alp, loss = matrynet.test(batch)
            valid_loss.append(loss)
            psnr = calc_psnr(im, gt)

            if step % args.infer_per_iter == 0:
                etime = time()
                print('[%d/%d] Average Loss %6.4lf' % (step, args.max_step, avg_loss / args.infer_per_iter))
                print('[%d/%d] PSNR %6.4lf' % (step, args.max_step, psnr))
                print('[%d/%d] Using %4.2lf seconds' % (step, args.max_step, etime - stime))
                avg_loss = 0

                save_image(im, os.path.join(args.infer_dir, 'Infer_%d.jpeg' % step))
                save_msi(rgb, wgt, alp, os.path.join(args.sphre_dir, 'MSI_%d.jpeg' % step))
                save_loss(train_loss, os.path.join(args.loss_dir, 'Train.jpeg'))
                save_loss(valid_loss, os.path.join(args.loss_dir, 'Valid.jpeg'))
                stime = time()

            if step % args.model_per_iter == 0:
                save_path = os.path.join(args.checkpoint_dir, 'checkpoint_%d.pkl' % step)
                torch.save(matrynet.state_dict(), save_path)


def demon(args):
    loader = Loader(args.glob_dir, args.test_dir, args.min_depth, args.max_depth, args.num_plane, args.batch_size)
    matrynet = MatryNet(args.num_plane, loader.image_h, loader.image_w,
                        args.coord_net, args.transform_inv,
                        args.learning_rate, args.betas
                        ).cuda()

    matrynet.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, args.experiment_name)))
    batch = loader.load_valid_batch()
    im, gt, rgb, wgt, alp, loss = matrynet.test(batch)

    psnr = calc_psnr(im, gt)
    print('[OUTPUT] PSNR', psnr)

    save_image(im, os.path.join(args.demo_dir, 'Target.jpeg'))
    save_image(gt, os.path.join(args.demo_dir, 'GT.jpeg'))

    save_image(rgb[0], os.path.join(args.demo_dir, 'Color_f.jpeg'))
    save_image(rgb[-1], os.path.join(args.demo_dir, 'Color_n.jpeg'))

    save_image(wgt[0], os.path.join(args.demo_dir, 'Weight_f.jpeg'))
    save_image(wgt[-1], os.path.join(args.demo_dir, 'Weight_n.jpeg'))

    save_image(alp[0], os.path.join(args.demo_dir, 'Alpha_f.jpeg'))
    save_image(alp[-1], os.path.join(args.demo_dir, 'Alpha_n.jpeg'))

    save_input(batch['input'], batch['depth'], os.path.join(args.demo_dir, 'Sphere.jpeg'))


def main():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    if args.op == 'train':
        train(args)
    else:
        demon(args)


if __name__ == '__main__':
    main()
