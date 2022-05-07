import os
import torch
from time import time
from Parser import parse_args
from Utils import prepare_for_train, save_batch, save_curve, save_image, save_output, calc_psnr
from Loader import Loader
from Model import MatryNet
from Projector import random_rotate_matrix


def update_rotation(loader: Loader, net: MatryNet, rf: float, tf: float):
    """Update Transform Rotate Matrix"""
    mat, inv = random_rotate_matrix(rf, tf)
    loader.load_rotate_matrix(inv)
    net.load_rotate_matrix(mat)


def train():
    args = parse_args()
    torch.manual_seed(args.random_seed)
    loader = Loader(args.glob_dir, args.image_dir, args.min_depth, args.max_depth, args.num_plane, args.batch_size)
    matrynet = MatryNet(args.num_plane, loader.image_h, loader.image_w, args.coord_net, args.transform_inv,
                        args.lr, args.betas, args.which_loss).cuda()

    prepare_for_train(args)
    save_batch(loader.load_valid_batch(), args.demon_dir)

    pre_train_step = 0
    if args.nf_scratch:
        fname = os.listdir(args.checkpoint_dir)[0]
        matrynet.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, fname)))
        pre_train_step = int(fname[11:-4])
    update_rotation(loader, matrynet, args.rot_factor, args.trs_factor)

    step = pre_train_step
    train_loss, valid_loss, valid_psnr = [], [], []
    avg_loss = 0
    stime = time()
    while step < args.max_step:
        batch = loader.load_train_batch()
        loss = matrynet.learn(batch)
        train_loss.append(loss)
        avg_loss += loss

        step += 1
        if step % args.rot_per_iter == 0:
            update_rotation(loader, matrynet, args.rot_factor, args.trs_factor)

        if step % args.valid_per_iter == 0:
            batch = loader.load_valid_batch()
            im, gt, rgb, wgt, alp, loss = matrynet.test(batch)
            psnr = calc_psnr(im, gt)
            valid_loss.append(loss)
            valid_psnr.append(psnr)

            if step % args.infer_per_iter == 0:
                etime = time()
                print('[%d/%d] Average Loss %6.4lf' % (step, args.max_step, avg_loss / args.infer_per_iter))
                print('[%d/%d] PSNR %6.4lf' % (step, args.max_step, psnr))
                print('[%d/%d] Using %4.2lf seconds' % (step, args.max_step, etime - stime))
                avg_loss = 0
                save_image(im, os.path.join(args.infer_dir, 'Infer_%d.jpeg' % step))
                save_output((rgb, wgt, alp), os.path.join(args.sphre_dir, 'MSI_%d.jpeg' % step))
                save_curve([train_loss, valid_loss, valid_psnr], pre_train_step, args.curve_dir)
                stime = time()

            if step % args.model_per_iter == 0:
                for fname in os.listdir(args.checkpoint_dir):
                    os.remove(os.path.join(args.checkpoint_dir, fname))
                save_path = os.path.join(args.checkpoint_dir, 'checkpoint_%d.pkl' % step)
                torch.save(matrynet.state_dict(), save_path)


if __name__ == '__main__':
    train()
