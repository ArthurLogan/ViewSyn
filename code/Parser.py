import argparse
import os
from Dataset import OdsSequence


def parse_args():
    """Parse argument"""
    parser = argparse.ArgumentParser('ViewSynthesis')

    # i/o
    parser.add_argument('-glob_dir', default='./glob/train/ods', help='Path to camera files')
    parser.add_argument('-image_dir', default='./data', help='Path to images directories')
    parser.add_argument('-checkpoint_dir', default='./checkpoint', help='Path to load/save the models')
    parser.add_argument('-result_dir', default='./result', help='Path to save result')
    parser.add_argument('-experiment_name', default='coord', help='Name for experiment to run')
    parser.add_argument('-infer_dir', default='infer/', help='Path to save inference images')
    parser.add_argument('-sphre_dir', default='sphre/', help='Path to save sphere images')
    parser.add_argument('-demon_dir', default='demon/', help='Path to save demonstrate images')
    parser.add_argument('-curve_dir', default='curve/', help='Path to save curves')

    # for eval
    parser.add_argument('-test_image_dir', default='./test_data', help='Path to test images directories')
    parser.add_argument('-test_demon_dir', default='./test', help='Path to save test images')

    # training hyper-parameters
    parser.add_argument('-lr', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('-betas', default=(0.9, 0.999), type=tuple, help='Beta hyper-parameter for Adam optimizer')
    parser.add_argument('-random_seed', default=8964, type=int, help='Random seed')
    parser.add_argument('-max_step', default=1000, type=int, help='Maximum number of training steps')
    parser.add_argument('-batch_size', default=1, type=int, help='Batch size, fixed to 1')

    # model-related
    parser.add_argument('-coord_net', action='store_true', help='Whether to append CoordNet')
    parser.add_argument('-transform_inv', action='store_true', help='Whether to train with transform inverse')
    parser.add_argument('-min_depth', default=1, type=int, help='Minimum scene depth')
    parser.add_argument('-max_depth', default=100, type=int, help='Maxmimum scene depth')
    parser.add_argument('-num_plane', default=32, type=int, help='Number of msi planes to predict')
    parser.add_argument('-nf_scratch', action='store_true', help='Train from scratch')

    # experiment-related
    parser.add_argument('-rot_factor', default=1.0, type=float, help='Rotation factor for transform_inverse_reg')
    parser.add_argument('-trs_factor', default=1.0, type=float, help='Transformation factor for transform_inverse_reg')
    parser.add_argument('-rot_per_iter', default=100, type=int, help='Update rotation matrix per iteration')

    # loss-related
    parser.add_argument('-which_loss', default='cube', choices=['cube', 'sphere'], help='Which loss to use')

    # demo-related
    parser.add_argument('-valid_per_iter', default=10, type=int, help='Test output per iteration')
    parser.add_argument('-infer_per_iter', default=50, type=int, help='Image output per iteration')
    parser.add_argument('-model_per_iter', default=500, type=int, help='Model output per iteration')

    args = parser.parse_args()
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.experiment_name)
    args.result_dir = os.path.join(args.result_dir, args.experiment_name)
    args.infer_dir = os.path.join(args.result_dir, args.infer_dir)
    args.sphre_dir = os.path.join(args.result_dir, args.sphre_dir)
    args.demon_dir = os.path.join(args.result_dir, args.demon_dir)
    args.curve_dir = os.path.join(args.result_dir, args.curve_dir)
    args.test_demon_dir = os.path.join(args.test_demon_dir, args.experiment_name)
    return args


def parse_glob(path: str):
    """Parse camera glob, return list of globs"""
    if not (os.path.isfile(path) and path[-4:] == '.txt'):
        return []
    sequence = []
    fr = open(path)
    lines = fr.readlines()
    for line in lines:
        line = line.strip().split(' ')
        if len(line) == 8:
            scene_id = line[0]
            image_id = line[1:4]
            baseline = float(line[4])
            tgt_pose = [float(x) for x in line[5:]]
            sequence.append(OdsSequence(scene_id, image_id, baseline, tgt_pose))
    fr.close()
    return sequence
