import argparse
import os
from Dataset import OdsSequence


def parse_args():
    """Parse argument"""
    parser = argparse.ArgumentParser('ViewSynthesis')

    # i/o
    parser.add_argument('-glob_dir', default='../glob/train/ods', help='Path to camera files')
    parser.add_argument('-train_dir', default='../data', help='Path to training images directories')
    parser.add_argument('-test_dir', default='../data', help='Path to testing images directories')
    parser.add_argument('-checkpoint_dir', default='../model', help='Path to load/save the models')
    parser.add_argument('-experiment_name', default='checkpoint_20000.pkl', help='Name for experiment to run')
    parser.add_argument('-result_dir', default='../result', help='Path to save result')
    parser.add_argument('-infer_dir', default='../result/img', help='Path to save inference images')
    parser.add_argument('-sphre_dir', default='../result/sphre', help='Path to save sphere images')
    parser.add_argument('-loss_dir', default='../result/loss', help='Path to save loss curves')
    parser.add_argument('-demo_dir', default='../result/demo', help='Path to save demo images')

    # training hyper-parameters
    parser.add_argument('-learning_rate', default=0.0002, type=float, help='Learning rate')
    parser.add_argument('-betas', default=(0.9, 0.999), type=tuple, help='Beta hyper-parameter for Adam optimizer')
    parser.add_argument('-random_seed', default=8964, type=int, help='Random seed')
    parser.add_argument('-max_step', default=20000, type=int, help='Maximum number of training steps')
    parser.add_argument('-batch_size', default=1, type=int, help='Batch size, fixed to 1')

    # model-related
    parser.add_argument('-op', default='train', choices=['train', 'demo'], help='Which operation to perform')
    parser.add_argument('-input_type', default='ODS', choices=['ODS', 'PP'], help='Input image type')
    parser.add_argument('-coord_net', action='store_true', help='Whether to append CoordNet')
    parser.add_argument('-transform_inv', action='store_true', help='Whether to train with transform inverse')
    parser.add_argument('-min_depth', default=1, type=int, help='Minimum scene depth')
    parser.add_argument('-max_depth', default=100, type=int, help='Maxmimum scene depth')
    parser.add_argument('-num_plane', default=32, type=int, help='Number of msi planes to predict')

    # experiment-related
    parser.add_argument('-rot_factor', default=1.0, type=float, help='Rotation factor for transform_inverse_reg')
    parser.add_argument('-trs_factor', default=1.0, type=float, help='Transformation factor for transform_inverse_reg')
    parser.add_argument('-rot_per_iter', default=100, type=int, help='Update rotation matrix per iteration')

    # loss-related
    parser.add_argument('-which_loss', default='pixel', choices=['pixel', 'elpips'], help='Which loss to use for training')
    parser.add_argument('-spherical_attention', action='store_true', help='Calculate loss with spherically-aware map')

    # demo-related
    parser.add_argument('-valid_per_iter', default=10, type=int, help='Test output per iteration')
    parser.add_argument('-infer_per_iter', default=50, type=int, help='Image output per iteration')
    parser.add_argument('-model_per_iter', default=10000, type=int, help='Model output per iteration')

    args = parser.parse_args()
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

