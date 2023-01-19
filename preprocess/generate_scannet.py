import torch
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import argparse

from preprocess.generate_high_res_map import compute_scale_and_shift, best_fit_transform, align_x, align_y, align_normal_x, align_normal_y

def main(args):
    assert args.mode in ["create_patches", "merge_patches"]
    out_path_prefix = './highres_tmp'
    out_path_for_training = os.path.join(args.data_root, args.scene, 'monosdf_depth')

    

    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ScanNet image size depth/normal estimation')
    parser.add_argument('--mode', required=True, help="choose from creating patches or merge pathces")
    parser.add_argument('--data_root', default="/home/ccw/Downloads/scannet_filtered")
    parser.add_argument('--scene', required=True, default="scene0017_00")
    args = parser.parse_args()
    main(args)
