import torch
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
import argparse
import natsort
import sys

# from generate_high_res_map import compute_scale_and_shift, best_fit_transform, align_x, align_y, align_normal_x, align_normal_y


def create_patches(rgbs, out_path_tmp, configs):
    for index, img_file in enumerate(rgbs):
        print("Processing image %d/%d" % (index, len(rgbs)))
        img = cv2.imread(img_file)
        H, W = img.shape[:2]

        assert H == 968, "Height of image should be 968"
        assert W == 1296, "Width of image should be 1296"

        x = W // 128
        y = H // 128

        # crop images
        for j in range(y - 1):
            for i in range(x - 1):
                if j == y - 2:
                    j_start = H - 384
                else:
                    j_start = j * 128
                if i == x - 2:
                    i_start = W - 384
                else:
                    i_start = i * 128
                img_cur = img[j_start:j_start + 384, i_start:i_start + 384, :]
                target_file = os.path.join(out_path_tmp, "%06d_%02d_%02d.jpg" % (index, j, i))
                cv2.imwrite(target_file, img_cur)

        # save middle file for alignments
        img_cur = img[H // 2 - 192:H // 2 + 192, W // 2 - 192:W // 2 + 192]
        target_file = os.path.join(out_path_tmp, "%06d_mid.jpg" % index)
        cv2.imwrite(target_file, img_cur)


def merge_patches(out_path_tmp, out_path, configs):
    H, W = 968, 1296
    x = W // 128
    y = H // 128

    raise NotImplementedError


def main(configs):
    assert configs.mode in ["create_patches", "merge_patches"]
    out_path_tmp = os.path.join(configs.data_root, configs.scene, 'high_res_tmp')
    out_path = os.path.join(configs.data_root, configs.scene, 'monosdf_depth')

    os.makedirs(out_path_tmp, exist_ok=True)
    os.makedirs(out_path, exist_ok=True)

    images_dir = os.path.join(configs.data_root, configs.scene, 'color')

    if configs.debug:
        rgbs = [os.path.join(images_dir, '0000.jpg')]
    else:
        rgbs = natsort.natsorted(glob.glob(os.path.join(images_dir, '*.jpg')))

    if configs.mode == "create_patches":
        print("Creating patches...")
        create_patches(rgbs, out_path_tmp, configs)
    elif configs.mode == "merge_patches":
        print("Merging patches...")
        merge_patches(out_path_tmp, out_path, configs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ScanNet image size depth/normal estimation')
    parser.add_argument('--mode', required=True, help="choose from creating patches or merge pathces")
    parser.add_argument('--data_root', default="/home/ccw/Downloads/scannet_filtered")
    parser.add_argument('--scene', default="scene0017_00")
    parser.add_argument('--debug', action='store_true')
    configs = parser.parse_args()
    print(configs)
    main(configs)
