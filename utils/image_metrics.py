BASELINE_PATH_PREFIX = "baseline_"
NRC_PATH_PREFIX      = "nrc_output_"
IMAGE_EXTENSION      = ".png"

import cv2
import numpy as np
import sys
import argparse
import os
from flip_loss import compute_ldrflip, color_space_transform
import torch

image_to_luminance = lambda image: np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr_value


def rmae(img1, img2):
    img1 = image_to_luminance(img1)
    img2 = image_to_luminance(img2)

    max_p = 255.0

    abs_diff = np.abs(img1 - img2) / max_p
    abs_ref = np.abs(img1) / max_p
    
    normalized_abs_diff = abs_diff / (abs_ref + 1e-10)
    rmae_value = np.sqrt(normalized_abs_diff)
    
    return np.mean(rmae_value)


def rmse(img1, img2):
    rmse_bands = []
    max_p = 255.0

    diff = img1 - img2
    mse_bands = np.mean(np.square(diff / max_p), axis=(0, 1))
    rmse_bands = np.sqrt(mse_bands)
    return np.mean(rmse_bands)

def flip(img1, img2):
    max_p = 255.0
    img1 = torch.from_numpy(img1 / max_p)
    img2 = torch.from_numpy(img2 / max_p)
    img1 = img1.unsqueeze(0).permute(0, 3, 1, 2)
    img2 = img2.unsqueeze(0).permute(0, 3, 1, 2)

    img1 = torch.clamp(img1, 0, 1)
    img2 = torch.clamp(img2, 0, 1)

    # Transform reference and test to opponent color space
    img1 = color_space_transform(img1, 'srgb2ycxcz')
    img2 = color_space_transform(img2, 'srgb2ycxcz')

    flip = compute_ldrflip(img1, img2, (0.7 * 1920 / 0.7) * np.pi / 100, 0.7, 0.5, 0.4, 0.95, 1e-15)
    flip = torch.squeeze(flip)
    return torch.mean(flip)

# metrics to use
metric_names = ["PSNR", "RMAE", "RMSE", "ꟻLIP"]
metrics = [psnr, rmae, rmse, flip]

# processing
def process(file_pairs):
    for bfile, nfile in file_pairs:
        print(f'[INFO] Processing files {bfile} <--> {nfile}.')
        base_img = cv2.imread(bfile)
        nrc_img = cv2.imread(nfile)

        if base_img is None:
            print(f'[WARN] Unable to read/open {bfile} properly. Skipping...')
            continue

        if nrc_img is None:
            print(f'[WARN] Unable to read/open {nfile} properly. Skipping...')
            continue

        if base_img.shape != nrc_img.shape:
            print(f'[WARN] Shape mismatch between images: {base_img.shape} <--> {nrc_img.shape}. Skipping...')
            continue

        base_img = base_img.astype(np.float32)
        nrc_img = nrc_img.astype(np.float32)

        for metric, name in zip(metrics, metric_names):
            data = metric(base_img, nrc_img)
            print(f'[DATA] {name}: {data}')


if __name__ == "__main__":
    # Define argparse parse
    parser = argparse.ArgumentParser(description='Get metrics from comparing two images.')

    # Define mutually exclusive groups
    group = parser.add_mutually_exclusive_group(required=True)

    # Add arguments
    group.add_argument('-filenames', metavar='filenames', type=str, nargs='*',
                        help='file(s) to process')
    group.add_argument('--autolook', metavar='autolook_directory', type=str, nargs='?',
                        help=f'perform auto lookup of files matching {BASELINE_PATH_PREFIX}*{IMAGE_EXTENSION}/{NRC_PATH_PREFIX}*{IMAGE_EXTENSION}')

    args = parser.parse_args()

    file_pairs = []

    if args.autolook:
        dirname = args.autolook

        files = os.listdir(dirname)
        filenames = [file for file in files if os.path.isfile(os.path.join(dirname, file))]

        baseline_files = [file for file in filenames if file.startswith(BASELINE_PATH_PREFIX) and file.endswith(IMAGE_EXTENSION)]
        nrc_files = [file for file in filenames if file.startswith(NRC_PATH_PREFIX) and file.endswith(IMAGE_EXTENSION)]

        baseline_files = set(baseline_files)
        nrc_files = set(nrc_files)

        for bfile in baseline_files:
            expected_nrc_filename = NRC_PATH_PREFIX + bfile[len(BASELINE_PATH_PREFIX):-len(IMAGE_EXTENSION)] + IMAGE_EXTENSION

            if expected_nrc_filename in nrc_files:
                print(f'[INFO] Found filenames {bfile} <--> {expected_nrc_filename} matching with each other.')
                file_pairs.append((bfile, expected_nrc_filename))
    else:
        if len(args.filenames) % 2 != 0:
            parser.error('Filenames must be provided in pairs!')
        
        img1s = args.filenames[::2]
        img2s = args.filenames[1::2]

        for filename in img1s:
            bname = os.path.basename(filename)

            if not bname.startswith(BASELINE_PATH_PREFIX) or not bname.endswith(IMAGE_EXTENSION):
                parser.error('Filenames are invalid.')
        
        for filename in img2s:
            bname = os.path.basename(filename)

            if not bname.startswith(NRC_PATH_PREFIX) or not bname.endswith(IMAGE_EXTENSION):
                parser.error('Filenames are invalid.')

        for img1name, img2name in zip(img1s, img2s):
            img1bname = os.path.basename(img1name)
            img2bname = os.path.basename(img2name)
            
            if img1bname[len(BASELINE_PATH_PREFIX):-len(IMAGE_EXTENSION)] != img2bname[len(NRC_PATH_PREFIX):-len(IMAGE_EXTENSION)]:
                parser.error('Filenames are mismatched.')
        
            file_pairs.append((img1name, img2name))

    process(file_pairs)
