# Adapted from https://www.kaggle.com/code/xhlulu/training-mobilenet-v2-in-4-min
import math
import argparse

import cv2
import numpy as np
import pandas as pd

from time import perf_counter


def pad_and_resize(image_path, pad=True, desired_size=224):
    def get_pad_width(im, new_shape, is_rgb=True):
        pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
        t, b = math.floor(pad_diff[0]/2), math.ceil(pad_diff[0]/2)
        l, r = math.floor(pad_diff[1]/2), math.ceil(pad_diff[1]/2)
        if is_rgb:
            pad_width = ((t,b), (l,r), (0, 0))
        else:
            pad_width = ((t,b), (l,r))
        return pad_width
        
    img = cv2.imread(image_path)
    
    if pad:
        pad_width = get_pad_width(img, max(img.shape))
        padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        padded = img
    
    padded = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(padded, (desired_size,)*2).astype('uint8')
    
    return resized


parser = argparse.ArgumentParser(description="Input to train mobilenetv2 model script")
parser.add_argument(
    "--raw_folder_location",
    type=str,
    default=None,
    required=True,
    help="Location to train folder.",
)

parser.add_argument(
    "--csv_location",
    type=str,
    default=None,
    required=True,
    help="Location to train csv file.",
)

parser.add_argument(
    "--processed_output_folder",
    type=str,
    default=None,
    required=True,
    help="Output folder for the processed train files.",
)

args = parser.parse_args()

start_time = perf_counter()
print("Starting script")

# Load Labels
df = pd.read_csv(f'{args.csv_location}')

for image_id in df['image_name']:
    print(f"Processing image_id: {image_id}")
    img = pad_and_resize(f'{args.raw_folder_location}/{image_id}.jpg')
    np.save(f"{args.processed_output_folder}/{image_id}.npy", img)

end_time = perf_counter()
elapsed_time = end_time-start_time

print(f'Total time: {elapsed_time}')
