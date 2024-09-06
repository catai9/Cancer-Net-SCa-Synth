import math
import argparse

import cv2
import numpy as np
import pandas as pd

from os import listdir
from os.path import isfile, join
from time import perf_counter


parser = argparse.ArgumentParser(description="Generate csv file for the given raw generated folder")
parser.add_argument(
    "--raw_folder_location",
    type=str,
    default=None,
    required=True,
    help="Location to folder.",
)

parser.add_argument(
    "--csv_location",
    type=str,
    default=None,
    required=True,
    help="Location to csv file.",
)

args = parser.parse_args()

start_time = perf_counter()
print("Starting script")

df = pd.DataFrame(columns=['image_name', 'target'])

onlyfiles = [f for f in listdir(args.raw_folder_location) if isfile(join(args.raw_folder_location, f))]


# Load Labels
for idx, file_name in enumerate(onlyfiles):
    print(f"File name: {file_name}")
    if file_name[:2] == 'be':
    	curr_target = 0
    else:
    	curr_target = 1
    df.loc[idx] = [file_name[:-4], curr_target]
    

df.to_csv(args.csv_location, encoding='utf-8', index=False)

end_time = perf_counter()
elapsed_time = end_time-start_time

print(f'Total time: {elapsed_time}')
