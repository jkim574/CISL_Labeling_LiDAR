# Title: .pcd to .npy file converter (multi file)
# Author: Mason Audet
# Date: 7/11/24
# Purpose: Converts a .pcd file (ascii or binary) into a .npy file that can then be used
#          as custom data in the OpenPCDet environment. Adds in a 4th demension (intensity)
#          by default to make it compatible with Kitti standards
#
# Usage: python pcd_to_npy.py --input_file <input_file_path> --output_file <output_file_path>
# NOTE: The paths for the input and output folder containing the files must end in a "/" to work properly
#
# Update tracker:
# 7/17/24: Achala Pandit & Melanie Gomez: Updated the script to convert all .pcd files in <input_file_path> folder to .npy files and save in <output_file_path> folder.


import open3d as o3d
import numpy as np
import argparse
from pathlib import Path
import glob

# Initialize the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input_file', type=str, default=None, help='Specify the input file location')
parser.add_argument('--output_file', type=str, default=None, help='Specify the output file location (new file name)')
args = parser.parse_args()

data_file_list = glob.glob(args.input_file+'*')

for file in data_file_list:
    print(file)
    pcd = o3d.io.read_point_cloud(file)
    o3d.io.write_point_cloud(file, pcd, write_ascii = True, print_progress = True)
    
    # Open the new .pcd file we just made
    pcd = o3d.io.read_point_cloud(file)
    point_array = np.asarray(pcd.points)
    
    # Adds in blank values for the "Intensity" coordinate that is required
    intensity_col = np.zeros((point_array.shape[0], 1))
    final_point_array = np.hstack((point_array, intensity_col))
    
    final_point_array[:, 0] *= -1 # Rotates the x axis by 180 degrees to match kitti
    final_point_array[:, 1] *= -1 # Rotates the y axis by 180 degrees to match kitti
    file_splits = file.split('/')
    dest = args.output_file+ file_splits[-1].split('.')[0]+'.npy'
    print(dest)
    np.save(dest, final_point_array)
