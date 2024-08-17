# Title: .pcd to .bin file converter (single file)
# Author: Alex Pasquali (Modified by Mason Audet)
# Date: 7/21/24
# Purpose: Converts a .pcd file (ascii or binary) into a .bin file that can then be used
#          as custom data in the OpenPCDet environment. Adds in a 4th demension (intensity)
#          by default to make it compatible with Kitti standards
#
# Usage: python pcd2bin.py <input_file_path> <output_file_path>
#
# Notes:
#	 This script is built off of code originating from Alex Pasquali and the original
#	 code can be found on his github at AlexPasqua.

    
import os
import argparse
import open3d as o3d
import numpy as np
from typing import Union
from open3d.geometry import PointCloud


def read_pcd(path: str) -> np.ndarray:
    """
    Reads a pointcloud with open3d and returns it as a numpy ndarray

    Args:
        path (str): path to the pointcloud to be read or the directory containing it/them

    Returns:
        np.ndarray: the pointcloud(s) as a numpy ndarray (dims: (pointcloud x) points x coordinates)
    """
    if os.path.isdir(path):
        pointclouds = []
        filenames = os.listdir(path)
        for filename in filenames:
            if filename[-4:] != '.pcd':
                continue
            pcd = o3d.io.read_point_cloud(path)
            
            # Write a new point cloud file saved with same name
            o3d.io.write_point_cloud(path, pcd, write_ascii = True, print_progress = True)
            
            # Open new file and append to list
            pcd = o3d.io.read_point_cloud(path)
            pointclouds.append(np.asarray(pcd.points))
            
        return np.array(pointclouds)
    
    elif os.path.isfile(path):
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)


def pcd_to_bin(pcd: Union[str, PointCloud, np.ndarray], out_path: str):
    """
    Convert pointcloud from '.pcd' to '.bin' format

    Args:
        pcd (Union[str, PointCloud, np.ndarray]): the pointcloud to be converted (either its path, or the pointcloud itself)
        out_path (str): the path to the destination '.bin' file
    """
    # if 'pcd' is a string, assume it's the path to the pointcloud to be converted
    if isinstance(pcd, str):
        pcd = read_pcd(path=pcd)
    # save the poinctloud to '.bin' format
    out_path += "" if out_path[-4:] == ".bin" else ".bin"
    
    # Adds in blank values for the "Intensity" coordinate that is required
    intensity_col = np.zeros((pcd.shape[0], 1))
    final_point_array = np.hstack((pcd, intensity_col))
    
    final_point_array[:, 0] *= -1 # Rotates the x axis by 180 degrees to match kitti
    final_point_array[:, 1] *= -1 # Rotates the y axis by 180 degrees to match kitti
    
    final_point_array.astype('float32').tofile(out_path)


def convert_all(in_dir_path: str, out_dir_path: str):
    """
    Converts all the pointclouds in the specified drectory from '.pcd' to '.bin' format

    Args:
        in_dir_path (str): path of the directory containing the pointclouds tio be converted
        out_dir_path (str): path of the directory where to put the resulting converted pointclouds

    Raises:
        ValueError:
            if 'in_dir_path' is neither a regular file nor a directory;
            if 'out_dir_path' is not a directory.

    Returns:
        np.ndarray: the converted pointclouds as numpy ndarrays
    """
    if not os.path.isdir(out_dir_path):
        raise ValueError(f"{out_dir_path} is not a directory.")

    if os.path.isfile(in_dir_path):
        file_name = in_dir_path.split('/')[-1]
        out_path = os.path.join(out_dir_path, file_name)
        pcd_to_bin(pcd=in_dir_path, out_path=out_path)     # in_dir_path is not a directory but a pointcloud in this case
    
    elif os.path.isdir(in_dir_path):
        pointclouds = []
        filenames = os.listdir(in_dir_path)
        for filename in filenames:
            if filename[-4:] != '.pcd':
                continue
            out_path = os.path.join(out_dir_path, filename[:-4] + '.bin')
            pcd_to_bin(pcd=os.path.join(in_dir_path, filename), out_path=out_path)

    else:
        raise ValueError(f"'{in_dir_path} is neither a directory or file")
        
def main():

    parser = argparse.ArgumentParser(description="Converts a single or multiple pointclouds form '.pcd' to '.bin' format.")
    parser.add_argument('in_path', type=str, action='store',
        help="The path to either the '.pcd' pointcloud to convert to '.bin' or the path to a directory containing various of these pointclouds")
    parser.add_argument('out_path', type=str, action='store',
        help="if in_path is a file, then out_path must be the desired name of the converted pointcloud; if in_path was a directory, then \
        'out_path must be the name of the directory where to put all the converted pointclouds")
    args = parser.parse_args()

    convert_all(in_dir_path=args.in_path, out_dir_path=args.out_path)


if __name__ == '__main__':
    main()
    
