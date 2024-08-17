# Title: LiDAR Model Fusion
# Author:  ZFTurbo (Modified By Joong Ho Kim, Melanie Gomez, Achala Pandit and Mason Audet)
# Date: 7/24/24
# Purpose: Openpcdet outputs the coordinates in the form of x, y, z, l, w, h, r however weighed boxes requires 
#          input of x_1, y_1, z_1, x_2, y_2, z_2 in order to perform model fusion. In order to remedy this,
#	   we made some necessary changes to the underlying functionality of Weighted Boxes Fusion. This script
#	   can now take multiple .csv files containing label data as input and return a .csv file with the best predictions
#	   from both models. When multiple models predict a label in a very similar area, this program detects it and chooses
#	   the label with the highest accuracy to represent the object. The result is an increase in automated labeling accuracy.
#
#
# Updates:
#
#	7/27/24: Mason Audet refitted this script to work with the newly modified ensemble_boxes_3d.py file from WBF.
#		 Currently testing without normalization as I don't know how those values will effect underlying
#		 dependencies (bbox). This script will now take in data from raw KITTI format (x, y, z, l, w, h, r),
#		 combine the two incoming sets of csv files, and send it to the weighted_boxes_fusion_3d function.
#
#
#	7/28/24: Mason Audet made a few final adjustments to ensure that the script runs as intended, added documentation,
#		 and removed and functions that were no longer needed.
#
#
#    8/01/24: Joong Ho Kim modified this file by adding support for multiple input files instead of just taking only 2 input files. 
#             Also added optional flags to set the IoU and skip box values.
#
#
#
# Usage: python lidar_model_fusion.py --input_files <input_file_1> <input_file_2> ... --output_file <output_file_name> [--iou_thr <iou_threshold>] [--skip_box_thr <skip_box_threshold>]


import pandas as pd
import numpy as np
from ensemble_boxes import *
import argparse
import math

# coding: utf-8

def fuse_models(input_files, output, iou_thr=0.55, skip_box_thr=.75):
    """
    NOTES: This function takes in multiple csv files as input, a name for an output file, the IoU threshold
    to determine if two 3D labels are representing the same object, and a threshold for box
    skipping that sets a minimum confidence score for the label to be included. It will print
    the final list of labels in the terminal and save the results to a .csv file.
    """
    
    # Read all input CSV files
    models = [pd.read_csv(file) for file in input_files]
    
    filename_ext = []
    boxes_ext = []
    scores_ext = []
    labels_ext = []
    result_df = pd.DataFrame()
    
    # Get unique filenames from all input files
    unique_values = set()
    for model in models:
        unique_values.update(model['filename'].unique())
    unique_values = sorted(unique_values)
    
    # For each file (scene) in the set of LiDAR scenes, the best predictions are combined and saved
    for value in unique_values:
        boxes_list = []
        scores_list = []
        labels_list = []
        
        for model in models:
            matching_rows = model[model['filename'] == value]
            
            cords = matching_rows[['x', 'y', 'z', 'l', 'w', 'h', 'r']].values.tolist()
            scores = matching_rows['scores'].values.tolist()
            labels = matching_rows['label'].values.tolist()
            
            if cords:  # Only add non-empty lists
                boxes_list.append(cords)
                scores_list.append(scores)
                labels_list.append(labels)
        
        if not boxes_list:  # Skip if no boxes for this filename
            continue
        
        # Call to the modified WBF function that will handle the merging operations
        boxes, scores, labels = weighted_boxes_fusion_3d(boxes_list, scores_list, labels_list, weights=None, iou_thr=iou_thr, skip_box_thr=skip_box_thr, conf_type='max')
        
        # Combines and concatenates data into one cohesive data frame
        df_boxes = pd.DataFrame(np.asarray(boxes), columns=['x', 'y', 'z', 'l', 'w', 'h', 'r'])
        df_labels = pd.DataFrame(np.asarray(labels), columns=['label'])
        df_scores = pd.DataFrame(np.asarray(scores), columns=['scores'])
        if len(boxes) == 0:
            df_file = pd.DataFrame(np.asarray([value] * 1), columns=['filename'])
        else:
            df_file = pd.DataFrame(np.asarray([value] * len(boxes)), columns=['filename'])
        combined_df = pd.concat([df_file, df_labels, df_scores, df_boxes], axis=1)
        result_df = pd.concat([result_df, combined_df], axis=0)
    
    print(result_df)
    
    # Stores the data frame to a .csv file
    result_df.to_csv(output, index=False)

if __name__ == '__main__':
    # Initialize the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_files', nargs='+', type=str, required=True, help='Specify the input csv file names')
    parser.add_argument('--output_file', type=str, required=True, help='Specify the output file name')
    parser.add_argument('--iou_thr', type=float, default=0.7, help='IoU threshold (default: 0.7)')
    parser.add_argument('--skip_box_thr', type=float, default=0.85, help='Skip box threshold (default: 0.85)')
    args = parser.parse_args()
    
    # Call to the fuse_models function
    fuse_models(args.input_files, args.output_file, iou_thr=args.iou_thr, skip_box_thr=args.skip_box_thr)

