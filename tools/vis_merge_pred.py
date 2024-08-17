"""

Title: vis_single_merge_pred.py
Author: Mason Audet
Date: 7/27/24

Description: This is a script that takes a .csv file as input and displays the LiDAR points along with the 
	     label boxes drawn on them for each frame detailed in the .csv file. This is a tool to help 
	     visualize the difference between two models predictions and the confidence-based model fusion results.

"""


import argparse
import pandas as pd
import open3d
from visual_utils import open3d_merge_vis_utils as V
import numpy as np

# Parser that creates an arg for file input
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--label_path', type=str, default='demo_data',
                        help='specify the .csv file with the label data')

    args = parser.parse_args()

    return args


def main():
    args = parse_config()
    
    # Read in the .csv file
    labels_csv = pd.read_csv(args.label_path)
    
    # Makes a list of unique file names
    unique_values = labels_csv['filename'].unique().tolist()
    
    for value in unique_values:
        matching_rows = labels_csv[labels_csv['filename'] == value]
        
        # Stores data from a matching row into local variables
        labels = matching_rows[['x', 'y', 'z', 'l', 'w', 'h', 'r']].values.tolist()  # Include r in the main coords
        scores = matching_rows['scores'].values.tolist()
        label_class = matching_rows['label'].values.tolist()
        file_name = matching_rows['filename'].values.tolist()
        
        # Converts the box definitions and class labels into arrays
        label_array = np.array(labels)
        label_class_array = np.array(label_class)
        
        # Converts the class number (1, 2, 3) into an int.
        label_class_array = label_class_array.astype(np.int64)
        
        # Loads in the points from the .npy file referenced for this set of labels
        points_from_csv = np.load(file_name[0])
        
        # Sends point cloud data, 3D box definitions, confidence scores, and corresponding 
        # labels to create a window displaying the scene, bounding boxes, and confidence scores
        V.draw_scenes(
        points=points_from_csv, ref_boxes=label_array,
        ref_scores=scores, ref_labels=label_class_array
        )


if __name__ == '__main__':
    main()
