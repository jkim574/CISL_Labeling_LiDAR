# coding: utf-8
__author__ = 'ZFTurbo: https://kaggle.com/zfturbo'

"""

Modified By: Mason Audet
Date: 7/27/24

NOTES:  This is an altered version of the original ensemble_boxes_wbf_3d.py
	It has an additional dependency that must be downloaded in order for it to run properly
		* The library being referenced is the BBox library and more specifically the BBox_3D class and jaccard_index_3d function
		* Because of this, the original bb_intersection_over_union_3d function has been removed and replaced with a call to the jaccard_index_3d function

"""


import warnings
import numpy as np
from numba import jit
from bbox import BBox3D
from bbox.metrics import iou_3d, jaccard_index_3d


#@jit(nopython=True) # Removed jit for testing

def prefilter_boxes(boxes, scores, labels, weights, thr):

    """
    NOTES: Takes in the boxes, scores, labels, weights, and thr values as parameters and returns an
           array of label definitions in the form (class_label, conf_score, x, y, z, l, w, h, r).
           * 'thr' is used to filter out any labels with a confidence score < thr
           * 'weights' dont really matter for our application and is set to '1' 
    """

    # Create dict with boxes stored by its label
    new_boxes = dict()

    for t in range(len(boxes)):

        if len(boxes[t]) != len(scores[t]):
            print('Error. Length of boxes arrays not equal to length of scores array: {} != {}')#.format(len(boxes[t]), len(scores[t])))
            #exit() REMOVED ERROR CHECKING TEMPORARILY

        if len(boxes[t]) != len(labels[t]):
            print('Error. Length of boxes arrays not equal to length of labels array: {} != {}')#.format(len(boxes[t]), len(labels[t])))
            #exit() REMOVED ERROR CHECKING TEMPORARILY

	# Code below has been changed to accept input for the new format (x, y, z, l, w, h, r)

        for j in range(len(boxes[t])):
            score = scores[t][j]
            if score < thr:
                continue
            label = int(labels[t][j])
            box_part = boxes[t][j]
            x = float(box_part[0])
            y = float(box_part[1])
            z = float(box_part[2])
            l = float(box_part[3])
            w = float(box_part[4])
            h = float(box_part[5])
            r = float(box_part[6])
               
            b = [int(label), float(score) * weights[t], x, y, z, l, w, h, r]   

            if label not in new_boxes:
                new_boxes[label] = []
            new_boxes[label].append(b)

    # Sort each list in dict by score and transform it to numpy array
    for k in new_boxes:
        current_boxes = np.array(new_boxes[k])
        new_boxes[k] = current_boxes[current_boxes[:, 1].argsort()[::-1]]

    return new_boxes


def get_weighted_box(boxes, conf_type='avg'):
    """
    Create weighted box for set of boxes
    :param boxes: set of boxes to fuse
    :param conf_type: type of confidence one of 'avg' or 'max'
    :return: weighted box
    
    NOTES: Under the new implementation, this function will select the box from each
    	   list of overlapping boxes that has the highest confidence score and return
    	   the label info. conf_type is no longer used in this function and will be
    	   removed in the future.
    
    """
    
    # The box size is changed from 8 elements to 9 elements to account for the rotation

    box = np.zeros(9, dtype=np.float64) # Create a blank placeholder for the chosen box
    conf = 0.0
    for b in boxes:
        if b[1] > conf: # if the confidence score of the current box is greater than the previous one
            conf = b[1] # Set the new max for confidence equal to the current boxes confidence
            box[0:] = b[0:] # Set the values of indices in the return box equal to equal to those of the current box 
    
    return box


def find_matching_box(boxes_list, new_box, match_iou):

    """
    NOTES: This function takes in the boxes_list along with the current box being compared (new_box)
           and a threshold for the minimum IoU score. It uses classes and functions from the BBox
           library to find the IoU for the two boxes being compared. jaccard_index_3d accepts our
           heading angle 'r' as input, allowing us to account for the 3D boxes angle when finding 
           the Intersection over Union.
    """

    best_iou = match_iou
    best_index = -1
    for i in range(len(boxes_list)):
        box = boxes_list[i]
        if box[0] != new_box[0]:
            continue
            
        # This portion has been modified to use the jaccard_index_3d function.
        # It takes the arguments: (a, b) where a and b are BBox3D objects
        # BBox3D objects must be created and then passed into the function here
            
        a = BBox3D(box[2], box[3], box[4], length=box[5], width=box[6], height=box[7], euler_angles=(0.0, 0.0, box[8] + 1e-10)) 
        b = BBox3D(new_box[2], new_box[3], new_box[4], length=new_box[5], width=new_box[6], height=new_box[7], euler_angles=(0.0, 0.0, new_box[8] + 1e-10))
            
        iou = jaccard_index_3d(a, b)
        if iou > best_iou:
            best_index = i
            best_iou = iou

    return best_index, best_iou


def weighted_boxes_fusion_3d(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0, conf_type='avg', allows_overflow=False):
    '''
    :NEW param boxes_list list of boxes predictions from the models
    Incoming format will be (x, y, z, l, w, h, r)
    
    NEW Order of boxes: x, y, z, l, w, h, r. We will attempt without normalized coordinates at first, and then test with normalized cords after.
    
    :param scores_list: list of scores for each model
    :param labels_list: list of labels for each model
    :param weights: list of weights for each model. Default: None, which means weight == 1 for each model
        ** We don't use these and the code can be modified to work without them.
    :param iou_thr: IoU value for boxes to be a match
    :param skip_box_thr: exclude boxes with score lower than this variable
    :param conf_type: how to calculate confidence in weighted boxes. 'avg': average value, 'max': maximum value
    	** We don't use this and it can be removed in the future
    :param allows_overflow: false if we want confidence score not exceed 1.0

    :NEW return: boxes: boxes coordinates (Order of boxes: x, y, z, l, w, h, r).
    :return: scores: confidence scores
    :return: labels: boxes labels
    '''

    if weights is None:
        weights = np.ones(len(boxes_list))
    if len(weights) != len(boxes_list):
        print('Warning: incorrect number of weights {}. Must be: {}. Set weights equal to 1.'.format(len(weights), len(boxes_list)))
        weights = np.ones(len(boxes_list))
    weights = np.array(weights)

    if conf_type not in ['avg', 'max']:
        print('Error. Unknown conf_type: {}. Must be "avg" or "max". Use "avg"'.format(conf_type))
        conf_type = 'avg'

    filtered_boxes = prefilter_boxes(boxes_list, scores_list, labels_list, weights, skip_box_thr)
    if len(filtered_boxes) == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,))

    overall_boxes = []
    for label in filtered_boxes:
        boxes = filtered_boxes[label]
        new_boxes = []
        weighted_boxes = []

        # Clusterize boxes
        for j in range(0, len(boxes)):
            index, best_iou = find_matching_box(weighted_boxes, boxes[j], iou_thr)
            if index != -1:
                new_boxes[index].append(boxes[j])
                weighted_boxes[index] = get_weighted_box(new_boxes[index], conf_type)
            else:
                new_boxes.append([boxes[j].copy()])
                weighted_boxes.append(boxes[j].copy())

        overall_boxes.append(np.array(weighted_boxes))

    overall_boxes = np.concatenate(overall_boxes, axis=0)
    overall_boxes = overall_boxes[overall_boxes[:, 1].argsort()[::-1]]
    boxes = overall_boxes[:, 2:]
    scores = overall_boxes[:, 1]
    labels = overall_boxes[:, 0]
    return boxes, scores, labels
