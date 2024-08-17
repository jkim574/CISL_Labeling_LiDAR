# Project Pipeline: Data Fusion Approach
## Created by Mason Audet
The figure below represents our intended data pipeline from raw LiDAR data to data-merged LiDAR labels that allow for higher prediciton accuracy.

```

(1) Raw data collected by CISL (in .db3 file format)  --> (2) db3_to_npy_png.py --> (3) demo.py -->

(4) get_predictions.py --> (5) lidar_model_fusion.py --> (6) vis_merge_pred.py

```

#######################################################################################

**(1)** Our team is given the raw data collected by CISL. This data comes in the form of a ROSbag (.db3) and contains timestamped, gps marked image and LiDAR data.


**(2)** The .db3 file for a vehicle's perspective in a scene is unpacked using a conversion script called `db3_to_npy_png.py` **(Created by Mason Audet and Bo Wu)**. This provides us with point cloud data (in the form of .npy files) along with .png images corresponding to each LiDAR frame. 
This script also adds a 4th dimension for intensity to each point in the **.npy** file and sets the value to "0". This is because the sensors used for data collection do not record intensity, but an intensity element is required to match the Kitti format. In addition, the "x" and "y"
coordinates in the CISL data are multipled by -1 in order to rotate them 180 degrees. This needs to be done otherwise the perspective of the collecting vehicle will be incorrect and you will only see data collected behind the car instead of in front. 


**Please note: The script for unpacking the .db3 file needs to be ran in a certain environment to work. We used a docker setup provided by Hang Qiu.**


An example of the translation is shown below.

```
 [x1, y1, z1]       [x1, y1, z1, i1]        [-x1, -y1, z1, i1]
       …        →           …          →            …
 [xn, yn, zn]       [xn, yn, zn, in]        [-xn, -yn, zn, in]

     (Raw)             (3D to 4D)               (Oriented)

```


**(3)** Now that we have the LiDAR data in the form of **.npy** files, we can leverage the OpenPCDet toolkit and use pretrained models to obtain predictions on the locations of cars, cyclists, and pedestrians in each frame of the scene. To do this, the existing `open3d_vis_utils.py` located in
`/OpenPCDet/tools/visual_utils/` must be replaced with the script from our repo (with the same name). Once the new file is inserted, `demo.py` can be run on a single **.npy** file from the folder created in **step 3**. Since our data is modifed to match the Kitti format, use a config file from the `kitti_models` folder and set a checkpoint based on the model selected for your config.

```
python demo.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt /OpenPCDet/tools/checkpoints/pv_rcnn_8369.pth \
    --data_path <path_to_input_file> \
    --ext .npy
```

After running the command, a 3D rendering with labeled data should appear. On each of the labels, the model's assigned confidence score is attatched in red text. If you did everything correctly at this point, the direction of the red arrow should correspond with the front
of the collecting vehicle, the green arrow should point to the left of the collecting vehicle, and the blue arrow should be pointing up. After verifying that the data is in the correct format and is displaying properly, you can close the window.


**(4)** Now that the custom data has been validated through a visual test, the `get_predictions.py` script can be used to generate a **.csv** file containing data infos on each bounding box in each frame of the scene. The execution of this script will be similar to that of `demo.py` 
except you will need to run this at least twice (using a different model each time).

```
# Example of generating a .csv on the first model
python get_predictions.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/pv_rcnn.yaml \
    --ckpt /OpenPCDet/tools/checkpoints/pv_rcnn_8369.pth \
    --data_path <path_to_input_file> \
    --out_path <path_to_output_location> \ 
    --ext .npy

# Example of generating a .csv on the second model
python get_predictions.py --cfg_file /OpenPCDet/tools/cfgs/kitti_models/<other_model>.yaml \
    --ckpt /OpenPCDet/tools/checkpoints/<other_model>.pth \
    --data_path <path_to_input_file> \
    --out_path <path_to_output_location> \
    --ext .npy
```


**(5)** `lidar_model_fusion.py` utilizes a modified version of `ensemble_boxes_wbf_3d.py` from `Weighted Boxes Fusion`. It will read in two **.csv** files created in the previous step and place data into three different arrays (one for the label definitions in the form of [x, y, z, l, w, h, r],
one for each labels class, and one for each labels confidence score). This data is then passed to the modified version of `ensemble_boxes_wbf_3d.py`, where bounding boxes with confidence scores lower than a set threshold are removed from the dataset. After this, dependencies from the `BBox`
library are utilized to perform a Jaccards Index on the 3D bounding boxes to determine which ones overlap with eachother. Those that overlap are then passed to another function which picks the bounding box with the highest confidence score to represent the object in question. All this data
is then outputted in the terminal and saved into a **.csv** file. The result is a set of labels for each scene that utilizes the best predictions for each object detected by the input models. 

```
# Example for running lidar_model_fusion.py
python lidar_model_fusion.py --model_1_in /home/<USER>/OpenPCDet/tools/1719945777_337186420_prediction_frame_PV_RCNN.csv \
    --model_2_in /home/<USER>/OpenPCDet/tools/1719945777_337186420_prediction_frame_A2_free.csv \
    --output_file fused_model_labels.csv
```


**(6)** The **.csv** file created by `lidar_model_fusion.py` can then be visualized using our `vis_merge_pred.py` script. For each **.npy** file listed in the fused **.csv** file, a window will populate that displays the point cloud and corresponding label data.

```
# Example for running vis_merge_pred.py
python vis_merge_pred.py --label_path /home/<USER>/OpenPCDet/tools/fused_model_labels.csv
```
