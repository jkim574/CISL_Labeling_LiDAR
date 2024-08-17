## OpenPCDet
The project mainly revolves around using a github repository known as OpenPCDet: (https://github.com/open-mmlab/OpenPCDet). It is a toolkit for lidar-based point cloud labeling and modeling. It utilizes Open3D for visualization of the point clouds. It has native support for .bin and .npy files, and other data formats like .bin or .ros can be supported by converting to the natively supported types through scripts, which you should be able to find in this repository (not added yet). 

We have used OpenPCDet for two primary purposes so far: visualization and automated labeling. OpenPCDet has support for a variety of diffrent "models" whose purpose is to analyse the point clouds and identify objects in them, and classify them into 3 categories: Car, cyclist, or pedestrian. These classifications are represented both visually and numerically through "bounding boxes", which are essentially a collection of coordinates with respect to the viewer (the exact technical representation of the bounding boxes will be covered later.) All of the different models have different accuracies for the different categories, with some better in categories like pedestrians or cycles, while other are better in cars. The main difference amongst a few of these models is the pre-processing techniques of the point clouds, such as the models relying on anchoring vs non anchoring techniques. You can find more information about the technical specifications of the models on the repository's link. 

## Visualisation of Outputs
This section showcases various images generated during the project using different scripts.

Visualising the confidence weights with colour coded boxes, with green >0.66, yellow > 0.33, and blue < 0.33. Below is an example of OpenPCDet's confidence visualisation.

![image](https://github.com/user-attachments/assets/a29ca853-6ecf-4eae-a008-2167547e2fbd)

These visualisations help in understanding the confidence levels and accuracy of object detection in LiDAR data.

![image](https://github.com/user-attachments/assets/0aac44f0-9630-4bbc-8e1a-cb4a62b438f2)



Progress with MMdetection3d for image analysis and prediction for multi sensor fusion:
![image](https://github.com/user-attachments/assets/ab93d617-8a05-41a9-a897-0915d07e5df0)
