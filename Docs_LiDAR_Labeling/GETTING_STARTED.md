# Getting Started

## Background
All of the information detailed in the following manuscript is a result of the efforts from Team 8 in the DS-PATH Summer 2024 Cohort. Our team consisted of **Mason Audet**, **Melanie Gomez**, **Joong Ho Kim**, **Aaryan Kumar**, and **Achala Pandit**. Our two guiding faculty members were **Zhiwu Xie** and **Hang Qiu** from University of California, Riverside.
All python files listed in this repository were built by members of our team either from scratch or from existing open source code. Credit for any of our modified open source code should go to the original authors (OpenMMLab and Weighted Boxes Fusion).


LiDAR (Light Detection and Ranging) is a technology used to measure distances via lazer by projecting light towards a target and measuring the time taken for the light to reflect back to the sensor. It has many uses, but a recent focus has been using LiDAR to give self-driving cars precise, three-dimensional maps of their surroundings. This project focused on exploring fusion methods to increase the accuracy of label predictions on point cloud data. We took on the task of finding a way to implement Model Fusion, but in the future we would be open to exploring Sensor Fusion and Agent Fusion.


## Section 1: Environment Setup

### Operating System, Hardware Requirements, and Driver Versions
In order to replicate our results, it is important to first replicate our environment. The operating system we used was Ubuntu 22.04. Most modern desktops with decent hardware will be capable of running the programs used. The only specific hardware requirement is **an Nvidia graphics card because OpenPCDet uses CUDA**.
The Nvidia Driver version will vary based on the installed GPU, but ours is 535.183.01. From the terminal, you will need to install the CUDA Toolkit with `sudo apt install nvidia-cuda-toolkit`. The version that is supposed to install in Ubuntu 22.04 is CUDA Toolkit 11.5.119. You can check to see if the toolkit installed 
correctly by running `nvidia-smi` in the terminal. You should receive a description similar to the one below.

![image](https://github.com/user-attachments/assets/51c68420-120b-4442-8b2b-303e5783b4d3)


### Setting Up Conda
The rest of the dependencies we installed were in a virtual environment. We used Miniconda3 to create an environment thats compatible with OpenPCDet. The latest version of Miniconda3 can be installed using the commands below:
```
apt update
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /opt/miniconda-installer.sh
bash /opt/miniconda-installer.sh
source ~/.bashrc
conda update --all
```


#### Quick Note for Conda
When you open a new terminal, you are in the `base` environment by default. For most purposes, you will need to be inside the `openpcdet` environment created above. To enter the environment, you must run `conda activate openpcdet` in the terminal. 
The easiest way to tell if you are in the environment or not is by checking the text at the front of your command line prompt. 

If the text says `base`, you are executing commands using resources from the `base` environment. If the text says `openpcdet`, you are executing commands using the dependencies you install. 

When you want to enter the `openpcdet` environment, use `conda activate openpcdet`. When you want to return to the `base` environment, use `conda deactivate`. Please see the photo below for reference:

![image](https://github.com/user-attachments/assets/a7ed284c-64be-4bd0-8c6c-1d506de8c294)


Before creating your Miniconda3 environment, you will have to install GCC: 
```
sudo apt install gcc
```
Now that you have Miniconda3 and GCC installed, you can use the commands below to create an environment with Python 3.8 and activate it.
```
conda create -n openpcdet python=3.8
conda activate openpcdet
```

While still inside the conda environment, you will need to install Pytorch 2.0.1 and Cuda 11.7 using the command below.
```
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```


### Cloning OpenPCDet
If you do not already have git on your machine, install it with `sudo apt install git`. Once installed, open a new terminal (ctrl + alt + t) and run the following command:
```
git clone https://github.com/jkim574/CISL_Labeling_LiDAR.git
cd CISL_Labeling_LiDAR
```

After cloning the repository and navigating to the OpenPCDet directory, you need to ensure that the required dependencies are correctly installed. Specifically, the Pillow package needs to be handled with care to avoid potential conflicts. Before installing the required dependencies, uninstall any existing Pillow package to avoid conflicts. Then, install all required dependencies, including the correct version of Pillow:
```
pip uninstall pillow
pip install -r requirements.txt
```

Before running the `setup.py` script, you need to install and setup GCC-9. Run the following in a terminal to download GCC-9 and set it as the default for the build.
```
sudo apt list gcc-9
sudo apt update
sudo apt install gcc-9 g++-9
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
```
NOTE: You might have to run `sudo apt install gcc-9` and `sudo apt install c++-9` as separate commands if the above does not work for you.


The final step is to make sure you are in a terminal navigated to the OpenPCDet folder and run the setup command:

```
python setup.py develop
```



## Section 2: File Setup

### Creating Necessary Folders
Now that all the necessary dependencies are installed within the conda environment and base environment, there are a few modifications that need to be made to the `CISL_Labelig_LiDAR` file structure.
For starters, you will need to navigate to the `CISL_Labelig_LiDAR` folder and open the `tools` folder. In the tools folder, you will need to create a new folder titled `checkpoints`. This is where all of your checkpoint **.pth**
files will be stored and referenced from when using models to create predictions. It is also recommended that you create another folder titled `generated_labels` with two subfolders called `raw` and `fused`. When finished, your OpenPCDet folder should look like 
the structure below.
```
CISL_Labelig_LiDAR
│ ...
├── tools
│    ├── cfgs
│    ├── checkpoints
│    ├── eval_utils
│    ├── generated_labels
│    │    ├── raw
│    │    ├── fused
│    ├── process_utils
│    ├── scripts
│    ├── train_utils
│    ├── visual_utils
│    ├── _init_path.py
│    ├── demo.py
│    ├── test.py
│    ├── train.py
│ ...
```

### Replacing Existing Files
There are a file that need to be replaced in the `ensemble-boxes` library folder. The path will be similar to this: /home/<USER>/miniconda3/envs/openpcdet/lib/python3.8/site-packages/ensemble-boxes/. Once you are there, you are going to need to replace a file called `ensemble_boxes_wbf_3d.py` with the modified version from our repository (located in CISL_Labeling_LiDAR).
```
python3.8
│ ...
├── site-packages
│    ├── ...
│    ├── ensemble-boxes
│    │    ├── __pycache.py__
│    │    ├── ensemble_boxes_nms.py
│    │    ├── ensemble_boxes_pyth...
│    │    ├── ensemble_boxes_nmw.py
│    │    ├── ensemble_boxes_wbf.py
│    │    ├── ensemble_boxes_wbf_1d.py
│    │    ├── ensemble_boxes_wbf_3d.py (MODIFIED SCRIPT FROM OUR REPO)
│    │    ├── ensemble_boxes_wbf_experimental.py
│    │    ├── ensemble_boxes_wbf_pytorch.py
│    │    ├── __init__.py
│    ├── ...
│ ...
```

## Section 3: Steps for Use

### Step 1: Choosing Models
The first step in generating bounding box predictions is choosing the pretrained models that you want to use. Click the link [here](https://github.com/open-mmlab/OpenPCDet/tree/master?tab=readme-ov-file#model-zoo) to navigate to the OpenPCDet GitHub page. On this page you will find a header titled `Model Zoo`. Under the `KITTI 3D Object Detection Baselines` section, there is a list of supported models and links to download them (on the right hand side). Please download **at least two** of these models before moving forward.

![image](https://github.com/user-attachments/assets/6ad9c990-ad8f-4a0b-a436-a781ba66e365)

The models are in the form of **.pth** files and are now located in your `Downloads` folder. You will need to open your `Downloads` folder and copy these **.pth** files into the `checkpoints` folder you created earlier (located in `OpenPCDet/tools/`). Once you have copied the files into the folder, your structure should look like this:
```
CISL_Labelig_LiDAR
│ ...
├── tools
│    ├── cfgs
│    ├── checkpoints
│    │    ├── pv_rcnn_8369.pth
│    │    ├── PartA2_free_7872.pth
│    │    ├── (additional models)
│    ├── eval_utils
│    ├── generated_labels
│    ├── ...
│ ...
```


### Step 2: Converting from .db3 to .npy
This next part can be tedious and requires an environment that differs from the one we currently have set up on these machines. To avoid large online data transfers, we were given a docker file that links our local machine to the base station in CISL at UCR. We use the environment on the base station to unpack the **ROS2 .db3** file into LiDAR frames in **.npy** form and images in **.png** form. The unpacked files are placed into a folder that is shared between the base station and our system, allowing us to move the files and manipulate them in the following steps.

First, install Docker [Official Docker link]( https://docs.docker.com/engine/install/ubuntu/)

Then, download the [Dockerfile](https://github.com/kumaraaryan511/Labeling_LiDAR_data_project/blob/main/Code/Conversion_Scripts/ros2_humble_jammy.Dockerfile) from our GitHub.
```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Install the necessary dependencies by updating Dockerfile 'ros2_humble_jammy.Dockerfile'.
One thing to note is that there was a compatibility issue between NumPy 2.0.1 and other modules in the environment. To resolve this, I needed to downgrade NumPy to a version below 2.0.
```
FROM ros:humble-ros-base-jammy

# Install ROS packages and system dependencies
RUN apt-get update && apt-get install -y \
    ros-${ROS_DISTRO}-rosbag2 \
    ros-${ROS_DISTRO}-cv-bridge \
    ros-${ROS_DISTRO}-sensor-msgs-py \
    python3-pip \
    sudo \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    numpy==1.24.3 \
    opencv-python \
    open3d

# Create non-root user
RUN useradd -ms <home directory> \
    && usermod -aG sudo <user name> \
    && echo <password>

# Switch to non-root user
USER <user name>

# Set the entrypoint to bash
WORKDIR <work directory>
```
Creating a non-root user in the Dockerfile is an important security practice in containerized environments. When you enter the container, you'll be logged in as the <username> user, not root.

Now we can run Docker file by running command
```
docker build -t cisl/ros2:humble --file ros2_humble_jammy.Dockerfile .
docker run -it --rm -v "<Local path to db3 files>:/home/cisl/SensorPlatform/" cisl/ros2:humble
```

After this, we are now inside the container's shell, where we can start working with ROS2 and your .db3 files. Change your directory to where python script **db3_to_npy_png.py** and **ROS2 .db3** files are located. **db3_to_npy_png.py** file should be located in **~/CISL_Labeling_LiDAR/data/custom/ros_to_npy/ directory**. Then run this command to convert **.db3** file into **.npy** and **.png** files. 

```
python3 db3_to_npy_png.py --input_file <your db3 files> --output_file <output file path>
```
This will convert all **.db3** files to **.npy** and **.png** files in your designated directory.


### Step 3: Running demo.py to Verify Data Format
Now we will need to run `OpenPCDet`'s built in `demo.py` file in order to verify that the data was unpacked correctly and was properly oriented to match KITTI specifications. Before we do this, it would be wise to move your unpacked data to an area that is easier to reference. These next steps will help you keep the data organized and make the commands you run much shorter.


Start by navigating to the folder at `/CISL_Labelig_LiDAR/data`. Once inside, you will need to create a new folder called `cisl`. Within the `cisl` folder, create a folder with the date of collection (ex. `072124`). Each <date_of_collection> folder should have subfolders for `tower_#` and each `tower_#` folder should have subfolders for `lidar` and `image`. Once you have made these directories, you will need to add the unpacked data into the corresponding folders. When finished, your folders should look something like this:
```
CISL_Labelig_LiDAR
│ ...
├── data
│    ├── ...
│    ├── argo2
│    ├── kitti
│    ├── cisl
│    │    ├── 072124
│    │    │    ├── Tower_1
│    │    │    │    ├── run_1
│    │    │    │    │    ├── lidar
│    │    │    │    │    │    ├── <.npy point clouds from this date, tower, and run>
│    │    │    │    │    ├── image
│    │    │    │    │    │    ├── <.png images from this date, tower, and run>
│    │    │    │    ├── run_2
│    │    │    │    ├── run_3
│    │    │    ├── Tower_2
│    │    │    ├── Tower_3
│ ...
```


**Please make sure that your file structure matches the one above and ensure that your data is in the correct folders before moving forward**


To run demo.py, open a new terminal prompt with `(ctrl + alt + t)` and follow the commands below:
```
conda activate openpcdet
cd CISL_Labelig_LiDAR/tools
python demo.py --cfg_file /cfgs/kitti_models/<CONFIG_FILE_FOR_CHOSEN_MODEL.yaml> \
--ckpt /checkpoints/<CORRESPONDING_MODEL_FILE.pth> \
--data_path /CISL_Labelig_LiDAR/data/cisl/072124/Tower_1/run_1/lidar/ \
--ext .npy
```

Once you hit enter on the last command line from the section above, a rendering of the 3D point cloud and object labels should appear. You can `click + drag` the screen to move around and `scroll` to zoom in or out. Holding `(ctrl)` or `(ctrl + shift)` while you click and drag will allow you to move the camera perspective differently. Hitting `(esc)` or clicking the `"x"` at the top of the window will close the rendering and populate the next frame. This will repeat until either the sequence is finished or you click on the terminal window and hit `(ctrl + c)` twice to interupt and abort.

You want to use this opportunity to make sure that the model is taking the file input correctly. If you did everything correct at this point, your `x axis (red)` should point in the same direction the collecting vehicle is travelling, the `y axis (green)` should be pointing to the left of the car, and the `z axis (blue)` should be pointed upwards. OpenPCDet is set up to only let the models search for objects in the positive x, y, and z directions. This means that anything behind the car will not be detected. **It would be worth looking into this issue in the future to see if the model can read the entire point cloud and not just the trimmed version.** Once you are done viewing, close out the viewer using one of the methods mentioned above and move on.


### Step 4: Collecting Model Predictions with get_predictions.py
Now that we have confirmed the data to be formatted correctly, we can collect label predicitons for a given scene from multiple models. We will do this using our `get_predictions.py` script. Follow the example below, keeping in mind that you need at least two **.csv** files for fusion. You can generate more if you would like by following the same format and changing the models/:
```
conda activate openpcdet
cd CISL_Labelig_LiDAR/tools

python get_predictions.py --cfg_file /cfgs/kitti_models/<CONFIG_FILE_FOR_CHOSEN_MODEL.yaml> \
--ckpt /checkpoints/<CORRESPONDING_MODEL_FILE.pth> \
--data_path /CISL_Labelig_LiDAR/data/cisl/072124/Tower_1/run_1/lidar/ \
--output /generated_labels/raw/<NAME_OF_FILE>.csv \
--ext .npy

python get_predictions.py --cfg_file /cfgs/kitti_models/<CONFIG_FILE_FOR_CHOSEN_MODEL.yaml> \
--ckpt /checkpoints/<CORRESPONDING_MODEL_FILE.pth> \
--data_path /CISL_Labelig_LiDAR/data/cisl/072124/Tower_1/run_1/lidar/ \
--output /generated_labels/raw/<NAME_OF_FILE>.csv \
--ext .npy

...

```


### Step 5: Using lidar_model_fusion.py to Fuse Label Predictions from Multiple Models
In order to generate a **.csv** file containing fusion results from your input models, you will need to use our `lidar_model_fusion.py` script by following the commands below:

Keep in mind that `--iou_thr` sets the minimum IoU threshold and `--skip_box_thr` sets the minimum confidence score value to include a label.
```
conda activate openpcdet
cd CISL_Labelig_LiDAR/tools


python lidar_model_fusion.py --input_files <PATH_TO_FILE_1> <PATH_TO_FILE_2> ... \
--output_file /generated_labels/fused/<NAME_OF_FILE>.csv \
--iou_thr 0.25 \
--skip_box_thr 0.75
```


### Step 6: Visualizing New Bounding Boxes with vis_merge_pred.py
The last step is only for if you want to visualize the entire scene frame by frame. This program will take a **.csv** file as input (raw or fused) and quickly take screenshots of each frame from the scene as it populates. The settings to use this tool vary based on the monitor you are using and the resolution. Getting this visualization tool to work takes a bit of trial and error. You will need to start by running the commands below:
```
conda activate openpcdet
cd CISL_Labelig_LiDAR/tools

python vis_merge_pred.py --label_path /generated_labels/fused/<FUSED_FILE_NAME>.csv
```

As soon as you hit enter, a new screen will pop up and close every 0.1 seconds. This is **NORMAL**. Quickly click on the terminal you ran the command from and interupt it to stop the progress ([ctrl + c] twice).


Now you will need to open a Files window and navigate to `/OpenPCDet/tools/visual_utils/` to open the `open3d_merge_vis_utils.py` file. Open a second Files window and navigate to `/OpenPCDet/data/cisl/072124/Tower_1/run_1/lidar/`. In this folder, you should see that in addition to the **.npy** files, there are also **.png** files for the first few sets of scene frames. Open one of the images and evaluate the position of the screenshot versus the frame of lidar data. If it is not capturing correctly, you will need to make changes to the image capture function's bbox input in `open3d_merge_vis_utils.py`. The screenshot below shows the section of code you will need to modify:

![image](https://github.com/user-attachments/assets/9e672525-8726-4f54-a094-a475d729663b)


In the snippet above, the `TL` coordinates correspond to the `Top Left` of the visualizer window produced by `open3d_merge_vis_utils.py`. The `BR` coordinates correspond to the `Bottom Right` of the window. The `#` symbol on `line 130` needs to be placed in the same location as the photo above to prevent the first visualizer window from closing when you run `vis_merge_pred.py`. In order to decide on the values for the `TL` and `BR` coordinates, you will need to use the `Ubuntu Screenshot tool`. You can find it by clicking the 3x3 grid of dots in the bottom left corner and searching "screen". 

Once you have the screenshot tool open, run the visualizer tool like you did in the beginning of step 6 (python vis_merge_pred.py --label_path /generated_labels/fused/<FUSED_FILE_NAME>.csv). This time, the visualizer window for the first frame should open and remain open. You will now need to use the screenshot tool to take two screenshots.

For the first screenshot, position the `top left of the capture box` at the `top left of your screen`. Drag the `bottom right of the capture box` to the `top left of the visualizer window`. Take the screenshot.

![image](https://github.com/user-attachments/assets/229dc3b5-ed94-48a9-abb6-d6caf629d08c)


For the second screenshot, position the `top left of the capture box` at the `top left of your screen`. Drag the `bottom right of the capture box` to the `bottom right of the visualizer window`. Take the screenshot.

![image](https://github.com/user-attachments/assets/4a4f3ed8-20e3-4303-acdf-d8f8a6b846fe)


Once you have taken both screenshots, open a new Files window and navigate to `Pictures/screenshots/`. Open the two screenshots and under the options menu, click on `INESRT HERE` to view the dimension details of the photo. What you have in front of you should look similar to the image below:

![image](https://github.com/user-attachments/assets/11d9807e-8f48-44bf-b8a4-a58b998d2dbf)


The `size` of the image is what you are going to want to pay attention to. The `size` of the smaller image is roughly the coordinate of `TL` and the `size` of the larger image is roughly the coordinate of `BR`. If you haven't already, interrupt the terminal with [(ctrl + c) twice] to close the visualizer window. Open the `open3d_merge_vis_utils.py` script again, change the values for `bbox` with your new values, remove the `#` that commented out the `close_window()` function, and save the file.

Now you can run `vis_merge_pred.py` as you did before and the visualizer window should populate and close rapidly like before. Once the last visualizer window has closed, you can close the terminal. Open a Files window and navigate to `/OpenPCDet/data/cisl/072124/Tower_1/run_1/lidar/`. Inside this folder, you should now see a screenshot that correlates to each of the **.npy** lidar files. If you double click the first image in the set and hold down the `right arrow key`, you will see a sort of animation effect from the quick changing of the frames.


## Moving Forward
The purpose of this program is to speed up the lengthy process of manually labeling the ground truth bounding boxes for objects in point cloud data. Instead of creating each bounding box for each frame of data in every scene you record, you now have a dataset that gives you a bulk of the labels for each frame with higher accuracy than that produced by a single model. The next step would be to feed this label data into a program like LabelCloud so that you can move frame by frame and make adjustments to the bounding boxes. Assuming you have 800 frames of data and it takes you two minutes to make labels for each frame, you spend over 26 hours labeling just that one scene. Having a set of labels with reasonably high accuracy for each frame in the scene allows you to make small adjustments and add/delete labels as neccesary, greatly reducing the number of man hours spent creating ground truth data. 
