# Authors: Professor Hang Qiu and Joong Ho Kim
# Date: 7/25/24
#
# Description: Needed to have the input data in pcd format instead of db3, but the machines we have now do not have the required environment  
#          	   to run the conversion script (version conflict, etc), so we built docker file to run the ros2 scripts.
#          	   Then the db3 files will be visible inside docker, we downloaded and ran the conversion script inside docker
#          	   and store the raw data in the same directory, which will be visible from outside the docker.
#         	 
# Usage: docker build -t cisl/ros2:humble --file ros2_humble_jammy.Dockerfile .
#    	 docker run -it --rm -v "<Local path to db3 files>:<cisl work directory>" cisl/ros2:humble

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











