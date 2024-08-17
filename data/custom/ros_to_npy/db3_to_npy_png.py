# Title: ROSbag to .npy and .png converter
# Authors: Bo Wu and Mason Audet
# Date: 7/12/24
#
# Description: A small modification of Bo Wu's original script that converted ROSbag files into .png and .pcd
#              files. This program converts .bag files to .npy and .png files. It also modifies the arrays to
#              be of size 4 instead of 3 [x, y, z, i]. Then, the x and y coordinates are translated 180 degrees
#              to match the orientation of Kitti before being saved.
#
# Usage: python rosbag_to_npy_png.py bag_file <path_to_bag_file> output_dir <Path_to_output_folder>


import os
import argparse
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import rosbag2_py
import sensor_msgs_py.point_cloud2 as pc2
#import open3d as o3d

class BagExtractor(Node):
    def __init__(self, bag_file, output_dir):
        super().__init__('bag_extractor')
        self.bag_file = bag_file
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)  # Create the output directory if it doesn't exist
        self.bridge = CvBridge()
        self.storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
        self.converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        self.reader = rosbag2_py.SequentialReader()
        self.reader.open(self.storage_options, self.converter_options)

    def extract(self):
        while self.reader.has_next():
            (topic, data, t) = self.reader.read_next()
            if topic == '/lucid/image_raw':
                img_msg = deserialize_message(data, Image)
                self.save_image(img_msg)
            elif topic == '/ouster/points':
                pc_msg = deserialize_message(data, PointCloud2)
                self.save_npy(pc_msg)
    
    def save_image(self, data):
        try:
            print(f"Received data for image: {data}")
            img_msg = self.bridge.imgmsg_to_cv2(data, "bgr8")
            img_filename = os.path.join(self.output_dir, f"image_{data.header.stamp.sec}_{data.header.stamp.nanosec}.png")
            cv2.imwrite(img_filename, img_msg)
            self.get_logger().info(f"Saved image: {img_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save image: {str(e)}")

    def save_npy(self, data):
        try:
            print(f"Received data for point cloud: {data}")
            points = []
            for point in pc2.read_points(data, field_names=("x", "y", "z"), skip_nans=True):
                points.append([point[0], point[1], point[2]])
            points = np.array(points, dtype=np.float64)  # Use float64 for compatibility with Open3D
            
            # Modify array to make it Kitti compatible
            intensity_col = np.zeros((points.shape[0], 1))
            final_point_array = np.hstack((points, intensity_col))
            
            final_point_array[:, 0] *= -1 # Rotates the x axis 180 degrees to match Kitti
            final_point_array[:, 1] *= -1 # Rotates the y axis 180 degrees to match Kitti
            
            # Save the point cloud data as a .npy file
            print(data.header.stamp.sec, data.header.stamp.nanosec)
            npy_filename = os.path.join(self.output_dir, f"cloud_{data.header.stamp.sec}_{data.header.stamp.nanosec}.npy")
            np.save(npy_filename, final_point_array)
            self.get_logger().info(f"Saved point cloud: {npy_filename}")
        except Exception as e:
            self.get_logger().error(f"Failed to save point cloud: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Extract images and point clouds from a ROS2 bag file.")
    parser.add_argument('bag_file', type=str, help="Path to the ROS2 bag file.")
    parser.add_argument('output_dir', type=str, help="Directory where the output files will be saved.")
    args = parser.parse_args()

    if not rclpy.ok():
        rclpy.init()
    extractor = BagExtractor(args.bag_file, args.output_dir)
    extractor.extract()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
