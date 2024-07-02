import numpy 
import cv2
import os
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

# pyntcloud can't read pcd data
# from pyntcloud import PyntCloud

import open3d as o3d
import argparse

valid_idx_yizhuang02 = [1, 3]
valid_idx_yizhuang06 = [1, 4]
valid_idx_yizhuang08 = [0, 3, 5]
valid_idx_yizhuang09 = [2, 11]
valid_idx_yizhuang10 = [9, 13]
valid_idx_yizhuang13 = [0, 2]
valid_idx_yizhuang16 = [3, 6]

frame_shift = 2


def main(source_root_dir, output_root_dir):
    label_dir = os.path.join(source_root_dir, "cooperative-vehicle-infrastructure/cooperative-vehicle-infrastructure/infrastructure-side")
    image_dir = os.path.join(source_root_dir, "cooperative-vehicle-infrastructure-infrastructure-side-image/cooperative-vehicle-infrastructure-infrastructure-side-image")
    lidar_dir = os.path.join(source_root_dir, "cooperative-vehicle-infrastructure-infrastructure-side-velodyne/cooperative-vehicle-infrastructure-infrastructure-side-velodyne")
    
    with open(os.path.join(label_dir, "data_info.json")) as f:
        data_info = json.load(f)
        
    seqs= {}
    for info in data_info:
        intersection_loc = info["intersection_loc"]
        batch_start_idx = info["batch_start_id"]
        batch_end_idx = info["batch_end_id"]
        
        # ex) sub_dir_name: "yizhuang09_019816_019965"
        sub_dir_name = f"{intersection_loc}_{batch_start_idx}_{batch_end_idx}"
        
        if intersection_loc not in seqs:
            seqs[intersection_loc] = []
            
        if sub_dir_name not in seqs[intersection_loc]:
            seqs[intersection_loc].append(sub_dir_name)

    stamps_yizhuang02_lidar = {}
    stamps_yizhuang06_lidar = {}
    stamps_yizhuang08_lidar = {}
    stamps_yizhuang09_lidar = {}
    stamps_yizhuang10_lidar = {}
    stamps_yizhuang13_lidar = {}
    stamps_yizhuang16_lidar = {}
    stamps_yizhuang02_image = {}
    stamps_yizhuang06_image = {}
    stamps_yizhuang08_image = {}
    stamps_yizhuang09_image = {}
    stamps_yizhuang10_image = {}
    stamps_yizhuang13_image = {}
    stamps_yizhuang16_image = {}
    
    for data in data_info:
        if data["intersection_loc"] == "yizhuang02":
            stamps_yizhuang02_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang02_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        elif data["intersection_loc"] == "yizhuang06":
            stamps_yizhuang06_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang06_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        elif data["intersection_loc"] == "yizhuang08":
            stamps_yizhuang08_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang08_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        elif data["intersection_loc"] == "yizhuang09":
            stamps_yizhuang09_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang09_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        elif data["intersection_loc"] == "yizhuang10":
            stamps_yizhuang10_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang10_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        elif data["intersection_loc"] == "yizhuang13":
            stamps_yizhuang13_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang13_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        elif data["intersection_loc"] == "yizhuang16":
            stamps_yizhuang16_lidar[data["pointcloud_timestamp"]] = data["pointcloud_path"].replace("velodyne/", "")
            stamps_yizhuang16_image[data["image_timestamp"]] = data["image_path"].replace("image/", "")
        else:
            print("Invalid intersection_loc")
    
    yizhuang02_pair = []
    for img_stamp in stamps_yizhuang02_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang02_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang02_pair.append({'image': stamps_yizhuang02_image[img_stamp], 'lidar': stamps_yizhuang02_lidar[lidar_stamp]})
    print('yizhuang02_pair Matched!')
        
    yizhuang06_pair = []
    for img_stamp in stamps_yizhuang06_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang06_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang06_pair.append({'image': stamps_yizhuang06_image[img_stamp], 'lidar': stamps_yizhuang06_lidar[lidar_stamp]})
    print('yizhuang06_pair Matched!')
        
    yizhuang08_pair = []
    for img_stamp in stamps_yizhuang08_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang08_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang08_pair.append({'image': stamps_yizhuang08_image[img_stamp], 'lidar': stamps_yizhuang08_lidar[lidar_stamp]})
    print('yizhuang08_pair Matched!')
    
    yizhuang09_pair = []
    for img_stamp in stamps_yizhuang09_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang09_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang09_pair.append({'image': stamps_yizhuang09_image[img_stamp], 'lidar': stamps_yizhuang09_lidar[lidar_stamp]})
    print('yizhuang09_pair Matched!')
    
    yizhuang10_pair = []
    for img_stamp in stamps_yizhuang10_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang10_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang10_pair.append({'image': stamps_yizhuang10_image[img_stamp], 'lidar': stamps_yizhuang10_lidar[lidar_stamp]})
    print('yizhuang10_pair Matched!')
    
    yizhuang13_pair = []
    for img_stamp in stamps_yizhuang13_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang13_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang13_pair.append({'image': stamps_yizhuang13_image[img_stamp], 'lidar': stamps_yizhuang13_lidar[lidar_stamp]})
    print('yizhuang13_pair Matched!')
    
    yizhuang16_pair = []
    for img_stamp in stamps_yizhuang16_image.keys():
        # find the closest lidar stamp
        lidar_stamp = min(stamps_yizhuang16_lidar.keys(), key=lambda x:abs(int(x)-int(img_stamp)))
        yizhuang16_pair.append({'image': stamps_yizhuang16_image[img_stamp], 'lidar': stamps_yizhuang16_lidar[lidar_stamp]})
    print('yizhuang16_pair Matched!')
        
    
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directories.")
    parser.add_argument('--source_root_dir', type=str, required=False, help='Path to the source root directory', default="D:/DAIR-V2X-C/Full Dataset (train&val)")
    parser.add_argument('--output_root_dir', type=str, required=False, help='Path to the output root directory', default="D:/DAIR-V2X-C-Infra")

    args = parser.parse_args()
    
    main(args.source_root_dir, args.output_root_dir)