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
    
    seqs_train = []
    seqs_val = []
    for key, value in seqs.items():
        train_idx = list(range(len(value)))
        if "yizhuang02" == key:
            valid_idx = valid_idx_yizhuang02
            for idx in valid_idx:
                train_idx.remove(idx)
        elif "yizhuang06" == key:
            valid_idx = valid_idx_yizhuang06
            for idx in valid_idx:
                train_idx.remove(idx)
        elif "yizhuang08" == key:
            valid_idx = valid_idx_yizhuang08
            for idx in valid_idx:
                train_idx.remove(idx)
        elif "yizhuang09" == key:
            valid_idx = valid_idx_yizhuang09
            for idx in valid_idx:
                train_idx.remove(idx)
        elif "yizhuang10" == key:
            valid_idx = valid_idx_yizhuang10
            for idx in valid_idx:
                train_idx.remove(idx)
        elif "yizhuang13" == key:
            valid_idx = valid_idx_yizhuang13
            for idx in valid_idx:
                train_idx.remove(idx)
        elif "yizhuang16" == key:
            valid_idx = valid_idx_yizhuang16
            for idx in valid_idx:
                train_idx.remove(idx)

                
        for idx in train_idx:
            seqs_train.append(value[idx])
        for idx in valid_idx:
            seqs_val.append(value[idx])
          
    if not os.path.exists(os.path.join(output_root_dir, "Train")):
        os.makedirs(os.path.join(output_root_dir, "Train"))
    else:
        # Remove all files in the directory and make the directory empty.
        os.system(f"rm -rf {os.path.join(output_root_dir, 'Train')}/*")
        
    if not os.path.exists(os.path.join(output_root_dir, "Valid")):
        os.makedirs(os.path.join(output_root_dir, "Valid"))
    else:
        os.system(f"rm -rf {os.path.join(output_root_dir, 'Valid')}/*")
    
    if not os.path.exists(os.path.join(output_root_dir, "Train", "Camera")):
        os.makedirs(os.path.join(output_root_dir, "Train", "Camera"))
    if not os.path.exists(os.path.join(output_root_dir, "Train", "Camera", "Image")):
        os.makedirs(os.path.join(output_root_dir, "Train", "Camera", "Image"))
    if not os.path.exists(os.path.join(output_root_dir, "Train", "Camera", "Label")):
        os.makedirs(os.path.join(output_root_dir, "Train", "Camera", "Label"))
    if not os.path.exists(os.path.join(output_root_dir, "Train", "LiDAR")):
        os.makedirs(os.path.join(output_root_dir, "Train", "LiDAR"))
    if not os.path.exists(os.path.join(output_root_dir, "Train", "LiDAR", "LiDAR")):
        os.makedirs(os.path.join(output_root_dir, "Train", "LiDAR", "LiDAR"))
    if not os.path.exists(os.path.join(output_root_dir, "Train", "LiDAR", "Label")):
        os.makedirs(os.path.join(output_root_dir, "Train", "LiDAR", "Label"))
    if not os.path.exists(os.path.join(output_root_dir, "Valid", "Camera")):
        os.makedirs(os.path.join(output_root_dir, "Valid", "Camera"))
    if not os.path.exists(os.path.join(output_root_dir, "Valid", "Camera", "Image")):
        os.makedirs(os.path.join(output_root_dir, "Valid", "Camera", "Image"))
    if not os.path.exists(os.path.join(output_root_dir, "Valid", "Camera", "Label")):
        os.makedirs(os.path.join(output_root_dir, "Valid", "Camera", "Label"))
    if not os.path.exists(os.path.join(output_root_dir, "Valid", "LiDAR")):
        os.makedirs(os.path.join(output_root_dir, "Valid", "LiDAR"))
    if not os.path.exists(os.path.join(output_root_dir, "Valid", "LiDAR", "LiDAR")):
        os.makedirs(os.path.join(output_root_dir, "Valid", "LiDAR", "LiDAR"))
    if not os.path.exists(os.path.join(output_root_dir, "Valid", "LiDAR", "Label")):
        os.makedirs(os.path.join(output_root_dir, "Valid", "LiDAR", "Label"))
    
    for info in data_info:
        image_path = os.path.join(image_dir, info["image_path"].split("/")[-1])
        pointcloud_path = os.path.join(lidar_dir, info["pointcloud_path"].split("/")[-1])
        
        intrinsic_path = os.path.join(label_dir, info["calib_camera_intrinsic_path"])
        extrinsic_path = os.path.join(label_dir, info["calib_virtuallidar_to_camera_path"])
        lidar_label_path = os.path.join(label_dir, info["label_lidar_std_path"])
        camera_label_path = os.path.join(label_dir, info["label_camera_std_path"])
        
        intersection_loc = info["intersection_loc"]
        batch_start_idx = info["batch_start_id"]
        batch_end_idx = info["batch_end_id"]
        
        # ex) sub_dir_name: "yizhuang09_019816_019965"
        sub_dir_name = f"{intersection_loc}_{batch_start_idx}_{batch_end_idx}"

        with open(intrinsic_path) as f:
            intrinsic = json.load(f)
            
        with open(extrinsic_path) as f:
            extrinsic = json.load(f)
            
        with open(lidar_label_path) as f:
            lidar_label = json.load(f)
            
        with open(camera_label_path) as f:
            camera_label = json.load(f)
            
        if sub_dir_name in seqs_train:
            output_path = os.path.join(output_root_dir, "Train")
        elif sub_dir_name in seqs_val:
            output_path = os.path.join(output_root_dir, "Valid")
            
        parsing(image_path, pointcloud_path, intrinsic, extrinsic, lidar_label, camera_label, output_path, sub_dir_name)
        
        break

def parsing(image_path, pointcloud_path, intrinsic, extrinsic, lidar_label, camera_label, output_path, sub_dir_name):
    if not os.path.exists(os.path.join(output_path, "Camera/Image", sub_dir_name)):
        os.makedirs(os.path.join(output_path, "Camera/Image", sub_dir_name))
    if not os.path.exists(os.path.join(output_path, "Camera/Label", sub_dir_name)):
        os.makedirs(os.path.join(output_path, "Camera/Label", sub_dir_name))
    if not os.path.exists(os.path.join(output_path, "LiDAR/LiDAR", sub_dir_name)):
        os.makedirs(os.path.join(output_path, "LiDAR/LiDAR", sub_dir_name))
    if not os.path.exists(os.path.join(output_path, "LiDAR/Label", sub_dir_name)):
        os.makedirs(os.path.join(output_path, "LiDAR/Label", sub_dir_name))
        
    # Save image
    image = cv2.imread(image_path)
    image_name = image_path[-10:]
    cv2.imwrite(os.path.join(output_path, 'Camera/Image', sub_dir_name, image_name), image)
    
    # image shape
    imh, imw, _ = image.shape
    # data name
    data_name = image_path[-10:-4]
    
    # Save pointcloud as .bin
    pointcloud = o3d.io.read_point_cloud(pointcloud_path)
    pointcloud = np.asarray(pointcloud.points)
    pointcloud = pointcloud.astype(np.float32)
    pointcloud.tofile(os.path.join(output_path, "LiDAR/LiDAR", sub_dir_name, f"{pointcloud_path[-10:-4]}.bin"))
    
    # camera label json dict
    camera_label_dict = {}
    camera_label_dict['intrinsic'] = {}
    camera_label_dict['intrinsic']['fx'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[0, 0]
    camera_label_dict['intrinsic']['fy'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[1, 1]
    camera_label_dict['intrinsic']['cx'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[0, 2]
    camera_label_dict['intrinsic']['cy'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[1, 2]
    
    ext_rotation = np.array(extrinsic["rotation"]).reshape(3, 3)
    ext_translation = np.array(extrinsic["translation"])
    
    # rotation to quaternion
    r = R.from_matrix(ext_rotation)
    ext_quaternion = r.as_quat()
    qx, qy, qz, qw = ext_quaternion
    
    camera_label_dict['extrinsic'] = {}
    camera_label_dict['extrinsic']['rotation'] = {}
    camera_label_dict['extrinsic']['rotation']['x'] = qx
    camera_label_dict['extrinsic']['rotation']['y'] = qy
    camera_label_dict['extrinsic']['rotation']['z'] = qz
    camera_label_dict['extrinsic']['rotation']['w'] = qw
    camera_label_dict['extrinsic']['translation'] = {}
    camera_label_dict['extrinsic']['translation']['x'] = float(ext_translation[0, 0])
    camera_label_dict['extrinsic']['translation']['y'] = float(ext_translation[1, 0])
    camera_label_dict['extrinsic']['translation']['z'] = float(ext_translation[2, 0])
    
    camera_label_dict['objects'] = []
    for obj in camera_label:
        obj_dict = {}
        obj_dict['class'] = obj['type']
        
        if '2d_box' in obj:
            obj_dict['box2d'] = {}
            xmin, ymin, xmax, ymax = obj['2d_box']['xmin'], obj['2d_box']['ymin'], obj['2d_box']['xmax'], obj['2d_box']['ymax']
            xmin = xmin / imw
            xmax = xmax / imw
            ymin = ymin / imh
            ymax = ymax / imh
            
            cx = (xmin + xmax) / 2
            cy = (ymin + ymax) / 2
            w = xmax - xmin
            h = ymax - ymin
            
            obj_dict['box2d']['cx'] = cx
            obj_dict['box2d']['cy'] = cy
            obj_dict['box2d']['w'] = w
            obj_dict['box2d']['h'] = h
            
        if '3d_dimensions' in obj and '3d_location' in obj:
            obj_dict['box3d'] = {}
            obj_dict['box3d']['size'] = {}
            obj_dict['box3d']['size']['width'] = obj['3d_dimensions']['w']
            obj_dict['box3d']['size']['length'] = obj['3d_dimensions']['l']
            obj_dict['box3d']['size']['height'] = obj['3d_dimensions']['h']
            
            obj_dict['box3d']['translation'] = {}
            obj_dict['box3d']['translation']['x'] = obj['3d_location']['x']
            
            rot_y = obj['rotation']
            # rotation to quaternion
            r = R.from_euler('z', rot_y, degrees=False)
            obj_quaternion = r.as_quat()
            obj_qx, obj_qy, obj_qz, obj_qw = obj_quaternion
            
            obj_dict['box3d']['rotation'] = {}
            obj_dict['box3d']['rotation']['x'] = obj_qx
            obj_dict['box3d']['rotation']['y'] = obj_qy
            obj_dict['box3d']['rotation']['z'] = obj_qz
            obj_dict['box3d']['rotation']['w'] = obj_qw
            
            obj_dict['box3d']['translation']['x'] = obj['3d_location']['x']
            obj_dict['box3d']['translation']['y'] = obj['3d_location']['y']
            obj_dict['box3d']['translation']['z'] = obj['3d_location']['z']
        
        camera_label_dict['objects'].append(obj_dict)
    
    # Save camera label as .json
    with open(os.path.join(output_path, "Camera/Label", sub_dir_name, f"{data_name}.json"), 'w') as f:
        json.dump(camera_label_dict, f, indent=4)
        
    pass
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directories.")
    parser.add_argument('--source_root_dir', type=str, required=False, help='Path to the source root directory')
    parser.add_argument('--output_root_dir', type=str, required=False, help='Path to the output root directory')

    args = parser.parse_args()
    
    # main(args.source_root_dir, args.output_root_dir)
    
    main("D:/DAIR-V2X-C/Full Dataset (train&val)", "D:/DAIR-V2X-C-Infra")