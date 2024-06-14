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
    
    cnt = 0
    for info in data_info:
        image_path = os.path.join(image_dir, info["image_path"].split("/")[-1])
        pointcloud_path = os.path.join(lidar_dir, info["pointcloud_path"].split("/")[-1])
        pointcloud_idx = f"{(int(pointcloud_path[-10:-4]) - frame_shift):06d}"
        pointcloud_path = f"{pointcloud_path[:-10]}{pointcloud_idx}.pcd"
                
        intrinsic_path = os.path.join(label_dir, info["calib_camera_intrinsic_path"])
        extrinsic_path = os.path.join(label_dir, info["calib_virtuallidar_to_camera_path"])
        lidar_label_path = os.path.join(label_dir, info["label_lidar_std_path"])
        lidar_idx = f"{(int(lidar_label_path[-11:-5]) - frame_shift):06d}"
        lidar_label_path = f"{lidar_label_path[:-11]}{lidar_idx}.json"
        camera_label_path = os.path.join(label_dir, info["label_camera_std_path"])
        
        if not os.path.exists(image_path) or not os.path.exists(pointcloud_path) or not os.path.exists(intrinsic_path) or not os.path.exists(extrinsic_path) or not os.path.exists(lidar_label_path) or not os.path.exists(camera_label_path):
            continue
        
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
        
        # if cnt > 5:
        #     break
        cnt += 1
        print(f"Processing {cnt}th data of {len(data_info)}: {sub_dir_name}")

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
    pointcloud_shifted = f"{(int(pointcloud_path[-10:-4]) + frame_shift):06d}"
    pointcloud.tofile(os.path.join(output_path, "LiDAR/LiDAR", sub_dir_name, f"{pointcloud_shifted}.bin"))
    
    # camera label json dict
    camera_label_dict = {}
    camera_label_dict['intrinsic'] = {}
    camera_label_dict['intrinsic']['fx'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[0, 0]
    camera_label_dict['intrinsic']['fy'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[1, 1]
    camera_label_dict['intrinsic']['cx'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[0, 2]
    camera_label_dict['intrinsic']['cy'] = np.array(intrinsic["cam_K"]).reshape(3, 3)[1, 2]
    
    ext_rotation = np.array(extrinsic["rotation"], dtype=np.float32).reshape(3, 3)
    ext_translation = np.array(extrinsic["translation"], dtype=np.float32).reshape(3, 1)
    ext_L2C = np.hstack([ext_rotation, ext_translation.reshape(3, 1)])
    ext_L2C = np.vstack([ext_L2C, np.array([0, 0, 0, 1])])
    
    camera_label_dict['extrinsic'] = {}
    camera_label_dict['extrinsic']['rotation'] = ext_rotation.reshape(-1).tolist()
    camera_label_dict['extrinsic']['translation'] = {}
    camera_label_dict['extrinsic']['translation']['x'] = float(ext_translation[0, 0])
    camera_label_dict['extrinsic']['translation']['y'] = float(ext_translation[1, 0])
    camera_label_dict['extrinsic']['translation']['z'] = float(ext_translation[2, 0])
    
    # parsing Camera Object Label
    camera_label_dict['objects'] = []
    for obj in lidar_label:
        obj_dict = {}
        obj_dict['class'] = obj['type']
        
        # LiDAR 3D Label is shifted by "frame_shift" frames
        # So, 2D label is "frame_shift" frames ahead of 3D label
        # in this time, we can't find corresponding 2D label for 3D label
        
        # if '2d_box' in obj:
        #     obj_dict['box2d'] = {}
        #     xmin, ymin, xmax, ymax = obj['2d_box']['xmin'], obj['2d_box']['ymin'], obj['2d_box']['xmax'], obj['2d_box']['ymax']
        #     xmin = xmin / imw
        #     xmax = xmax / imw
        #     ymin = ymin / imh
        #     ymax = ymax / imh
            
        #     cx = (xmin + xmax) / 2
        #     cy = (ymin + ymax) / 2
        #     w = xmax - xmin
        #     h = ymax - ymin
            
        #     obj_dict['box2d']['cx'] = cx
        #     obj_dict['box2d']['cy'] = cy
        #     obj_dict['box2d']['w'] = w
        #     obj_dict['box2d']['h'] = h
            
        if '3d_dimensions' in obj and '3d_location' in obj:
            obj_dict['box3d'] = {}
            obj_dict['box3d']['size'] = {}
            obj_dict['box3d']['size']['width'] = obj['3d_dimensions']['w']
            obj_dict['box3d']['size']['length'] = obj['3d_dimensions']['l']
            obj_dict['box3d']['size']['height'] = obj['3d_dimensions']['h']
            
            rot_3d = obj['rotation']
            # rotation to quaternion
            obj_rotation = np.array([
                [np.cos(rot_3d), -np.sin(rot_3d), 0],
                [np.sin(rot_3d), np.cos(rot_3d), 0],
                [0, 0, 1]
            ])
            
            obj_tx, obj_ty, obj_tz = obj['3d_location']['x'], obj['3d_location']['y'], obj['3d_location']['z']
            
            ext_obj = np.hstack([obj_rotation, np.array([[obj_tx], [obj_ty], [obj_tz]])])
            ext_obj = np.vstack([ext_obj, np.array([0, 0, 0, 1])])
            
            ext = ext_L2C @ ext_obj
            
            tx = ext[0, 3]
            ty = ext[1, 3]
            tz = ext[2, 3]
            
            obj_dict['box3d']['rotation'] = ext[:3, :3].reshape(-1).tolist()
            
            obj_dict['box3d']['translation'] = {}
            obj_dict['box3d']['translation']['x'] = tx
            obj_dict['box3d']['translation']['y'] = ty
            obj_dict['box3d']['translation']['z'] = tz
        
        camera_label_dict['objects'].append(obj_dict)
        
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
        
        camera_label_dict['objects'].append(obj_dict)
    
    # Save camera label as .json
    with open(os.path.join(output_path, "Camera/Label", sub_dir_name, f"{data_name}.json"), 'w') as f:
        json.dump(camera_label_dict, f, indent=4)
    
    # LiDAR
    lidar_label_dict = {}
    lidar_label_dict['columns'] = ['x', 'y', 'z']
    lidar_label_dict['extrinsic'] = {}
    lidar_label_dict['extrinsic']['rotation'] = np.zeros(3).tolist()
    lidar_label_dict['extrinsic']['translation'] = np.zeros(3).tolist()
    
    # Parsing LiDAR Object Label
    lidar_label_dict['objects'] = []
    for obj in lidar_label:
        obj_dict = {}
        obj_dict['class'] = obj['type']
            
        if '3d_dimensions' in obj and '3d_location' in obj:
            obj_dict['box3d'] = {}
            obj_dict['box3d']['size'] = {}
            obj_dict['box3d']['size']['width'] = obj['3d_dimensions']['w']
            obj_dict['box3d']['size']['length'] = obj['3d_dimensions']['l']
            obj_dict['box3d']['size']['height'] = obj['3d_dimensions']['h']
            
            rot_3d = obj['rotation']
            # rotation to quaternion
            obj_rotation = np.array([
                [np.cos(rot_3d), -np.sin(rot_3d), 0],
                [np.sin(rot_3d), np.cos(rot_3d), 0],
                [0, 0, 1]
            ])
            
            obj_tx, obj_ty, obj_tz = obj['3d_location']['x'], obj['3d_location']['y'], obj['3d_location']['z']
            
            ext_obj = np.hstack([obj_rotation, np.array([[obj_tx], [obj_ty], [obj_tz]])])
            ext_obj = np.vstack([ext_obj, np.array([0, 0, 0, 1])])
            
            tx = ext_obj[0, 3]
            ty = ext_obj[1, 3]
            tz = ext_obj[2, 3]
            
            obj_dict['box3d']['rotation'] = ext_obj[:3, :3].reshape(-1).tolist()
            
            obj_dict['box3d']['translation'] = {}
            obj_dict['box3d']['translation']['x'] = tx
            obj_dict['box3d']['translation']['y'] = ty
            obj_dict['box3d']['translation']['z'] = tz
        
        lidar_label_dict['objects'].append(obj_dict)

    # Save LiDAR label as .json
    with open(os.path.join(output_path, "LiDAR/Label", sub_dir_name, f"{pointcloud_shifted}.json"), 'w') as f:
        json.dump(lidar_label_dict, f, indent=4)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process directories.")
    parser.add_argument('--source_root_dir', type=str, required=False, help='Path to the source root directory')
    parser.add_argument('--output_root_dir', type=str, required=False, help='Path to the output root directory')

    args = parser.parse_args()
    
    # main(args.source_root_dir, args.output_root_dir)
    
    main("D:/DAIR-V2X-C/Full Dataset (train&val)", "D:/DAIR-V2X-C-Infra")