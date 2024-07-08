import os
import cv2
import numpy as np
import json
import argparse

import xml.etree.ElementTree as ET
from tqdm import tqdm

import shutil

def making_dirs(target_dir):
    if not os.path.exists(f"{target_dir}/Train"):
        os.makedirs(f"{target_dir}/Train")
    if not os.path.exists(f"{target_dir}/Train/StereoLeft"):
        os.makedirs(f"{target_dir}/Train/StereoLeft")
    if not os.path.exists(f"{target_dir}/Train/StereoLeft/Image"):
        os.makedirs(f"{target_dir}/Train/StereoLeft/Image")
    if not os.path.exists(f"{target_dir}/Train/StereoLeft/Label"):
        os.makedirs(f"{target_dir}/Train/StereoLeft/Label")
        
    if not os.path.exists(f"{target_dir}/Train/StereoRight"):
        os.makedirs(f"{target_dir}/Train/StereoRight")
    if not os.path.exists(f"{target_dir}/Train/StereoRight/Image"):
        os.makedirs(f"{target_dir}/Train/StereoRight/Image")
    if not os.path.exists(f"{target_dir}/Train/StereoRight/Label"):
        os.makedirs(f"{target_dir}/Train/StereoRight/Label")
        
    if not os.path.exists(f"{target_dir}/Train/LiDAR"):
        os.makedirs(f"{target_dir}/Train/LiDAR")
    if not os.path.exists(f"{target_dir}/Train/LiDAR/LiDAR"):
        os.makedirs(f"{target_dir}/Train/LiDAR/LiDAR")
    if not os.path.exists(f"{target_dir}/Train/LiDAR/Label"):
        os.makedirs(f"{target_dir}/Train/LiDAR/Label")
        
    if not os.path.exists(f"{target_dir}/Valid"):
        os.makedirs(f"{target_dir}/Valid")
    if not os.path.exists(f"{target_dir}/Valid/StereoLeft"):
        os.makedirs(f"{target_dir}/Valid/StereoLeft")
    if not os.path.exists(f"{target_dir}/Valid/StereoLeft/Image"):
        os.makedirs(f"{target_dir}/Valid/StereoLeft/Image")
    if not os.path.exists(f"{target_dir}/Valid/StereoLeft/Label"):
        os.makedirs(f"{target_dir}/Valid/StereoLeft/Label")
        
    if not os.path.exists(f"{target_dir}/Valid/StereoRight"):
        os.makedirs(f"{target_dir}/Valid/StereoRight")
    if not os.path.exists(f"{target_dir}/Valid/StereoRight/Image"):
        os.makedirs(f"{target_dir}/Valid/StereoRight/Image")
    if not os.path.exists(f"{target_dir}/Valid/StereoRight/Label"):
        os.makedirs(f"{target_dir}/Valid/StereoRight/Label")
        
    if not os.path.exists(f"{target_dir}/Valid/LiDAR"):
        os.makedirs(f"{target_dir}/Valid/LiDAR")
    if not os.path.exists(f"{target_dir}/Valid/LiDAR/LiDAR"):
        os.makedirs(f"{target_dir}/Valid/LiDAR/LiDAR")
    if not os.path.exists(f"{target_dir}/Valid/LiDAR/Label"):
        os.makedirs(f"{target_dir}/Valid/LiDAR/Label")

def making_sub_dirs(target_dir, purpose, sub_dir):
    if not os.path.exists(f"{target_dir}/{purpose}/StereoLeft/Image/{sub_dir}"):
        os.makedirs(f"{target_dir}/{purpose}/StereoLeft/Image/{sub_dir}")
    if not os.path.exists(f"{target_dir}/{purpose}/StereoLeft/Label/{sub_dir}"):
        os.makedirs(f"{target_dir}/{purpose}/StereoLeft/Label/{sub_dir}")
        
    if not os.path.exists(f"{target_dir}/{purpose}/StereoRight/Image/{sub_dir}"):
        os.makedirs(f"{target_dir}/{purpose}/StereoRight/Image/{sub_dir}")
    if not os.path.exists(f"{target_dir}/{purpose}/StereoRight/Label/{sub_dir}"):
        os.makedirs(f"{target_dir}/{purpose}/StereoRight/Label/{sub_dir}")
        
    if not os.path.exists(f"{target_dir}/{purpose}/LiDAR/LiDAR/{sub_dir}"):
        os.makedirs(f"{target_dir}/{purpose}/LiDAR/LiDAR/{sub_dir}")
    if not os.path.exists(f"{target_dir}/{purpose}/LiDAR/Label/{sub_dir}"):
        os.makedirs(f"{target_dir}/{purpose}/LiDAR/Label/{sub_dir}")

def convert(source_dir, target_dir):
    calib_dir = f"{source_dir}/calib"
    image_2_dir = f"{source_dir}/image_2"
    image_3_dir = f"{source_dir}/image_3"
    velodyne_dir = f"{source_dir}/velodyne"
    label_dir = f"{source_dir}/label_2"
    
    making_dirs(target_dir)
    
    # 0~5999: Train
    # 6000~7480: Valid
    
    for idx in tqdm(range(7481)):
        sub_dir = idx // 1000
        
        if idx < 6000:
            making_sub_dirs(target_dir, 'Train', sub_dir)
        else:
            making_sub_dirs(target_dir, 'Valid', sub_dir)
        
        calib_file = f"{calib_dir}/{str(idx).zfill(6)}.txt"
        image_2_file = f"{image_2_dir}/{str(idx).zfill(6)}.png"
        image_3_file = f"{image_3_dir}/{str(idx).zfill(6)}.png"
        velodyne_file = f"{velodyne_dir}/{str(idx).zfill(6)}.bin"
        label_file = f"{label_dir}/{str(idx).zfill(6)}.txt"
        
        if not os.path.exists(calib_file) or not os.path.exists(image_2_file) or not os.path.exists(image_3_file) or not os.path.exists(velodyne_file) or not os.path.exists(label_file):
            print(f"File not found: {idx}")
            continue
        
        with open(calib_file, 'r') as f:
            calib = f.readlines()
            P2 = np.array(calib[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(calib[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            Tr_velo_to_cam = np.array(calib[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            
        P2 = np.concatenate([P2, np.array([[0, 0, 0, 0]])], axis=0)
        out = cv2.decomposeProjectionMatrix(P2[:3, :])
        K2 = out[0]
        R2 = out[1]
        t2 = -(out[2][:3] / out[2][3])
        
        P3 = np.concatenate([P3, np.array([[0, 0, 0, 0]])], axis=0)
        out = cv2.decomposeProjectionMatrix(P3[:3, :])
        K3 = out[0]
        R3 = out[1]
        t3 = -(out[2][:3] / out[2][3])
        
        extrinsic_02 = np.concatenate([R2, t2.reshape(3, 1)], axis=1)
        extrinsic_03 = np.concatenate([R3, t3.reshape(3, 1)], axis=1)
        extrinsic_02 = np.concatenate([extrinsic_02, np.array([[0, 0, 0, 1]])], axis=0)
        extrinsic_03 = np.concatenate([extrinsic_03, np.array([[0, 0, 0, 1]])], axis=0)
        
        extrinsic_03 = extrinsic_03 @ np.linalg.inv(extrinsic_02)
        R3 = extrinsic_03[:3, :3]
        t3 = extrinsic_03[:3, 3]
        
        image_2 = cv2.imread(image_2_file)
        image_3 = cv2.imread(image_3_file)
        
        if image_2 is None:
            print(f"Image not found: {image_2_file}")
        if image_3 is None:
            print(f"Image not found: {image_3_file}")
            
        label_2 = dict()
        label_3 = dict()
        label_lidar = dict()
        
        obj_2 = []
        obj_3 = []
        obj_lidar = []
        
        # Camera Intrinsics
        fx_2 = K2[0, 0]
        fy_2 = K2[1, 1]
        cx_2 = K2[0, 2]
        cy_2 = K2[1, 2]
        
        fx_3 = K3[0, 0]
        fy_3 = K3[1, 1]
        cx_3 = K3[0, 2]
        cy_3 = K3[1, 2]
        
        label_2['intrinsic'] = {}
        label_2['intrinsic']['fx'] = float(fx_2)
        label_2['intrinsic']['fy'] = float(fy_2)
        label_2['intrinsic']['cx'] = float(cx_2)
        label_2['intrinsic']['cy'] = float(cy_2)
        
        label_3['intrinsic'] = {}
        label_3['intrinsic']['fx'] = float(fx_3)
        label_3['intrinsic']['fy'] = float(fy_3)
        label_3['intrinsic']['cx'] = float(cx_3)
        label_3['intrinsic']['cy'] = float(cy_3)
        
        # camera Extrinsics
        label_2['extrinsic'] = {}
        label_2['extrinsic']['rotation'] = [float(x) for x in R2.flatten()]
        label_2['extrinsic']['translation'] = {}
        label_2['extrinsic']['translation']['x'] = float(t2[0])
        label_2['extrinsic']['translation']['y'] = float(t2[1])
        label_2['extrinsic']['translation']['z'] = float(t2[2])
        
        label_3['extrinsic'] = {}
        label_3['extrinsic']['rotation'] = [float(x) for x in R3.flatten()]
        label_3['extrinsic']['translation'] = {}
        label_3['extrinsic']['translation']['x'] = float(t3[0])
        label_3['extrinsic']['translation']['y'] = float(t3[1])
        label_3['extrinsic']['translation']['z'] = float(t3[2])
        
        # LiDAR
        label_lidar['columns'] = ['x', 'y', 'z', 'intensity']
        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        lidar_extrinsic = np.linalg.inv(Tr_velo_to_cam)
        label_lidar['extrinsic'] = {}
        label_lidar['extrinsic']['rotation'] = [float(x) for x in lidar_extrinsic[:3, :3].flatten()]
        label_lidar['extrinsic']['translation'] = {}
        label_lidar['extrinsic']['translation']['x'] = float(lidar_extrinsic[0, 3])
        label_lidar['extrinsic']['translation']['y'] = float(lidar_extrinsic[1, 3])
        label_lidar['extrinsic']['translation']['z'] = float(lidar_extrinsic[2, 3])
        
        with open(label_file, 'r') as f:
            labels = f.readlines()
            labels = [label.strip().split(' ') for label in labels]
            
            for label in labels:
                one_obj_2 = {}
                one_obj_3 = {}
                one_obj_lidar = {}
                objectClass = label[0]
                if objectClass == 'DontCare':
                    continue
                
                bbox2d_x1 = float(label[4])
                bbox2d_y1 = float(label[5])
                bbox2d_x2 = float(label[6])
                bbox2d_y2 = float(label[7])
                
                bbox3d_h = float(label[8])
                bbox3d_w = float(label[9])
                bbox3d_l = float(label[10])
                bbox3d_x = float(label[11])
                bbox3d_y = float(label[12])
                bbox3d_z = float(label[13])
                bbox3d_rot_y = float(label[14])
                
                rot_y = np.array([
                    [np.cos(bbox3d_rot_y), 0, np.sin(bbox3d_rot_y)],
                    [0, 1, 0],
                    [-np.sin(bbox3d_rot_y), 0, np.cos(bbox3d_rot_y)]
                ])
                
                yz_change = np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])
                
                rot_y = rot_y @ yz_change
                
                R_obj = rot_y
                t_obj = np.array([bbox3d_x, bbox3d_y - (bbox3d_h / 2), bbox3d_z])
                
                # Left Label
                one_obj_2['class'] = objectClass
                one_obj_2['box2d'] = {}
                one_obj_2['box2d']['cx'] = (bbox2d_x1 + bbox2d_x2) / 2
                one_obj_2['box2d']['cy'] = (bbox2d_y1 + bbox2d_y2) / 2
                one_obj_2['box2d']['w'] = bbox2d_x2 - bbox2d_x1
                one_obj_2['box2d']['h'] = bbox2d_y2 - bbox2d_y1
                
                one_obj_2['box3d'] = {}
                one_obj_2['box3d']['size'] = {}
                one_obj_2['box3d']['size']['height'] = bbox3d_h
                one_obj_2['box3d']['size']['width'] = bbox3d_w
                one_obj_2['box3d']['size']['length'] = bbox3d_l
                one_obj_2['box3d']['rotation'] = [float(x) for x in rot_y.flatten()]
                one_obj_2['box3d']['translation'] = {}
                one_obj_2['box3d']['translation']['x'] = bbox3d_x
                one_obj_2['box3d']['translation']['y'] = bbox3d_y - bbox3d_h / 2
                one_obj_2['box3d']['translation']['z'] = bbox3d_z
                
                obj_2.append(one_obj_2)
                
                # Right Label
                obj_extrinsic = np.concatenate([R_obj, t_obj.reshape(3, 1)], axis=1)
                obj_extrinsic = np.concatenate([obj_extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
                obj_extrinsic = extrinsic_03 @ np.linalg.inv(extrinsic_02) @ obj_extrinsic
                
                one_obj_3['class'] = objectClass
                one_obj_3['box3d'] = {}
                one_obj_3['box3d']['size'] = {}
                one_obj_3['box3d']['size']['height'] = bbox3d_h
                one_obj_3['box3d']['size']['width'] = bbox3d_w
                one_obj_3['box3d']['size']['length'] = bbox3d_l
                one_obj_3['box3d']['rotation'] = [float(x) for x in obj_extrinsic[:3, :3].flatten()]
                one_obj_3['box3d']['translation'] = {}
                one_obj_3['box3d']['translation']['x'] = float(obj_extrinsic[0, 3])
                one_obj_3['box3d']['translation']['y'] = float(obj_extrinsic[1, 3])
                one_obj_3['box3d']['translation']['z'] = float(obj_extrinsic[2, 3])
                
                obj_3.append(one_obj_3)
                
                # LiDAR Label
                one_obj_lidar['class'] = objectClass
                one_obj_lidar['box3d'] = {}
                one_obj_lidar['box3d']['size'] = {}
                one_obj_lidar['box3d']['size']['height'] = bbox3d_h
                one_obj_lidar['box3d']['size']['width'] = bbox3d_w
                one_obj_lidar['box3d']['size']['length'] = bbox3d_l
                
                obj_extrinsic = np.concatenate([R_obj, t_obj.reshape(3, 1)], axis=1)
                obj_extrinsic = np.concatenate([obj_extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
                obj_extrinsic = lidar_extrinsic @ np.linalg.inv(extrinsic_02) @ obj_extrinsic
                
                one_obj_lidar['box3d']['rotation'] = [float(x) for x in obj_extrinsic[:3, :3].flatten()]
                one_obj_lidar['box3d']['translation'] = {}
                one_obj_lidar['box3d']['translation']['x'] = float(obj_extrinsic[0, 3])
                one_obj_lidar['box3d']['translation']['y'] = float(obj_extrinsic[1, 3])
                one_obj_lidar['box3d']['translation']['z'] = float(obj_extrinsic[2, 3])
                
                obj_lidar.append(one_obj_lidar)
                
        label_2['object'] = obj_2
        label_3['object'] = obj_3
        label_lidar['object'] = obj_lidar
        
        # Save
        if idx < 6000:
            with open(f"{target_dir}/Train/StereoLeft/Label/{sub_dir}/{str(idx).zfill(6)}.json", 'w') as f:
                json.dump(label_2, f, indent=4)
            with open(f"{target_dir}/Train/StereoRight/Label/{sub_dir}/{str(idx).zfill(6)}.json", 'w') as f:
                json.dump(label_3, f, indent=4)
            with open(f"{target_dir}/Train/LiDAR/Label/{sub_dir}/{str(idx).zfill(6)}.json", 'w') as f:
                json.dump(label_lidar, f, indent=4)
                
            cv2.imwrite(f"{target_dir}/Train/StereoLeft/Image/{sub_dir}/{str(idx).zfill(6)}.png", image_2)
            cv2.imwrite(f"{target_dir}/Train/StereoRight/Image/{sub_dir}/{str(idx).zfill(6)}.png", image_3)
            shutil.copy(velodyne_file, f"{target_dir}/Train/LiDAR/LiDAR/{sub_dir}/{str(idx).zfill(6)}.bin")
        else:
            with open(f"{target_dir}/Valid/StereoLeft/Label/{sub_dir}/{str(idx).zfill(6)}.json", 'w') as f:
                json.dump(label_2, f, indent=4)
            with open(f"{target_dir}/Valid/StereoRight/Label/{sub_dir}/{str(idx).zfill(6)}.json", 'w') as f:
                json.dump(label_3, f, indent=4)
            with open(f"{target_dir}/Valid/LiDAR/Label/{sub_dir}/{str(idx).zfill(6)}.json", 'w') as f:
                json.dump(label_lidar, f, indent=4)
                
            cv2.imwrite(f"{target_dir}/Valid/StereoLeft/Image/{sub_dir}/{str(idx).zfill(6)}.png", image_2)
            cv2.imwrite(f"{target_dir}/Valid/StereoRight/Image/{sub_dir}/{str(idx).zfill(6)}.png", image_3)
            shutil.copy(velodyne_file, f"{target_dir}/Valid/LiDAR/LiDAR/{sub_dir}/{str(idx).zfill(6)}.bin")
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', type=str, default='D:/kitti3ddet')
    parser.add_argument('--target_dir', type=str, default='D:/KITTI_3D_Detection')
    args = parser.parse_args()

    source_dir = f"{args.source_dir}/training"

    convert(source_dir, args.target_dir)