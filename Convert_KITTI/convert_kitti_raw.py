import os
import cv2
import numpy as np
import json
import argparse

import xml.etree.ElementTree as ET
import uuid
from tqdm import tqdm

validation_list = [
    '2011_09_26_drive_0011_sync',
    '2011_09_26_drive_0017_sync',
    '2011_09_26_drive_0022_sync',
    '2011_09_26_drive_0029_sync',
    '2011_09_26_drive_0039_sync',
    '2011_09_26_drive_0052_sync',
    '2011_09_26_drive_0060_sync',
    '2011_09_26_drive_0079_sync'
]

def making_directories(target_dir):
    if not os.path.exists(os.path.join(target_dir, 'Train')):
        os.makedirs(os.path.join(target_dir, 'Train'))
        
    if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoLeft')):
        os.makedirs(os.path.join(target_dir, 'Train', 'StereoLeft'))
    if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoLeft', 'Image')):
        os.makedirs(os.path.join(target_dir, 'Train', 'StereoLeft', 'Image'))
    if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoLeft', 'Label')):
        os.makedirs(os.path.join(target_dir, 'Train', 'StereoLeft', 'Label'))
        
    if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoRight')):
        os.makedirs(os.path.join(target_dir, 'Train', 'StereoRight'))
    if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoRight', 'Image')):
        os.makedirs(os.path.join(target_dir, 'Train', 'StereoRight', 'Image'))
    if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoRight', 'Label')):
        os.makedirs(os.path.join(target_dir, 'Train', 'StereoRight', 'Label'))
        
    if not os.path.exists(os.path.join(target_dir, 'Train', 'LiDAR')):
        os.makedirs(os.path.join(target_dir, 'Train', 'LiDAR'))
    if not os.path.exists(os.path.join(target_dir, 'Train', 'LiDAR', 'LiDAR')):
        os.makedirs(os.path.join(target_dir, 'Train', 'LiDAR', 'LiDAR'))
    if not os.path.exists(os.path.join(target_dir, 'Train', 'LiDAR', 'Label')):
        os.makedirs(os.path.join(target_dir, 'Train', 'LiDAR', 'Label'))
    
    
    if not os.path.exists(os.path.join(target_dir, 'Valid')):
        os.makedirs(os.path.join(target_dir, 'Valid'))
        
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoLeft')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'StereoLeft'))
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Image')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Image'))
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Label')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Label'))
        
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoRight')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'StereoRight'))
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoRight', 'Image')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'StereoRight', 'Image'))
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoRight', 'Label')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'StereoRight', 'Label'))
        
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'LiDAR')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'LiDAR'))
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'LiDAR', 'LiDAR')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'LiDAR', 'LiDAR'))
    if not os.path.exists(os.path.join(target_dir, 'Valid', 'LiDAR', 'Label')):
        os.makedirs(os.path.join(target_dir, 'Valid', 'LiDAR', 'Label'))

def making_sub_directories(target_dir, sub_dir, validation=False):
    if not validation:
        if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoLeft', 'Image', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Train', 'StereoLeft', 'Image', sub_dir))
        if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoLeft', 'Label', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Train', 'StereoLeft', 'Label', sub_dir))
        
        if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoRight', 'Image', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Train', 'StereoRight', 'Image', sub_dir))
        if not os.path.exists(os.path.join(target_dir, 'Train', 'StereoRight', 'Label', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Train', 'StereoRight', 'Label', sub_dir))
        
        if not os.path.exists(os.path.join(target_dir, 'Train', 'LiDAR', 'LiDAR', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Train', 'LiDAR', 'LiDAR', sub_dir))
        if not os.path.exists(os.path.join(target_dir, 'Train', 'LiDAR', 'Label', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Train', 'LiDAR', 'Label', sub_dir))
    else:
        if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Image', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Image', sub_dir))
        if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Label', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Label', sub_dir))
        
        if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoRight', 'Image', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Valid', 'StereoRight', 'Image', sub_dir))
        if not os.path.exists(os.path.join(target_dir, 'Valid', 'StereoRight', 'Label', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Valid', 'StereoRight', 'Label', sub_dir))
        
        if not os.path.exists(os.path.join(target_dir, 'Valid', 'LiDAR', 'LiDAR', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Valid', 'LiDAR', 'LiDAR', sub_dir))
        if not os.path.exists(os.path.join(target_dir, 'Valid', 'LiDAR', 'Label', sub_dir)):
            os.makedirs(os.path.join(target_dir, 'Valid', 'LiDAR', 'Label', sub_dir))

def converting(source_dir, target_dir):
    calib_dir = os.path.join(source_dir, 'calib')
    data_dir = os.path.join(source_dir, 'data')
    
    making_directories(target_dir)
    
    # Read Camera Calibration
    with open(os.path.join(calib_dir, 'calib_cam_to_cam.txt')) as f:
        calib = f.readlines()
    calib = [x.strip() for x in calib]
    calib = [x.split(': ') for x in calib]
    calib = {x[0]: x[1].split(' ') for x in calib}
    
    # 02: left color camera
    # 03: right color camera
    intrinsic_left = np.array(calib['P_rect_00'], dtype=np.float32).reshape(3, 4)[:3, :3]
    intrinsic_right = np.array(calib['P_rect_00'], dtype=np.float32).reshape(3, 4)[:3, :3]
    extrinsic_left = np.eye(4)
    extrinsic_left[:3, 3] = np.array(calib['P_rect_02'], dtype=np.float32).reshape(3, 4)[:3, 3] / np.array(calib['P_rect_02'], dtype=np.float32).reshape(3, 4)[0, 0]
    extrinsic_right = np.eye(4)
    extrinsic_right[:3, 3] = np.array(calib['P_rect_03'], dtype=np.float32).reshape(3, 4)[:3, 3] / np.array(calib['P_rect_03'], dtype=np.float32).reshape(3, 4)[0, 0]
    
    R0 = np.array(calib['R_rect_00'], dtype=np.float32).reshape(3, 3)
    
    # Read LiDAR Calibration
    with open(os.path.join(calib_dir, 'calib_velo_to_cam.txt')) as f:
        calib = f.readlines()
    calib = [x.strip() for x in calib]
    calib = [x.split(': ') for x in calib]
    calib = {x[0]: x[1].split(' ') for x in calib}
    
    R = np.array(calib['R'], dtype=np.float32).reshape(3, 3)
    t = np.array(calib['T'], dtype=np.float32).reshape(3, 1)
    
    extrinsic_velo = np.vstack((np.hstack((R, t)), np.array([0, 0, 0, 1])))
    extrinsic_velo[:3, :3] = R0 @ extrinsic_velo[:3, :3]
    
    # Read Data
    print('Converting Data...')
    for sub_dir in tqdm(os.listdir(data_dir)):
        is_validation = sub_dir in validation_list
        making_sub_directories(target_dir, sub_dir, is_validation)
        
        image_left_list = os.listdir(os.path.join(data_dir, sub_dir, 'image_02', 'data'))
        image_right_list = os.listdir(os.path.join(data_dir, sub_dir, 'image_03', 'data'))
        image_left_list.sort()
        image_right_list.sort()
        
        lidar_list = os.listdir(os.path.join(data_dir, sub_dir, 'velodyne_points', 'data'))
        lidar_list.sort()
        
        tracklet = os.path.join(data_dir, sub_dir, 'tracklet_labels.xml')
        tree = ET.parse(tracklet)
        root = tree.getroot()
        
        n_data = len(image_left_list)
        
        labels = []
        objectids = []
        for i in range(n_data):
            labels.append([])
            objectids.append([])
            
        tracklets = root.find('tracklets')
        items = tracklets.findall('item')
        for item in items:
            first_frame = int(item.find('first_frame').text)
            object_class = item.find('objectType').text
            
            obj_height = float(item.find('h').text)
            obj_width = float(item.find('w').text)
            obj_length = float(item.find('l').text)
            
            poses = item.find('poses')
            pose = poses.findall('item')
            
            obj_id = str(uuid.uuid4().hex)
            
            for idx, p in enumerate(pose):
                tx = float(p.find('tx').text)
                ty = float(p.find('ty').text)
                tz = float(p.find('tz').text)
                rx = float(p.find('rx').text)
                ry = float(p.find('ry').text)
                rz = float(p.find('rz').text)
                
                if rx != 0 or ry != 0:
                    print(rx, ry)
                    continue
                
                if rz > np.pi:
                    rz -= 2 * np.pi
                    
                points_3d = np.array([
                    [obj_length, obj_length, -obj_length, -obj_length, obj_length, obj_length, -obj_length, -obj_length],
                    [obj_width, -obj_width, -obj_width, obj_width, obj_width, -obj_width, -obj_width, obj_width],
                    [obj_height, obj_height, obj_height, obj_height, -obj_height, -obj_height, -obj_height, -obj_height]
                ])
                
                points_3d[0, :] -= obj_length / 2
                points_3d[1, :] += obj_width / 2
                points_3d[2, :] -= obj_height / 2
                
                tz = tz + obj_height / 2
                
                ext_obj = np.array([
                    [np.cos(rz), -np.sin(rz), 0, tx],
                    [np.sin(rz), np.cos(rz), 0, ty],
                    [0, 0, 1, tz],
                    [0, 0, 0, 1]
                ])
                
                points_3d = ext_obj @ np.vstack((points_3d, np.ones((1, 8))))
                
                labels[first_frame + idx].append([object_class, [obj_height, obj_width, obj_length], [tx, ty, tz, rx, ry, rz], points_3d, ext_obj])
                objectids[first_frame + idx].append(obj_id)
                
        for idx, (image_left_file, image_right_file, lidar_file) in enumerate(zip(image_left_list, image_right_list, lidar_list)):
            image_left = cv2.imread(os.path.join(data_dir, sub_dir, 'image_02', 'data', image_left_file))
            image_right = cv2.imread(os.path.join(data_dir, sub_dir, 'image_03', 'data', image_right_file))
            lidar = np.fromfile(os.path.join(data_dir, sub_dir, 'velodyne_points', 'data', lidar_file), dtype=np.float32).reshape(-1, 4)
            
            imh, imw, _ = image_left.shape

            # Save Image
            if not is_validation:
                cv2.imwrite(os.path.join(target_dir, 'Train', 'StereoLeft', 'Image', sub_dir, image_left_file), image_left)
                cv2.imwrite(os.path.join(target_dir, 'Train', 'StereoRight', 'Image', sub_dir, image_right_file), image_right)
                lidar.tofile(os.path.join(target_dir, 'Train', 'LiDAR', 'LiDAR', sub_dir, lidar_file))
            else:
                cv2.imwrite(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Image', sub_dir, image_left_file), image_left)
                cv2.imwrite(os.path.join(target_dir, 'Valid', 'StereoRight', 'Image', sub_dir, image_right_file), image_right)
                lidar.tofile(os.path.join(target_dir, 'Valid', 'LiDAR', 'LiDAR', sub_dir, lidar_file))
            
            # Save StereoLeft Label
            label = {}
            label['intrinsic'] = {}
            label['intrinsic']['fx'] = float(intrinsic_left[0, 0])
            label['intrinsic']['fy'] = float(intrinsic_left[1, 1])
            label['intrinsic']['cx'] = float(intrinsic_left[0, 2])
            label['intrinsic']['cy'] = float(intrinsic_left[1, 2])
            
            label['extrinsic'] = {}
            label['extrinsic']['rotation'] = list(extrinsic_left[:3, :3].reshape(-1))
            label['extrinsic']['translation'] = {}
            label['extrinsic']['translation']['x'] = float(extrinsic_left[0, 3])
            label['extrinsic']['translation']['y'] = float(extrinsic_left[1, 3])
            label['extrinsic']['translation']['z'] = float(extrinsic_left[2, 3])
            
            label['objects'] = []
            for obj_idx, obj in enumerate(labels[idx]):
                obj_class = obj[0]
                h, w, l = obj[1]
                tx, ty, tz, rx, ry, rz = obj[2]
                points_3d = obj[3]
                ext_obj = obj[4]
                
                obj_label = {}
                obj_label['id'] = objectids[idx][obj_idx]
                obj_label['class'] = obj_class
                
                obj_label['box3d'] = {}
                obj_label['box3d']['size'] = {}
                obj_label['box3d']['size']['height'] = float(h)
                obj_label['box3d']['size']['width'] = float(w)
                obj_label['box3d']['size']['length'] = float(l)
                
                ext_obj = extrinsic_left @ extrinsic_velo @ ext_obj
                
                obj_label['box3d']['rotation'] = list(ext_obj[:3, :3].reshape(-1))
                
                obj_label['box3d']['translation'] = {}
                obj_label['box3d']['translation']['x'] = float(ext_obj[0, 3])
                obj_label['box3d']['translation']['y'] = float(ext_obj[1, 3])
                obj_label['box3d']['translation']['z'] = float(ext_obj[2, 3])
                
                label['objects'].append(obj_label)
                
            if not is_validation:
                with open(os.path.join(target_dir, 'Train', 'StereoLeft', 'Label', sub_dir, image_left_file.replace('png', 'json')), 'w') as f:
                    json.dump(label, f, indent=4)
            else:
                with open(os.path.join(target_dir, 'Valid', 'StereoLeft', 'Label', sub_dir, image_left_file.replace('png', 'json')), 'w') as f:
                    json.dump(label, f, indent=4)
                
            # Save StereoRight Label
            label = {}
            label['intrinsic'] = {}
            label['intrinsic']['fx'] = float(intrinsic_right[0, 0])
            label['intrinsic']['fy'] = float(intrinsic_right[1, 1])
            label['intrinsic']['cx'] = float(intrinsic_right[0, 2])
            label['intrinsic']['cy'] = float(intrinsic_right[1, 2])
            
            label['extrinsic'] = {}
            label['extrinsic']['rotation'] = list(extrinsic_right[:3, :3].reshape(-1))
            label['extrinsic']['translation'] = {}
            label['extrinsic']['translation']['x'] = float(extrinsic_right[0, 3])
            label['extrinsic']['translation']['y'] = float(extrinsic_right[1, 3])
            label['extrinsic']['translation']['z'] = float(extrinsic_right[2, 3])
            
            label['objects'] = []
            for obj_idx, obj in enumerate(labels[idx]):
                obj_class = obj[0]
                h, w, l = obj[1]
                tx, ty, tz, rx, ry, rz = obj[2]
                points_3d = obj[3]
                ext_obj = obj[4]
                
                obj_label = {}
                obj_label['id'] = objectids[idx][obj_idx]
                obj_label['class'] = obj_class

                obj_label['box3d'] = {}
                obj_label['box3d']['size'] = {}
                obj_label['box3d']['size']['height'] = float(h)
                obj_label['box3d']['size']['width'] = float(w)
                obj_label['box3d']['size']['length'] = float(l)
                
                ext_obj = extrinsic_right @ extrinsic_velo @ ext_obj
                
                obj_label['box3d']['rotation'] = list(ext_obj[:3, :3].reshape(-1))
                
                obj_label['box3d']['translation'] = {}
                obj_label['box3d']['translation']['x'] = float(ext_obj[0, 3])
                obj_label['box3d']['translation']['y'] = float(ext_obj[1, 3])
                obj_label['box3d']['translation']['z'] = float(ext_obj[2, 3])
                
                label['objects'].append(obj_label)
                
            if not is_validation:
                with open(os.path.join(target_dir, 'Train', 'StereoRight', 'Label', sub_dir, image_right_file.replace('png', 'json')), 'w') as f:
                    json.dump(label, f, indent=4)
            else:
                with open(os.path.join(target_dir, 'Valid', 'StereoRight', 'Label', sub_dir, image_right_file.replace('png', 'json')), 'w') as f:
                    json.dump(label, f, indent=4)
                        
            # Save LiDAR Label
            label = {}
            label['columns'] = ['x', 'y', 'z', 'intensity']
            label['extrinsic'] = {}
            ext_lidar = np.linalg.inv(extrinsic_velo)
            label['extrinsic']['rotation'] = ext_lidar[:3, :3].tolist()
            label['extrinsic']['translation'] = {}
            label['extrinsic']['translation']['x'] = ext_lidar[0, 3]
            label['extrinsic']['translation']['y'] = ext_lidar[1, 3]
            label['extrinsic']['translation']['z'] = ext_lidar[2, 3]
            
            label['objects'] = []
            for obj_idx, obj in enumerate(labels[idx]):
                obj_class = obj[0]
                h, w, l = obj[1]
                tx, ty, tz, rx, ry, rz = obj[2]
                ext_obj = obj[4]
                
                obj_label = {}
                obj_label['id'] = objectids[idx][obj_idx]
                obj_label['class'] = obj_class
                
                obj_label['box3d'] = {}
                obj_label['box3d']['size'] = {}
                obj_label['box3d']['size']['height'] = h
                obj_label['box3d']['size']['width'] = w
                obj_label['box3d']['size']['length'] = l
                
                obj_label['box3d']['rotation'] = ext_obj[:3, :3].tolist()
                
                obj_label['box3d']['translation'] = {}
                obj_label['box3d']['translation']['x'] = ext_obj[0, 3]
                obj_label['box3d']['translation']['y'] = ext_obj[1, 3]
                obj_label['box3d']['translation']['z'] = ext_obj[2, 3]
                
                label['objects'].append(obj_label)
                
            if not is_validation:
                with open(os.path.join(target_dir, 'Train', 'LiDAR', 'Label', sub_dir, lidar_file.replace('bin', 'json')), 'w') as f:
                    json.dump(label, f, indent=4)
            else:
                with open(os.path.join(target_dir, 'Valid', 'LiDAR', 'Label', sub_dir, lidar_file.replace('bin', 'json')), 'w') as f:
                    json.dump(label, f, indent=4)
                        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process directories.")
    parser.add_argument('--source_root_dir', type=str, required=False, help='Path to the source root directory', default='D:\kittiraw')
    parser.add_argument('--output_root_dir', type=str, required=False, help='Path to the output root directory', default='D:\KITTI_Raw')

    args = parser.parse_args()
    
    converting(args.source_root_dir, args.output_root_dir)