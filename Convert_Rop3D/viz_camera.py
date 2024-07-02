import cv2
import numpy as np
import os
import json
import yaml
import scipy


source_dir = 'D:/Rope3D_cits/Rope3D_data'
file_list_txt = 'train_2100.txt'

with open(os.path.join(source_dir, 'ImageSets', file_list_txt), 'r') as f:
    file_list = f.readlines()
    
file_list.sort()
for file_name in file_list:
    # remove "\n"
    if "\n" in file_name:
        file_name = file_name[:-1]
    print(file_name)
    
    calib_file_path = os.path.join(source_dir, 'calib', file_name + '.txt')
    denorm_file_path = os.path.join(source_dir, 'denorm', file_name + '.txt')
    extrinsics_file_path = os.path.join(source_dir, 'extrinsics', file_name + '.yaml')
    image_file_path = os.path.join(source_dir, 'image_2', file_name + '.jpg')
    label_file_path = os.path.join(source_dir, 'label_2', file_name + '.txt')
    
    if not os.path.exists(calib_file_path):
        print('File not found: ', calib_file_path)
        continue
    if not os.path.exists(denorm_file_path):
        print('File not found: ', denorm_file_path)
        continue
    if not os.path.exists(extrinsics_file_path):
        print('File not found: ', extrinsics_file_path)
        continue
    if not os.path.exists(image_file_path):
        print('File not found: ', image_file_path)
        continue
    if not os.path.exists(label_file_path):
        print('File not found: ', label_file_path)
        continue
    
    # Load calibration matrix
    with open(calib_file_path, 'r') as f:
        calib = f.readlines()[0].split(' ')[1:]
    
    fx = float(calib[0])
    fy = float(calib[5])
    cx = float(calib[2])
    cy = float(calib[6])
    intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    
    # Load Extrinsics
    with open(extrinsics_file_path, 'r') as f:
        extrinsics = yaml.load(f, Loader=yaml.FullLoader)
        
    qw = extrinsics['transform']['rotation']['w']
    qx = extrinsics['transform']['rotation']['x']
    qy = extrinsics['transform']['rotation']['y']
    qz = extrinsics['transform']['rotation']['z']
    
    tx = extrinsics['transform']['translation']['x']
    ty = extrinsics['transform']['translation']['y']
    tz = extrinsics['transform']['translation']['z']
    
    tx = 0
    ty = 0
    tz = 0
    
    cam_rot = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
                        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
                        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
    cam_trans = np.array([[tx], [ty], [tz]])
    cam_extrinsics = np.vstack([np.hstack([cam_rot, cam_trans]), np.array([0, 0, 0, 1])])
    
    # Load Denorm
    with open(denorm_file_path, 'r') as f:
        denorm = f.readlines()[0].split(' ')
    denorm = [float(x) for x in denorm]
    
    
    # Load Image
    image = cv2.imread(image_file_path)
    imh, imw, _ = image.shape
    
    # Load Label
    with open(label_file_path, 'r') as f:
        labels = f.readlines()
        
    edges_front_back = [[0, 1], [1, 2], [2, 3], [3, 0], 
                        [4, 5], [5, 6], [6, 7], [7, 4]]
    edges_side = [[0, 4], [1, 5], [2, 6], [3, 7]]
        
        
    for label in labels:
        if '\n' in label:
            label = label[:-1]
        label = label.split(' ')
        object_type = label[0]
        label = [float(x) for x in label[1:]]
        
        # 2D Bounding Box
        x1, y1, x2, y2 = label[3:7]
        
        # 3D Bounding Box
        h, w, l = label[7:10]
        x, y, z = label[10:13]
        ry = label[13]
        
        # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        # cv2.putText(image, object_type, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        
        if x < 0.1 and y < 0.1 and z < 0.1:
            continue
        
        points = np.array([
            [l/2, l/2, l/2, l/2, -l/2, -l/2, -l/2, -l/2],
            [0, 0, -h, -h, 0, 0, -h, -h],
            [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],
        ])
        
        # Object Rotation Matrix using ry
        R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        rot_denorm = np.array([[1, 0, 0], [0, -denorm[1], +denorm[2]], [0, -denorm[2], -denorm[1]]])
        
        # Transform Objects
        points = np.dot(R, points)
        points = np.dot(rot_denorm, points)
        points = points + np.array([[x], [y], [z]])

        # Project to Image
        points = np.dot(intrinsics, points)
        points = points[:2, :] / points[2, :]
        
        # Draw 3D Bounding Box
        for edge in edges_front_back:
            cv2.line(image, (int(points[0, edge[0]]), int(points[1, edge[0]])), (int(points[0, edge[1]]), int(points[1, edge[1]])), (255, 0, 0), 2)
        for edge in edges_side:
            cv2.line(image, (int(points[0, edge[0]]), int(points[1, edge[0]])), (int(points[0, edge[1]]), int(points[1, edge[1]])), (0, 255, 0), 2)

        # Draw 3D Bounding Box axis
        points_axis = np.array([[0, -h/2, 0], [l/2, -h/2, 0], [0, -h, 0], [0, -h/2, w/2]]).T
        points_axis = np.dot(R, points_axis)
        points_axis = np.dot(rot_denorm, points_axis)
        points_axis = points_axis + np.array([[x], [y], [z]])
        points_axis = np.dot(intrinsics, points_axis)
        points_axis = points_axis[:2, :] / points_axis[2, :]
        
        cv2.line(image, (int(points_axis[0, 0]), int(points_axis[1, 0])), (int(points_axis[0, 1]), int(points_axis[1, 1])), (0, 0, 255), 2)
        cv2.line(image, (int(points_axis[0, 0]), int(points_axis[1, 0])), (int(points_axis[0, 2]), int(points_axis[1, 2])), (0, 255, 0), 2)
        cv2.line(image, (int(points_axis[0, 0]), int(points_axis[1, 0])), (int(points_axis[0, 3]), int(points_axis[1, 3])), (255, 0, 0), 2)

    image = cv2.resize(image, (int(imw/2), int(imh/2)))
    cv2.imshow('image', image)
    cv2.waitKey(0)