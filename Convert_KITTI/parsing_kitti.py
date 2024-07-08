import os
import cv2, json
import numpy as np

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import Element, ElementTree, SubElement

import shutil
import uuid

from utils import *

dirs = ['/home/harsper/kitti_original_full/2011_09_26']

if not os.path.exists('KITTI_LiDAR/LiDAR'):
    os.makedirs('KITTI_LiDAR/LiDAR')

if not os.path.exists('KITTI_LiDAR/Label'):
    os.makedirs('KITTI_LiDAR/Label')

if not os.path.exists('KITTI_StereoL/Image'):
    os.makedirs('KITTI_StereoL/Image')

if not os.path.exists('KITTI_StereoL/Label'):
    os.makedirs('KITTI_StereoL/Label')

if not os.path.exists('KITTI_StereoR/Image'):
    os.makedirs('KITTI_StereoR/Image')

if not os.path.exists('KITTI_StereoR/Label'):
    os.makedirs('KITTI_StereoR/Label')

for dir_main in dirs:
    file_list_temp = os.listdir(dir_main)
    file_list_temp.sort()

    file_list = []
    for file in file_list_temp:
        if os.path.isdir(os.path.join(dir_main, file)):
            file_list.append(file)

    #read calib
    with open(f"{dir_main}/calib_cam_to_cam.txt", 'r') as f:
        calib = f.readlines()
        calib = [line.strip().split(' ') for line in calib]
        P_rect_00 = np.array(calib[9][1:13]).astype(np.float32).reshape(3, 4)
        K_02 = np.array(calib[19][1:10]).astype(np.float32).reshape(3, 3)
        K_03 = np.array(calib[27][1:10]).astype(np.float32).reshape(3, 3)
        R_rect_02 = np.array(calib[24][1:10]).astype(np.float32).reshape(3, 3)
        R_rect_03 = np.array(calib[32][1:10]).astype(np.float32).reshape(3, 3)
        P_rect_02 = np.array(calib[25][1:13]).astype(np.float32).reshape(3, 4)
        P_rect_03 = np.array(calib[33][1:13]).astype(np.float32).reshape(3, 4)
        R_rect_00 = np.array(calib[8][1:10]).astype(np.float32).reshape(3, 3)
    
    intrinsic_02 = P_rect_00[:, :3]
    intrinsic_03 = P_rect_00[:, :3]
    extrinsic_02 = np.eye(4, 4)
    extrinsic_02[0, 3] = P_rect_02[0, 3] / P_rect_02[0, 0]
    extrinsic_03 = np.eye(4, 4)
    extrinsic_03[0, 3] = P_rect_03[0, 3] / P_rect_03[0, 0]
   

    with open(f"{dir_main}/calib_velo_to_cam.txt", 'r') as f:
        calib = f.readlines()
        calib = [line.strip().split(' ') for line in calib]
        R = np.array(calib[1][1:10]).astype(np.float32).reshape(3, 3)
        T = np.array(calib[2][1:10]).astype(np.float32).reshape(3, 1)

    extrinsic_lidar = np.vstack((np.hstack((R, T)), np.array([0, 0, 0, 1])))
    R_rect_00 = np.vstack((R_rect_00, np.array([0, 0, 0])))
    R_rect_00 = np.hstack((R_rect_00, np.array([[0], [0], [0], [1]])))
    extrinsic_lidar = np.dot(R_rect_00, extrinsic_lidar)

    for dir_sub in file_list:
        #Image
        image_left_list = os.listdir(f"{dir_main}/{dir_sub}/image_02/data")
        image_right_list = os.listdir(f"{dir_main}/{dir_sub}/image_03/data")
        image_left_list.sort()
        image_right_list.sort()

        #Lidar
        lidar_list = os.listdir(f"{dir_main}/{dir_sub}/velodyne_points/data")
        lidar_list.sort()

        #Label (xml)
        tracklet = ET.parse(f"{dir_main}/{dir_sub}/tracklet_labels.xml")
        tracklet_root = tracklet.getroot()

        n_data = len(image_left_list)

        labels = []
        objectids = []
        projected_2d_left = []
        projected_2d_right = []
        for i in range(n_data):
            labels.append([])
            objectids.append([])
            projected_2d_left.append([])
            projected_2d_right.append([])

        tracklets = tracklet_root.find('tracklets')
        items = tracklets.findall('item')
        for item in items:
            first_frame = int(item.find('first_frame').text)
            ObjectClass = item.find('objectType').text
            if ObjectClass == 'Tram':
                continue
            if ObjectClass == 'Misc':
                continue
            h = item.find('h').text
            w = item.find('w').text
            l = item.find('l').text

            poses = item.find('poses')
            pose = poses.findall('item')
            for idx, p in enumerate(pose):

                tx = p.find('tx').text
                ty = p.find('ty').text
                tz = p.find('tz').text
                rx = float(p.find('rx').text)
                ry = float(p.find('ry').text)
                rz = float(p.find('rz').text)

                if rx != '0' or ry != '0':
                    print(rx, ry)

                if float(rz) > np.pi:
                    rz = float(rz) - 2 * np.pi

                points_before = np.array([
                    [float(l)/2, float(w)/2, float(h)/2],
                    [float(l)/2, -float(w)/2, float(h)/2],
                    [-float(l)/2, -float(w)/2, float(h)/2],
                    [-float(l)/2, float(w)/2, float(h)/2],
                    [float(l)/2, float(w)/2, -float(h)/2],
                    [float(l)/2, -float(w)/2, -float(h)/2],
                    [-float(l)/2, -float(w)/2, -float(h)/2],
                    [-float(l)/2, float(w)/2, -float(h)/2]
                ])

                tz = float(tz) + float(h) / 2
                
                extrinsic_object = np.array([
                    [np.cos(rz), -np.sin(rz), 0, float(tx)],
                    [np.sin(rz), np.cos(rz), 0, float(ty)],
                    [0, 0, 1, float(tz)],
                    [0, 0, 0, 1]
                ])

                points = np.hstack((points_before, np.ones((8, 1))))
                points = np.dot(extrinsic_object, points.T).T

                # Class, [h, w, l], [rz], [tx, ty, tz], points_before, points
                labels[first_frame+idx].append([ObjectClass, [h, w, l], [rx, ry, rz], [tx, ty, tz], points_before, points, extrinsic_object[:3, :3]])
                objectids[first_frame+idx].append(uuid.uuid4().hex)

        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]

        for idx, (image_left_file, image_right_file, lidar_file) in enumerate(zip(image_left_list, image_right_list, lidar_list)):
            print(f"{dir_main}/{dir_sub}/image_02/data/{image_left_file}")
            image_left = cv2.imread(f"{dir_main}/{dir_sub}/image_02/data/{image_left_file}")
            image_right = cv2.imread(f"{dir_main}/{dir_sub}/image_03/data/{image_right_file}")
            lidar = np.fromfile(f"{dir_main}/{dir_sub}/velodyne_points/data/{lidar_file}", dtype=np.float32).reshape(-1, 4)

            # lidar_to_image = np.linalg.inv(extrinsic_lidar)
            lidar_to_image = extrinsic_lidar

            imh, imw, _ = image_left.shape

            for li, label in enumerate(labels[idx]):
                ObjectClass = label[0]
                h, w, l = label[1]
                rx, ry, rz = label[2]
                tx, ty, tz = label[3]
                points_before = label[4]
                points = label[5]
                rot_matrix = label[6]

                #project to image_left
                points_left = points
                points_left = np.dot(lidar_to_image, points_left.T).T
                points_left = np.dot(extrinsic_02, points_left.T).T
                points_left = points_left[:, :-1]
                points_left_projected = np.dot(intrinsic_02, points_left.T).T
                points_left_projected = points_left_projected[:, :2] / points_left_projected[:, 2:]

                x1_left = np.clip(int(np.min(points_left_projected[:, 0])), 0, imw)
                x2_left = np.clip(int(np.max(points_left_projected[:, 0])), 0, imw)
                y1_left = np.clip(int(np.min(points_left_projected[:, 1])), 0, imh)
                y2_left = np.clip(int(np.max(points_left_projected[:, 1])), 0, imh)

                # forward line
                center_left = np.mean(points_left, axis=0)
                forward_left = np.mean(points_left[[0, 1, 4, 5]], axis=0)
                forward_points_left = np.array([center_left, forward_left])
                forward_points_projected_left = np.dot(intrinsic_02, forward_points_left.T).T
                forward_points_projected_left = forward_points_projected_left[:, :2] / forward_points_projected_left[:, 2:]

                #project to image_right
                points_right = points
                points_right = np.dot(lidar_to_image, points_right.T).T
                points_right = np.dot(extrinsic_03, points_right.T).T
                points_right = points_right[:, :-1]
                points_right_projected = np.dot(intrinsic_03, points_right.T).T
                points_right_projected = points_right_projected[:, :2] / points_right_projected[:, 2:]

                x1_right = np.clip(int(np.min(points_right_projected[:, 0])), 0, imw)
                x2_right = np.clip(int(np.max(points_right_projected[:, 0])), 0, imw)
                y1_right = np.clip(int(np.min(points_right_projected[:, 1])), 0, imh)
                y2_right = np.clip(int(np.max(points_right_projected[:, 1])), 0, imh)

                # forward line
                center_right = np.mean(points_right, axis=0)
                forward_right = np.mean(points_right[[0, 1, 4, 5]], axis=0)
                forward_points_right = np.array([center_right, forward_right])
                forward_points_projected_right = np.dot(intrinsic_02, forward_points_right.T).T
                forward_points_projected_right = forward_points_projected_right[:, :2] / forward_points_projected_right[:, 2:]

                projected_2d_left[idx].append([x1_left, y1_left, x2_left, y2_left])
                projected_2d_right[idx].append([x1_right, y1_right, x2_right, y2_right])

                #draw
                for edge in edges:
                    cv2.line(image_left, (int(points_left_projected[edge[0]][0]), int(points_left_projected[edge[0]][1])), (int(points_left_projected[edge[1]][0]), int(points_left_projected[edge[1]][1])), (0, 0, 255), 1)
                    cv2.line(image_right, (int(points_right_projected[edge[0]][0]), int(points_right_projected[edge[0]][1])), (int(points_right_projected[edge[1]][0]), int(points_right_projected[edge[1]][1])), (0, 0, 255), 1)
                cv2.putText(image_left, ObjectClass, (int(points_left_projected[0][0]), int(points_left_projected[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                cv2.putText(image_right, ObjectClass, (int(points_right_projected[0][0]), int(points_right_projected[0][1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)

                cv2.rectangle(image_left, (x1_left, y1_left), (x2_left, y2_left), (0, 255, 0), 1)
                cv2.rectangle(image_right, (x1_right, y1_right), (x2_right, y2_right), (0, 255, 0), 1)

                cv2.line(image_left, (int(forward_points_projected_left[0][0]), int(forward_points_projected_left[0][1])), (int(forward_points_projected_left[1][0]), int(forward_points_projected_left[1][1])), (0, 255, 0), 1)
                cv2.line(image_right, (int(forward_points_projected_right[0][0]), int(forward_points_projected_right[0][1])), (int(forward_points_projected_right[1][0]), int(forward_points_projected_right[1][1])), (0, 255, 0), 1)

            image = np.vstack((image_left, image_right))

            cv2.imshow("image", image)
            cv2.waitKey(10)

            if not os.path.exists(f"KITTI_StereoL/Label/{dir_sub}"):
                os.makedirs(f"KITTI_StereoL/Label/{dir_sub}")
            
            if not os.path.exists(f"KITTI_StereoR/Label/{dir_sub}"):
                os.makedirs(f"KITTI_StereoR/Label/{dir_sub}")

            if not os.path.exists(f"KITTI_StereoL/Image/{dir_sub}"):
                os.makedirs(f"KITTI_StereoL/Image/{dir_sub}")
            
            if not os.path.exists(f"KITTI_StereoR/Image/{dir_sub}"):
                os.makedirs(f"KITTI_StereoR/Image/{dir_sub}")
            
            if not os.path.exists(f"KITTI_LiDAR/LiDAR/{dir_sub}"):
                os.makedirs(f"KITTI_LiDAR/LiDAR/{dir_sub}")
            
            if not os.path.exists(f"KITTI_LiDAR/Label/{dir_sub}"):
                os.makedirs(f"KITTI_LiDAR/Label/{dir_sub}")

            image_left = cv2.imread(f"{dir_main}/{dir_sub}/image_02/data/{image_left_file}")
            image_right = cv2.imread(f"{dir_main}/{dir_sub}/image_03/data/{image_right_file}")
            lidar = np.fromfile(f"{dir_main}/{dir_sub}/velodyne_points/data/{lidar_file}", dtype=np.float32).reshape(-1, 4)

            cv2.imwrite(f"KITTI_StereoL/Image/{dir_sub}/{image_left_file.split('.')[0]}.jpg", image_left)
            cv2.imwrite(f"KITTI_StereoR/Image/{dir_sub}/{image_right_file.split('.')[0]}.jpg", image_right)
            shutil.copy(f"{dir_main}/{dir_sub}/velodyne_points/data/{lidar_file}", f"KITTI_LiDAR/LiDAR/{dir_sub}/{lidar_file}")

            #write StereoL Label
            root = dict()
            global_rotation = extrinsic_02[:3, :3]
            root['extrinsic_rotation'] = [str(x) for x in global_rotation.reshape(-1)]
            root['extrinsic_translation'] = [str(x) for x in extrinsic_02[:3, 3].reshape(-1)]
            root['intrinsic'] = [str(x) for x in intrinsic_02.reshape(-1)]

            root['object'] = []
            for obj_id, label in enumerate(labels[idx]):
                obj = dict()
                obj['id'] = objectids[idx][obj_id]
                obj['class'] = label[0]

                box3d = dict()
                size = dict()
                size['height'] = float(label[1][0])
                size['width'] = float(label[1][1])
                size['length'] = float(label[1][2])
                box3d['size'] = size

                obj_rz = float(label[2][2])
                obj_rotation = np.array([
                    [np.cos(obj_rz), -np.sin(obj_rz), 0],
                    [np.sin(obj_rz), np.cos(obj_rz), 0],
                    [0, 0, 1]
                ])
                obj_translation = np.array(label[3]).reshape(-1, 1)
                obj_extrinsic = np.vstack((np.hstack((obj_rotation, obj_translation.reshape(-1, 1))).astype(np.float32), np.array([0, 0, 0, 1]))).astype(np.float32)
                obj_extrinsic = np.dot(lidar_to_image, obj_extrinsic)
                obj_extrinsic = np.dot(extrinsic_02, obj_extrinsic)

                rotation = obj_extrinsic[:3, :3]
                box3d['rotation'] = [str(x) for x in rotation.reshape(-1)]
                box3d['translation'] = [str(x) for x in obj_extrinsic[:3, 3].reshape(-1)]

                box2d = dict()
                x1 = projected_2d_left[idx][obj_id][0]
                y1 = projected_2d_left[idx][obj_id][1]
                x2 = projected_2d_left[idx][obj_id][2]
                y2 = projected_2d_left[idx][obj_id][3]
                box2d['cx'] = float((x1 + x2) / 2 / imw)
                box2d['cy'] = float((y1 + y2) / 2 / imh)
                box2d['w'] = float((x2 - x1) / imw)
                box2d['h'] = float((y2 - y1) / imh)
                
                obj['box3d'] = box3d
                obj['box2d'] = box2d

                root['object'].append(obj)

            # json write
            json_file = f"KITTI_StereoL/Label/{dir_sub}/{image_left_file.split('.')[0]}.json"
            with open(json_file, 'w') as f:
                json.dump(root, f, indent=4)


            # #write StereoR Label
            root = dict()
            root['extrinsic_rotation'] = [str(x) for x in extrinsic_03[:3, :3].reshape(-1)]
            root['extrinsic_translation'] = [str(x) for x in extrinsic_03[:3, 3].reshape(-1)]
            root['intrinsic'] = [str(x) for x in intrinsic_03.reshape(-1)]

            root['object'] = list()
            for obj_id, label in enumerate(labels[idx]):
                obj = dict()
                obj['id'] = objectids[idx][obj_id]
                obj['class'] = label[0]

                box3d = dict()
                size = dict()

                size['height'] = float(label[1][0])
                size['width'] = float(label[1][1])
                size['length'] = float(label[1][2])
                box3d['size'] = size

                obj_rz = float(label[2][2])
                obj_rotation = np.array([
                    [np.cos(obj_rz), -np.sin(obj_rz), 0],
                    [np.sin(obj_rz), np.cos(obj_rz), 0],
                    [0, 0, 1]
                ])
                obj_translation = np.array(label[3]).reshape(-1, 1)
                obj_extrinsic = np.vstack((np.hstack((obj_rotation, obj_translation.reshape(-1, 1))).astype(np.float32), np.array([0, 0, 0, 1]))).astype(np.float32)
                obj_extrinsic = np.dot(lidar_to_image, obj_extrinsic)
                obj_extrinsic = np.dot(extrinsic_03, obj_extrinsic)

                rotation = obj_extrinsic[:3, :3]
                box3d['rotation'] = [str(x) for x in rotation.reshape(-1)]
                translation = obj_extrinsic[:3, 3]
                box3d['translation'] = [str(x) for x in translation.reshape(-1)]

                box2d = dict()
                x1 = projected_2d_right[idx][obj_id][0]
                y1 = projected_2d_right[idx][obj_id][1]
                x2 = projected_2d_right[idx][obj_id][2]
                y2 = projected_2d_right[idx][obj_id][3]
                box2d['cx'] = float((x1 + x2) / 2 / imw)
                box2d['cy'] = float((y1 + y2) / 2 / imh)
                box2d['w'] = float((x2 - x1) / imw)
                box2d['h'] = float((y2 - y1) / imh)

                obj['box3d'] = box3d
                obj['box2d'] = box2d
                
                root['object'].append(obj)

            json_file = f"KITTI_StereoR/Label/{dir_sub}/{image_right_file.split('.')[0]}.json"
            with open(json_file, 'w') as f:
                json.dump(root, f, indent=4)

            #write LiDAR Label
            root = dict()
            root['columns'] = ['x', 'y', 'z', 'intensity']
            extrinsic_lidar_inv = np.linalg.inv(extrinsic_lidar)
            global_rotation = extrinsic_lidar_inv[:3, :3]
            root['extrinsic_rotation'] = [str(x) for x in global_rotation.reshape(-1)]
            global_translation = extrinsic_lidar_inv[:3, 3]
            root['extrinsic_translation'] = [str(x) for x in global_translation.reshape(-1)]

            root['object'] = list()
            for obj_id, label in enumerate(labels[idx]):
                obj = dict()
                obj['id'] = objectids[idx][obj_id]
                obj['class'] = label[0]

                box3d = dict()

                size = dict()
                size['height'] = float(label[1][0])
                size['width'] = float(label[1][1])
                size['length'] = float(label[1][2])
                box3d['size'] = size
                
                obj_rz = float(label[2][2])
                obj_rotation = np.array([
                    [np.cos(obj_rz), -np.sin(obj_rz), 0],
                    [np.sin(obj_rz), np.cos(obj_rz), 0],
                    [0, 0, 1]
                ])
                obj_translation = np.array(label[3]).reshape(-1, 1)

                rotation = obj_rotation
                box3d['rotation'] = [str(x) for x in rotation.reshape(-1)]
                translation = obj_translation.reshape(-1)
                box3d['translation'] = [str(x) for x in translation]

                obj['box3d'] = box3d

                root['object'].append(obj)

            json_file = f"KITTI_LiDAR/Label/{dir_sub}/{lidar_file.split('.')[0]}.json"
            with open(json_file, 'w') as f:
                json.dump(root, f, indent=4)



