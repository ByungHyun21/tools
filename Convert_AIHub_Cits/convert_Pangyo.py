# convert from Training to Train
# convert from Validation to Valid

import json
import os
from pyntcloud import PyntCloud
import cv2
import numpy as np
import uuid
from scipy.spatial.transform import Rotation as R

def get_idx(folder_name):
    # example: L_2210_Suwon_A_A_C0167
    # return: 167
    return int(folder_name.split('_')[-1][1:])

def converting(path_from, path_to, isTrain):    
    origin_path = os.path.join(path_from, 'source')
    label_path = os.path.join(path_from, 'label')
    
    output_camera_dir = os.path.join(path_to, 'Camera')
    output_lidar_dir = os.path.join(path_to, 'LiDAR')
    output_camera_image_dir = os.path.join(output_camera_dir, 'Image')
    output_camera_label_dir = os.path.join(output_camera_dir, 'Label')
    output_lidar_lidar_dir = os.path.join(output_lidar_dir, 'LiDAR')
    output_lidar_label_dir = os.path.join(output_lidar_dir, 'Label')
    
    if not os.path.exists(output_camera_dir):
        os.makedirs(output_camera_dir)
    
    if not os.path.exists(output_lidar_dir):
        os.makedirs(output_lidar_dir)
    
    if not os.path.exists(output_camera_image_dir):
        os.makedirs(output_camera_image_dir)
    
    if not os.path.exists(output_camera_label_dir):
        os.makedirs(output_camera_label_dir)
    
    if not os.path.exists(output_lidar_lidar_dir):
        os.makedirs(output_lidar_lidar_dir)
    
    if not os.path.exists(output_lidar_label_dir):
        os.makedirs(output_lidar_label_dir)
    
        
    folder_list = os.listdir(origin_path)
    
    count = 0
    max_count = len(folder_list)
    for folder in folder_list:
        count = count + 1
        # if count % 100 != 0:
        #     count += 1
        #     continue
        
        # Grouping folders
        if "L_2210_Suwon_A_A" in folder:
            #Train: C0204 부터 Suwon_B 로 가야함 (라이다 위쪽)
            #Valid: C0222 부터 Suwon_B 로 가야함 (라이다 위쪽)
            
            if isTrain:
                if get_idx(folder) < 204:
                    folder_output = folder.replace("L_2210_Suwon_A_A", "Suwon_A_1")
                else:
                    folder_output = folder.replace("L_2210_Suwon_A_A", "Suwon_BU_1")
            else:
                if get_idx(folder) < 222:
                    folder_output = folder.replace("L_2210_Suwon_A_A", "Suwon_A_1")
                else:
                    folder_output = folder.replace("L_2210_Suwon_A_A", "Suwon_BU_1")
                    
        elif "L_2210_Suwon_B_F" in folder:
            folder_output = folder.replace("L_2210_Suwon_B_F", "Suwon_A_2")
        elif "L_2210_Suwon_B_N" in folder:
            folder_output = folder.replace("L_2210_Suwon_B_N", "Suwon_A_3")
        elif "L_2211_Suwon_B_A" in folder:
            folder_output = folder.replace("L_2211_Suwon_B_A", "Suwon_A_4")
        elif "L_2211_Suwon_B_D" in folder:
            #Train: C0152 부터 Suwon_B 로 가야함 (라이다 위쪽)
            #Valid: C0163 부터 Suwon_B 로 가야함 (라이다 위쪽)
            
            if isTrain:
                if get_idx(folder) < 152:
                    folder_output = folder.replace("L_2211_Suwon_B_D", "Suwon_A_5")
                else:
                    folder_output = folder.replace("L_2211_Suwon_B_D", "Suwon_BU_5")
            else:
                if get_idx(folder) < 163:
                    folder_output = folder.replace("L_2211_Suwon_B_D", "Suwon_A_5")
                else:
                    folder_output = folder.replace("L_2211_Suwon_B_D", "Suwon_BU_5")
            
        elif "L_2211_Suwon_B_F" in folder:
            folder_output = folder.replace("L_2211_Suwon_B_F", "Suwon_A_6")
        elif "L_2210_Suwon_A_F" in folder:
            #Train: C0201까지 라이다 위치 위쪽임
            #Train: C4651까지 라이다 위치 아래임 (라이다 이상 데이터)
            #Train: C4652부터 Suwon_A로 가야함 (Suwon A Extrinsic 사용해야함)
            #Valid: C0193까지 라이다 위치 위쪽임
            #Valid: C0204부터 라이다 위치 아래임 (라이다 이상 데이터)
            #Valid: C4663부터 Suwon A로 가야함 (Suwon A Extrinsic 사용해야함)
            
            if isTrain:
                if get_idx(folder) <= 201:
                    folder_output = folder.replace("L_2210_Suwon_A_F", "Suwon_BU_7")
                elif get_idx(folder) < 4652:
                    continue
                else:
                    folder_output = folder.replace("L_2210_Suwon_A_F", "Suwon_A_7")
            else:
                if get_idx(folder) <= 193:
                    folder_output = folder.replace("L_2210_Suwon_A_F", "Suwon_BU_7")
                elif get_idx(folder) < 4663:
                    continue
                else:
                    folder_output = folder.replace("L_2210_Suwon_A_F", "Suwon_A_7")
        elif "L_2210_Suwon_A_N" in folder:
            #Valid: 라이다 전부 위쪽임
            
            folder_output = folder.replace("L_2210_Suwon_A_N", "Suwon_BU_8")
        elif "L_2210_Suwon_B_A" in folder:
            #Train: C0201까지 Suwon_B_3 (라이다 위쪽)
            #Train: C0204부터 Suwon_A로 가야함
            #Valid: C0203까지 Suwon_B_3 (라이다 위쪽)
            #Valid: C0214부터 Suwon_A로 가야함
            
            if isTrain:
                if get_idx(folder) < 204:
                    folder_output = folder.replace("L_2210_Suwon_B_A", "Suwon_BU_9")
                else:
                    folder_output = folder.replace("L_2210_Suwon_B_A", "Suwon_A_9")
            else:
                if get_idx(folder) < 214:
                    folder_output = folder.replace("L_2210_Suwon_B_A", "Suwon_BU_9")
                else:
                    folder_output = folder.replace("L_2210_Suwon_B_A", "Suwon_A_9")
            
        elif "L_2211_Suwon_A_A" in folder:
            #Train: C0639까지 라이다 위치 위쪽임
            #Train: C0640부터 라이다 위치 아래임 (라이다 이상 데이터)
            #Valid: C0638까지 라이다 위치 위쪽임
            #Valid: C0643부터 라이다 위치 아래임 (라이다 이상 데이터)
            
            if isTrain:
                if get_idx(folder) < 640:
                    folder_output = folder.replace("L_2211_Suwon_A_A", "Suwon_BU_10")
                else:
                    continue
            else:
                if get_idx(folder) < 643:
                    folder_output = folder.replace("L_2211_Suwon_A_A", "Suwon_BU_10")
                else:
                    continue
            
        elif "L_2211_Suwon_A_D" in folder:
            #Train: C0151까지 라이다 위치 위쪽임
            #Train: C0152부터 C0225까지 Suwon_A로 가야함
            #Train: C0228부터 라이다 위치 아래임 (라이다 이상 데이터)
            #Valid: C0139까지 라이다 위치 위쪽임
            #Valid: C0154부터 C0226까지 Suwon_A로 가야함
            #Valid: C0242부터 라이다 위치 아래임 (라이다 이상 데이터)
            
            if isTrain:
                if get_idx(folder) < 152:
                    folder_output = folder.replace("L_2211_Suwon_A_D", "Suwon_BU_11")
                elif get_idx(folder) <= 225:
                    folder_output = folder.replace("L_2211_Suwon_A_D", "Suwon_A_11")
                else:
                    continue
            else:
                if get_idx(folder) <= 139:
                    folder_output = folder.replace("L_2211_Suwon_A_D", "Suwon_BU_11")
                elif get_idx(folder) <= 226:
                    folder_output = folder.replace("L_2211_Suwon_A_D", "Suwon_A_11")
                else:
                    continue
            
        elif "L_2211_Suwon_A_F" in folder:
            #Train: C0150까지 라이다 위치 위쪽임
            #Train: C0152부터 라이다 위치 아래임 (라이다 이상 데이터)
            #Valid: C0146까지 라이다 위치 위쪽임
            #Valid: C0151부터 라이다 위치 아래임 (라이다 이상 데이터)
            
            if isTrain:
                if get_idx(folder) < 152:
                    folder_output = folder.replace("L_2211_Suwon_A_F", "Suwon_BU_12")
                else:
                    continue
            else:
                if get_idx(folder) < 151:
                    folder_output = folder.replace("L_2211_Suwon_A_F", "Suwon_BU_12")
                else:
                    continue
            
        elif "L_2211_Pangyo_C_A" in folder:
            folder_output = folder.replace("L_2211_Pangyo_C_A", "Pangyo_A_13")
        elif "L_2211_Pangyo_C_D" in folder:
            folder_output = folder.replace("L_2211_Pangyo_C_D", "Pangyo_A_14")
        elif "L_2211_Pangyo_C_F" in folder:
            folder_output = folder.replace("L_2211_Pangyo_C_F", "Pangyo_A_15")
        elif "L_2211_Pangyo_C_N" in folder:
            #Train: C0147 까지 이상데이터
            #Valid: C0124 까지 이상데이터
            
            if isTrain:
                if get_idx(folder) <= 147:
                    continue
            else:
                if get_idx(folder) <= 124:
                    continue
                
            folder_output = folder.replace("L_2211_Pangyo_C_N", "Pangyo_A_16")
        elif "L_2211_Pangyo_D_A" in folder:
            folder_output = folder.replace("L_2211_Pangyo_D_A", "Pangyo_B_17")
        elif "L_2211_Pangyo_D_D" in folder:
            folder_output = folder.replace("L_2211_Pangyo_D_D", "Pangyo_B_18")
        elif "L_2211_Pangyo_D_F" in folder:
            folder_output = folder.replace("L_2211_Pangyo_D_F", "Pangyo_B_19")
            
        print(folder, folder_output, f"{count}/{max_count}")
        calib_path = origin_path + '/' + folder + '/sensor_raw_data' + '/calib'
        image_path = origin_path + '/' + folder + '/sensor_raw_data' + '/camera'
        lidar_path = origin_path + '/' + folder + '/sensor_raw_data' + '/lidar'
        
        calib_path2 = os.listdir(calib_path)[0]
        calib_path = calib_path + '/' + calib_path2
        calib_file = calib_path + '/' + os.listdir(calib_path)[0]
        
        iamge_path2 = os.listdir(image_path)[0]
        image_files = os.listdir(image_path + '/' + iamge_path2)
        
        lidar_files = os.listdir(lidar_path)
        
        label_files = os.listdir(label_path + '/' + folder + '/sensor_raw_data/lidar')
        
        # mkdir to_path
        if not os.path.exists(path_to + '/LiDAR/LiDAR/' + folder_output):
            os.makedirs(path_to + '/LiDAR/LiDAR/' + folder_output)
            
        if not os.path.exists(path_to + '/LiDAR/Label/' + folder_output):
            os.makedirs(path_to + '/LiDAR/Label/' + folder_output)
            
        if not os.path.exists(path_to + '/Camera/Image/' + folder_output):
            os.makedirs(path_to + '/Camera/Image/' + folder_output)
            
        if not os.path.exists(path_to + '/Camera/Label/' + folder_output):
            os.makedirs(path_to + '/Camera/Label/' + folder_output)
        
        #read calib
        with open(calib_file) as f:
            calib = json.load(f)
            
        for img_f, lidar_f, label_f in zip(image_files, lidar_files, label_files):
            cv_image_path = os.path.join(image_path, iamge_path2, img_f)
            img = cv2.imread(cv_image_path)
            
            if "Pangyo_A" in folder_output:
                with open("Pangyo_A.json") as f:
                    int_dist = json.load(f)
            elif "Pangyo_B" in folder_output:
                with open("Pangyo_B.json") as f:
                    int_dist = json.load(f)
            elif "Suwon_A" in folder_output:
                with open("Suwon_A.json") as f:
                    int_dist = json.load(f)
            elif "Suwon_B" in folder_output:
                with open("Suwon_B.json") as f:
                    int_dist = json.load(f)
            
            intrinsic = np.array(int_dist['intrinsic']).reshape(3, 3)
            distortion = np.array(int_dist['distortion'])
            
            h, w = img.shape[:2]
            newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intrinsic, distortion, (w, h), 1, (w, h))
            undistorted = cv2.undistort(img, intrinsic, distortion, None, newcameramtx)
            
            intrinsic = newcameramtx
            # roi
            x, y, w, h = roi
            undistorted = undistorted[y:y+h, x:x+w]
            intrinsic[0, 2] -= x
            intrinsic[1, 2] -= y
            
            # Image
            img_save_name = f"{folder_output}_{img_f.split('_')[-1]}"
            cv2.imwrite(path_to + '/Camera/Image/' + folder_output + '/' + img_save_name, undistorted)
            
            
            lidar = PyntCloud.from_file(lidar_path + '/' + lidar_f)
            points = np.array(lidar.points)
            
            with open(label_path + '/' + folder + '/sensor_raw_data/lidar/' + label_f) as f:
                labels = json.load(f)
            
            
            
            # LiDAR ( save .bin )
            # numpy to bin 
            points = np.array(points, dtype=np.float32)
            points = points.tobytes()
            
            lidar_save_name = f"{folder_output}_{lidar_f.split('_')[-1]}"
            with open(path_to + '/LiDAR/LiDAR/' + folder_output + '/' + lidar_save_name.replace('.pcd', '.bin'), 'wb') as f:
                f.write(points)
                
            # LiDAR Label
            lidar_label_json = {}
            lidar_label_json['columns'] = ['x', 'y', 'z', 'intensity']
            lidar_label_json['extrinsic'] = {}
            lidar_label_json['extrinsic']['rotation'] = [1, 0, 0, 0, 1, 0, 0, 0, 1]
            lidar_label_json['extrinsic']['translation'] = {}
            lidar_label_json['extrinsic']['translation']['x'] = 0
            lidar_label_json['extrinsic']['translation']['y'] = 0
            lidar_label_json['extrinsic']['translation']['z'] = 0
            objects = []
            for label in labels:
                obj = {}
                obj['id'] = uuid.uuid4().hex
                obj['class'] = label['obj_type']
                obj['box3d'] = {}
                obj['box3d']['size'] = {}
                obj['box3d']['size']['width'] = label['psr']['scale']['y']
                obj['box3d']['size']['height'] = label['psr']['scale']['z']
                obj['box3d']['size']['length'] = label['psr']['scale']['x']
                rot_x_rad = label['psr']['rotation']['x']
                rot_y_rad = label['psr']['rotation']['y']
                rot_z_rad = label['psr']['rotation']['z']
                
                rot_x_mat = np.array([[1, 0, 0], 
                                      [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)], 
                                      [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]])
                rot_y_mat = np.array([[np.cos(rot_y_rad), 0, np.sin(rot_y_rad)], 
                                      [0, 1, 0], 
                                      [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]])
                rot_z_mat = np.array([[np.cos(rot_z_rad), -np.sin(rot_z_rad), 0], 
                                      [np.sin(rot_z_rad), np.cos(rot_z_rad), 0], 
                                      [0, 0, 1]])
                # rot = rot_x_mat @ rot_y_mat @ rot_z_mat
                rot = rot_y_mat @ rot_x_mat @ rot_z_mat
                rot = [rot[0][0], rot[0][1], rot[0][2], rot[1][0], rot[1][1], rot[1][2], rot[2][0], rot[2][1], rot[2][2]]
                
                obj['box3d']['rotation'] = rot
                
                obj['box3d']['translation'] = {}
                obj['box3d']['translation']['x'] = label['psr']['position']['x']
                obj['box3d']['translation']['y'] = label['psr']['position']['y']
                obj['box3d']['translation']['z'] = label['psr']['position']['z']
                
                if label['psr']['scale']['x'] < 0.1:
                    continue
                if label['psr']['scale']['y'] < 0.1:
                    continue
                if label['psr']['scale']['z'] < 0.1:
                    continue
                
                objects.append(obj)
            
            lidar_label_json['objects'] = objects
            
            # save json
            label_save_name = f"{folder_output}_{label_f.split('_')[-1]}"
            with open(path_to + '/LiDAR/Label/' + folder_output + '/' + label_save_name, 'w') as f:
                json.dump(lidar_label_json, f, indent=4)
            
            if "Suwon_A" in folder_output:
                rx_rad = np.deg2rad(-179.87110819807293)
                ry_rad = np.deg2rad(0.2765009523758263)
                rz_rad = np.deg2rad(-105.2122093108067)
                tx_m = 0.03876690000000002
                ty_m = -4.31577
                tz_m = -1.14171
                
                rot = R.from_euler('ZYX', [rx_rad, ry_rad, rz_rad], degrees=False).as_matrix()
                extrinsic = np.vstack([np.hstack([rot, np.array([tx_m, ty_m, tz_m]).reshape(3, 1)]), np.array([0, 0, 0, 1])])
            elif "Suwon_BU" in folder_output:
                rx_rad = np.deg2rad(179.98232680999925)
                ry_rad = np.deg2rad(-0.639301430720137)
                rz_rad = np.deg2rad(-118.34683057639378)
                tx_m = -0.07289489999999992
                ty_m = -3.8362600000000002
                tz_m = -2.17651
                
                rot = R.from_euler('ZYX', [rx_rad, ry_rad, rz_rad], degrees=False).as_matrix()
                extrinsic = np.vstack([np.hstack([rot, np.array([tx_m, ty_m, tz_m]).reshape(3, 1)]), np.array([0, 0, 0, 1])])
            elif "Pangyo_A" in folder_output:
                rx_rad = np.deg2rad(179.60999142423705)
                ry_rad = np.deg2rad(-3.188922654624339)
                rz_rad = np.deg2rad(-103.86386543837874)
                tx_m = 0.0890227
                ty_m = 4.82294
                tz_m = 1.23326
                
                rot = R.from_euler('ZYX', [rx_rad, ry_rad, rz_rad], degrees=False).as_matrix()
                extrinsic = np.vstack([np.hstack([rot, np.array([tx_m, ty_m, tz_m]).reshape(3, 1)]), np.array([0, 0, 0, 1])])
            elif "Pangyo_B" in folder_output:
                rx_rad = np.deg2rad(179.81126527412312)
                ry_rad = np.deg2rad(-1.0551613772704485)
                rz_rad = np.deg2rad(-105.09124827414638)
                tx_m = 0.19392
                ty_m = 23.5698
                tz_m = 6.3116
                
                rot = R.from_euler('ZYX', [rx_rad, ry_rad, rz_rad], degrees=False).as_matrix()
                extrinsic = np.vstack([np.hstack([rot, np.array([tx_m, ty_m, tz_m]).reshape(3, 1)]), np.array([0, 0, 0, 1])])
            else :
                # Image Label
                extrinsic = np.array(calib['extrinsic']).reshape(4, 4)
            # extrinsic = np.array(calib['extrinsic']).reshape(4, 4)
            image_label_json = {}
            image_label_json['intrinsic'] = {}
            image_label_json['intrinsic']['fx'] = intrinsic[0, 0]
            image_label_json['intrinsic']['fy'] = intrinsic[1, 1]
            image_label_json['intrinsic']['cx'] = intrinsic[0, 2]
            image_label_json['intrinsic']['cy'] = intrinsic[1, 2]
            image_label_json['extrinsic'] = {}
            image_label_json['extrinsic']['rotation'] = extrinsic[:3, :3].reshape(1, 9).tolist()[0]
            image_label_json['extrinsic']['translation'] = {}
            image_label_json['extrinsic']['translation']['x'] = extrinsic[0, 3]
            image_label_json['extrinsic']['translation']['y'] = extrinsic[1, 3]
            image_label_json['extrinsic']['translation']['z'] = extrinsic[2, 3]
            
            objects_image = []
            
            L2C = np.array([0, -1,  0,  0,
                        0, 0,  -1, 0,
                        1, 0,  0,  0,
                        0, 0,  0,  1]).reshape(4, 4)
            for label in labels:
                obj = {}
                obj['id'] = uuid.uuid4().hex
                obj['class'] = label['obj_type']
                obj['box3d'] = {}
                obj['box3d']['size'] = {}
                obj['box3d']['size']['width'] = label['psr']['scale']['y']
                obj['box3d']['size']['height'] = label['psr']['scale']['z']
                obj['box3d']['size']['length'] = label['psr']['scale']['x']
                rot_x_rad = label['psr']['rotation']['x']
                rot_y_rad = label['psr']['rotation']['y']
                rot_z_rad = label['psr']['rotation']['z']
                
                rot_x_mat = np.array([[1, 0, 0], 
                                      [0, np.cos(rot_x_rad), -np.sin(rot_x_rad)], 
                                      [0, np.sin(rot_x_rad), np.cos(rot_x_rad)]])
                rot_y_mat = np.array([[np.cos(rot_y_rad), 0, np.sin(rot_y_rad)], 
                                      [0, 1, 0], 
                                      [-np.sin(rot_y_rad), 0, np.cos(rot_y_rad)]])
                rot_z_mat = np.array([[np.cos(rot_z_rad), -np.sin(rot_z_rad), 0], 
                                      [np.sin(rot_z_rad), np.cos(rot_z_rad), 0], 
                                      [0, 0, 1]])
                # rot = rot_z_mat @ rot_y_mat @ rot_x_mat
                rot = rot_x_mat @ rot_y_mat @ rot_z_mat
                tx = label['psr']['position']['x']
                ty = label['psr']['position']['y']
                tz = label['psr']['position']['z']
                trans = np.array([tx, ty, tz]).reshape(3, 1)
                
                ext_obj = np.vstack([np.hstack([rot, trans]), np.array([0, 0, 0, 1])])
                ext_obj = extrinsic @ ext_obj
                
                obj['box3d']['rotation'] = ext_obj[:3, :3].reshape(1, 9).tolist()[0]
                obj['box3d']['translation'] = {}
                obj['box3d']['translation']['x'] = ext_obj[0, 3]
                obj['box3d']['translation']['y'] = ext_obj[1, 3]
                obj['box3d']['translation']['z'] = ext_obj[2, 3]
                
                if label['psr']['scale']['x'] < 0.1:
                    continue
                if label['psr']['scale']['y'] < 0.1:
                    continue
                if label['psr']['scale']['z'] < 0.1:
                    continue
                
                objects_image.append(obj)
                
            image_label_json['objects'] = objects_image
            
            # save json
            label_save_name = f"{folder_output}_{img_f.split('_')[-1]}"
            with open(path_to + '/Camera/Label/' + folder_output + '/' + label_save_name.replace('.jpg', '.json'), 'w') as f:
                json.dump(image_label_json, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process directories.")
    parser.add_argument('--source_root_dir', type=str, required=False, help='Path to the source root directory', default='D:\dataset')
    parser.add_argument('--output_root_dir', type=str, required=False, help='Path to the output root directory', default='D:\AIHUB_cits')

    args = parser.parse_args()
    
    converting(os.path.join(args.source_root_dir, 'dataset/Validation'), os.path.join(args.output_root_dir, 'Valid'), False)    
    converting(os.path.join(args.source_root_dir, 'dataset/Training'), os.path.join(args.output_root_dir, 'Train'), True)