import cv2
import numpy as np
import os
import json
import yaml
import scipy

def parsing(source_dir, target_dir, file_list_txt):
    if not os.path.exists(target_dir):
        return
    
    if 'train' in file_list_txt:
        if not os.path.exists(os.path.join(target_dir, 'Train')):
            os.makedirs(os.path.join(target_dir, 'Train'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Train'))
            
        if not os.path.exists(os.path.join(target_dir, 'Train/Camera')):
            os.makedirs(os.path.join(target_dir, 'Train/Camera'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Train/Camera'))
            
        if not os.path.exists(os.path.join(target_dir, 'Train/Camera/Image')):
            os.makedirs(os.path.join(target_dir, 'Train/Camera/Image'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Train/Camera/Image'))
            
        if not os.path.exists(os.path.join(target_dir, 'Train/Camera/Label')):
            os.makedirs(os.path.join(target_dir, 'Train/Camera/Label'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Train/Camera/Label'))    
    else:
        if not os.path.exists(os.path.join(target_dir, 'Valid')):
            os.makedirs(os.path.join(target_dir, 'Valid'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Valid'))
        
        if not os.path.exists(os.path.join(target_dir, 'Valid/Label')):
            os.makedirs(os.path.join(target_dir, 'Valid/Label'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Valid/Label'))
            
        if not os.path.exists(os.path.join(target_dir, 'Valid/Label')):
            os.makedirs(os.path.join(target_dir, 'Valid/Label'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Valid/Label'))
        
        if not os.path.exists(os.path.join(target_dir, 'Valid/Camera/Image')):
            os.makedirs(os.path.join(target_dir, 'Valid/Camera/Image'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Valid/Camera/Image'))
            
        if not os.path.exists(os.path.join(target_dir, 'Valid/Camera/Label')):
            os.makedirs(os.path.join(target_dir, 'Valid/Camera/Label'))
        else:
            os.system('rm -r ' + os.path.join(target_dir, 'Valid/Camera/Label'))
    
    if 'train' in file_list_txt:
        image_dir = os.path.join(target_dir, 'Train/Camera/Image')
        label_dir = os.path.join(target_dir, 'Train/Camera/Label')
    else:
        image_dir = os.path.join(target_dir, 'Valid/Camera/Image')
        label_dir = os.path.join(target_dir, 'Valid/Camera/Label')
        
    with open(os.path.join(source_dir, 'ImageSets', file_list_txt), 'r') as f:
        file_list = f.readlines()
        
    file_list.sort()
    for file_name in file_list:
        # remove "\n"
        if "\n" in file_name:
            file_name = file_name[:-1]
        print(file_name)
        
        image_subfolder_path = os.path.join(image_dir, file_name.split('_')[1])
        label_subfolder_path = os.path.join(label_dir, file_name.split('_')[1])
        if not os.path.exists(image_subfolder_path):
            os.makedirs(image_subfolder_path)
        if not os.path.exists(label_subfolder_path):
            os.makedirs(label_subfolder_path)
        
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
        
        file_name_save = ''
        for idx, fn in enumerate(file_name.split('_')):
            if idx != 5:
                file_name_save += fn
            else:
                index = int(fn)
                file_name_save += str(index).zfill(6)
                
            if idx != 6:
                file_name_save += '_'
        
        image = cv2.imread(image_file_path)
        imh, imw, _ = image.shape
        cv2.imwrite(os.path.join(image_subfolder_path, file_name_save + '.jpg'), image)
        
        # Load calibration matrix
        with open(calib_file_path, 'r') as f:
            calib = f.readlines()[0].split(' ')[1:]
        
        fx = float(calib[0])
        fy = float(calib[5])
        cx = float(calib[2])
        cy = float(calib[6])
        
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
        
        # cam_rot = np.array([[1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        #                     [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        #                     [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]])
        cam_trans = np.array([[tx], [ty], [tz]])
        
        # Load Denorm
        with open(denorm_file_path, 'r') as f:
            denorm = f.readlines()[0].split(' ')
        denorm = [float(x) for x in denorm]
        
        theta = np.arctan2(denorm[2], denorm[1])
        sin = np.sin(theta)
        cos = np.cos(theta)
        
        cam_rot = np.array([
            [1, 0, 0],
            [0, cos, -sin],
            [0, sin, cos]
        ])
        
        l2c = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ])
        
        cam_rot = cam_rot @ l2c
        
        
        # Load Image
        image = cv2.imread(image_file_path)
        imh, imw, _ = image.shape
        
        # Load Label
        with open(label_file_path, 'r') as f:
            labels = f.readlines()

        camera_label_dict = {}
        camera_label_dict['intrinsic'] = {}
        camera_label_dict['intrinsic']['fx'] = fx
        camera_label_dict['intrinsic']['fy'] = fy
        camera_label_dict['intrinsic']['cx'] = cx
        camera_label_dict['intrinsic']['cy'] = cy
        
        camera_label_dict['extrinsic'] = {}
        camera_label_dict['extrinsic']['rotation'] = cam_rot.reshape(-1).tolist()
        camera_label_dict['extrinsic']['translation'] = {}
        camera_label_dict['extrinsic']['translation']['x'] = 0
        camera_label_dict['extrinsic']['translation']['y'] = 0
        camera_label_dict['extrinsic']['translation']['z'] = 0
        
        camera_label_dict['objects'] = []
        for label in labels:
            if '\n' in label:
                label = label[:-1]
            label = label.split(' ')
            object_type = label[0]
            label = [float(x) for x in label[1:]]
            
            # 2D Bounding Box
            x1, y1, x2, y2 = label[3:7]
            
            obj = {}
            obj['class'] = object_type
            obj['box2d'] = {}
            obj['box2d']['cx'] = (x1 + x2) / 2 / imw
            obj['box2d']['cy'] = (y1 + y2) / 2 / imh
            obj['box2d']['w'] = (x2 - x1) / imw
            obj['box2d']['h'] = (y2 - y1) / imh
            
            obj['box3d'] = {}
            
            # 3D Bounding Box
            h, w, l = label[7:10]
            x, y, z = label[10:13]
            ry = label[13]
            
            # cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            # cv2.putText(image, object_type, (int(x1), int(y1)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
            if x < 0.1 and y < 0.1 and z < 0.1:
                continue
            
            # Object Rotation Matrix using ry
            R = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
            rot_denorm = np.array([[1, 0, 0], [0, -denorm[1], +denorm[2]], [0, -denorm[2], -denorm[1]]])
            
            obj_rotation = rot_denorm @ R
            obj_rotation = obj_rotation.reshape(-1).tolist()
            obj_translation = [x, y, z]
            
            obj['box3d']['size'] = {}
            obj['box3d']['size']['width'] = h
            obj['box3d']['size']['height'] = w
            obj['box3d']['size']['length'] = l
            obj['box3d']['rotation'] = obj_rotation
            obj['box3d']['translation'] = {}
            obj['box3d']['translation']['x'] = x
            obj['box3d']['translation']['y'] = y - h / 2
            obj['box3d']['translation']['z'] = z
            
            camera_label_dict['objects'].append(obj)
            
        # Save LiDAR label as .json
        with open(os.path.join(label_subfolder_path, f"{file_name_save}.json"), 'w') as f:
            json.dump(camera_label_dict, f, indent=4)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process directories.")
    parser.add_argument('--source_root_dir', type=str, required=False, help='Path to the source root directory', default='D:\Rope3D_data')
    parser.add_argument('--output_root_dir', type=str, required=False, help='Path to the output root directory', default='D:\Rope3D_cits')

    args = parser.parse_args()

    source_dir = args.source_root_dir

    target_dir, file_list_txt = os.path.join(args.output_root_dir , "Rope3D_2100"), 'train_2100.txt'
    parsing(source_dir, target_dir, file_list_txt)

    target_dir, file_list_txt = os.path.join(args.output_root_dir , "Rope3D_2700"), 'train_2700.txt'
    parsing(source_dir, target_dir, file_list_txt)

    target_dir, file_list_txt = os.path.join(args.output_root_dir , "Rope3D_2100"), 'test_2100.txt'
    parsing(source_dir, target_dir, file_list_txt)

    target_dir, file_list_txt = os.path.join(args.output_root_dir , "Rope3D_2700"), 'test_2700.txt'
    parsing(source_dir, target_dir, file_list_txt)