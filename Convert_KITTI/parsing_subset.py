import os, shutil
import cv2, json
import numpy as np
#uuid
import uuid

def parse_file(root_dir):
    calib_dir = f"{root_dir}/calib"
    image_2_dir = f"{root_dir}/image_2"
    image_3_dir = f"{root_dir}/image_3"
    velodyne_dir = f"{root_dir}/velodyne"

    target_dir = "/home/harsper/바탕화면/KITTI_Subset"
    if not os.path.exists(f"{target_dir}/KITTI_StereoL_Subset"):
        os.makedirs(f"{target_dir}/KITTI_StereoL_Subset")
    if not os.path.exists(f"{target_dir}/KITTI_StereoR_Subset"):
        os.makedirs(f"{target_dir}/KITTI_StereoR_Subset")
    if not os.path.exists(f"{target_dir}/KITTI_LiDAR_Subset"):
        os.makedirs(f"{target_dir}/KITTI_LiDAR_Subset")

    if not os.path.exists(f"{target_dir}/KITTI_StereoL_Subset/Image"):
        os.makedirs(f"{target_dir}/KITTI_StereoL_Subset/Image")
    if not os.path.exists(f"{target_dir}/KITTI_StereoL_Subset/Label"):
        os.makedirs(f"{target_dir}/KITTI_StereoL_Subset/Label")
    if not os.path.exists(f"{target_dir}/KITTI_StereoR_Subset/Image"):
        os.makedirs(f"{target_dir}/KITTI_StereoR_Subset/Image")
    if not os.path.exists(f"{target_dir}/KITTI_StereoR_Subset/Label"):
        os.makedirs(f"{target_dir}/KITTI_StereoR_Subset/Label")
    if not os.path.exists(f"{target_dir}/KITTI_LiDAR_Subset/LiDAR"):
        os.makedirs(f"{target_dir}/KITTI_LiDAR_Subset/LiDAR")
    if not os.path.exists(f"{target_dir}/KITTI_LiDAR_Subset/Label"):
        os.makedirs(f"{target_dir}/KITTI_LiDAR_Subset/Label")


    for idx in range(7481):
        sub_dir = idx // 1000

        # if idx >10:
        #     return

        if not os.path.exists(f"{target_dir}/KITTI_StereoL_Subset/Image/{sub_dir}"):
            os.makedirs(f"{target_dir}/KITTI_StereoL_Subset/Image/{sub_dir}")
        if not os.path.exists(f"{target_dir}/KITTI_StereoL_Subset/Label/{sub_dir}"):
            os.makedirs(f"{target_dir}/KITTI_StereoL_Subset/Label/{sub_dir}")
        if not os.path.exists(f"{target_dir}/KITTI_StereoR_Subset/Image/{sub_dir}"):
            os.makedirs(f"{target_dir}/KITTI_StereoR_Subset/Image/{sub_dir}")
        if not os.path.exists(f"{target_dir}/KITTI_StereoR_Subset/Label/{sub_dir}"):
            os.makedirs(f"{target_dir}/KITTI_StereoR_Subset/Label/{sub_dir}")
        if not os.path.exists(f"{target_dir}/KITTI_LiDAR_Subset/LiDAR/{sub_dir}"):
            os.makedirs(f"{target_dir}/KITTI_LiDAR_Subset/LiDAR/{sub_dir}")
        if not os.path.exists(f"{target_dir}/KITTI_LiDAR_Subset/Label/{sub_dir}"):
            os.makedirs(f"{target_dir}/KITTI_LiDAR_Subset/Label/{sub_dir}")

        calib_file = f"{calib_dir}/{idx:06d}.txt"
        image_2_file = f"{image_2_dir}/{idx:06d}.png"
        image_3_file = f"{image_3_dir}/{idx:06d}.png"
        velodyne_file = f"{velodyne_dir}/{idx:06d}.bin"
        label_file = f"{root_dir}/label_2/{idx:06d}.txt"

        if not os.path.exists(calib_file):
            print(f"File {calib_file} does not exist")
        if not os.path.exists(image_2_file):
            print(f"File {image_2_file} does not exist")
        if not os.path.exists(image_3_file):
            print(f"File {image_3_file} does not exist")
        if not os.path.exists(velodyne_file):
            print(f"File {velodyne_file} does not exist")

        # read calib file
        with open(calib_file, 'r') as f:
            calib = f.readlines()
            P2 = np.array(calib[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            P3 = np.array(calib[3].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            R0_rect = np.array(calib[4].strip().split(' ')[1:], dtype=np.float32).reshape(3, 3)
            Tr_velo_to_cam = np.array(calib[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)
            Tr_imu_to_velo = np.array(calib[6].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)

        # TODO: 확인 필요
        # P2[:3, :3] = P2[:3, :3] @ R0_rect

        P2 = np.concatenate([P2, np.array([[0, 0, 0, 1]])], axis=0)
        out = cv2.decomposeProjectionMatrix(P2[:3, :])
        K2 = out[0]
        R2 = out[1]
        t2 = -(out[2][:3] / out[2][3])

        P3 = np.concatenate([P3, np.array([[0, 0, 0, 1]])], axis=0)
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

        # read image_2 file
        image_2 = cv2.imread(image_2_file)
        if image_2 is None:
            print(f"Error reading image {image_2_file}")
        # read image_3 file
        image_3 = cv2.imread(image_3_file)
        if image_3 is None:
            print(f"Error reading image {image_3_file}")


        # read label file
        label_2 = dict()
        label_3 = dict()
        label_lidar = dict()
        obj_2 = []
        obj_3 = []
        obj_lidar = []

        label_2['extrinsic_rotation'] = [str(x) for x in R2.reshape(-1)]
        label_2['extrinsic_translation'] = [str(x) for x in t2.reshape(-1)]
        label_2['intrinsic'] = [str(x) for x in K2.reshape(-1)]
        
        label_3['extrinsic_rotation'] = [str(x) for x in R3.reshape(-1)]
        label_3['extrinsic_translation'] = [str(x) for x in t3.reshape(-1)]
        label_3['intrinsic'] = [str(x) for x in K3.reshape(-1)]

        label_lidar['columns'] = ['x', 'y', 'z', 'intensity']
        Tr_velo_to_cam = np.concatenate([Tr_velo_to_cam, np.array([[0, 0, 0, 1]])], axis=0)
        lidar_extrinsic = np.linalg.inv(Tr_velo_to_cam)
        label_lidar['extrinsic_rotation'] = [str(x) for x in lidar_extrinsic[:3, :3].reshape(-1)]
        label_lidar['extrinsic_translation'] = [str(x) for x in lidar_extrinsic[:3, 3].reshape(-1)]
        # label_lidar['extrinsic_rotation'] = [str(x) for x in np.ones((3, 3)).reshape(-1)]
        # label_lidar['extrinsic_translation'] = [str(x) for x in np.zeros((1, 3)).reshape(-1)]

        with open(label_file, 'r') as f:
            labels = f.readlines()
            labels = [label.strip().split(' ') for label in labels]
            
            for label in labels:
                one_obj_2 = dict()
                one_obj_3 = dict()
                one_obj_lidar = dict()
                objectClass = label[0] #'Car', 'Pedestrian', ...
                if objectClass == 'DontCare':
                    continue

                object_id = str(uuid.uuid4().hex)

                truncated = float(label[1])
                occluded = float(label[2])
                alpha = float(label[3])

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

                one_obj_2['id'] = object_id
                one_obj_2['class'] = objectClass

                box3d = dict()
                size = dict()
                size['height'] = bbox3d_h
                size['width'] = bbox3d_w
                size['length'] = bbox3d_l
                box3d['size'] = size

                yz_change = np.array([
                    [1, 0, 0],
                    [0, 0, -1],
                    [0, 1, 0]
                ])

                rot_y = rot_y @ yz_change

                box3d['rotation'] = rot_y.reshape(-1).tolist()
                box3d['translation'] = [bbox3d_x, bbox3d_y - (bbox3d_h / 2), bbox3d_z]

                R_obj = rot_y
                t_obj = np.array([bbox3d_x, bbox3d_y - (bbox3d_h / 2), bbox3d_z]).reshape(3, 1)

                box2d = dict()
                box2d['cx'] = ((bbox2d_x1 + bbox2d_x2) / 2) / image_2.shape[1]
                box2d['cy'] = ((bbox2d_y1 + bbox2d_y2) / 2) / image_2.shape[0]
                box2d['w'] = (bbox2d_x2 - bbox2d_x1) / image_2.shape[1]
                box2d['h'] = (bbox2d_y2 - bbox2d_y1) / image_2.shape[0]

                one_obj_2['box3d'] = box3d
                one_obj_2['box2d'] = box2d

                obj_2.append(one_obj_2)
    #########################
                one_obj_3['id'] = object_id
                one_obj_3['class'] = objectClass

                box3d = dict()
                size = dict()
                size['height'] = bbox3d_h
                size['width'] = bbox3d_w
                size['length'] = bbox3d_l
                box3d['size'] = size

                obj_extrinsic = np.concatenate([R_obj, t_obj], axis=1)
                obj_extrinsic = np.concatenate([obj_extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
                obj_extrinsic = extrinsic_03 @ np.linalg.inv(extrinsic_02) @ obj_extrinsic

                box3d['rotation'] = obj_extrinsic[:3, :3].reshape(-1).tolist()
                box3d['translation'] = obj_extrinsic[:3, 3].reshape(-1).tolist()

                one_obj_3['box3d'] = box3d

                obj_3.append(one_obj_3)

#####################
                one_obj_lidar['id'] = object_id
                one_obj_lidar['class'] = objectClass

                box3d = dict()
                size = dict()
                size['height'] = bbox3d_h
                size['width'] = bbox3d_w
                size['length'] = bbox3d_l
                box3d['size'] = size

                obj_extrinsic = np.concatenate([R_obj, t_obj], axis=1)
                obj_extrinsic = np.concatenate([obj_extrinsic, np.array([[0, 0, 0, 1]])], axis=0)
                obj_extrinsic = lidar_extrinsic @ np.linalg.inv(extrinsic_02) @ obj_extrinsic

                box3d['rotation'] = obj_extrinsic[:3, :3].reshape(-1).tolist()
                box3d['translation'] = obj_extrinsic[:3, 3].reshape(-1).tolist()

                one_obj_lidar['box3d'] = box3d

                obj_lidar.append(one_obj_lidar)

                


            label_2['object'] = obj_2
            label_3['object'] = obj_3
            label_lidar['object'] = obj_lidar

            filename = f"{idx}".zfill(10)
            with open(os.path.join(f"{target_dir}/KITTI_StereoL_Subset/Label/{sub_dir}/{filename}.json"), 'w') as f:
                json.dump(label_2, f, indent=4)
            with open(os.path.join(f"{target_dir}/KITTI_StereoR_Subset/Label/{sub_dir}/{filename}.json"), 'w') as f:
                json.dump(label_3, f, indent=4)
            with open(os.path.join(f"{target_dir}/KITTI_LiDAR_Subset/Label/{sub_dir}/{filename}.json"), 'w') as f:
                json.dump(label_lidar, f, indent=4)

            # write image
            cv2.imwrite(os.path.join(f"{target_dir}/KITTI_StereoL_Subset/Image/{sub_dir}/{filename}.jpg"), image_2)
            cv2.imwrite(os.path.join(f"{target_dir}/KITTI_StereoR_Subset/Image/{sub_dir}/{filename}.jpg"), image_3)
            shutil.copy(os.path.join(f"{velodyne_file}"), 
                        os.path.join(f"{target_dir}/KITTI_LiDAR_Subset/LiDAR/{sub_dir}/{filename}.bin"))


parse_file('/home/harsper/kitti_original/training')