import os
import cv2
import json
import numpy as np

def main(root_dir):
    for purpose in ['Train', 'Valid']:
        sensors = os.listdir(os.path.join(root_dir, purpose))    
        for sensor in sensors:
            if 'Image' in os.listdir(os.path.join(root_dir, purpose, sensor)):
                generator(os.path.join(root_dir, purpose, sensor, 'Image'))

def generator(root_dir):
    sub_dirs = os.listdir(root_dir)
    for sub_dir in sub_dirs:
        image_files = os.listdir(os.path.join(root_dir, sub_dir))
        
        for image_file in image_files:
            image = cv2.imread(os.path.join(root_dir, sub_dir, image_file))
            label_file = image_file.split('.')[0] + '.json'
            with open(os.path.join(root_dir.replace('Image', 'Label'), sub_dir, label_file)) as f:
                label = json.load(f)
                
            if 'intrinsic' not in label:
                print('This is not 3D Label: ', image_file)
                continue
            
            fx = label['intrinsic']['fx']
            fy = label['intrinsic']['fy']
            cx = label['intrinsic']['cx']
            cy = label['intrinsic']['cy']
            intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                
            edges_front = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4]
            ]    
            edges_side = [
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            for obj in label['objects']:
                if 'box2d' in obj:
                    continue
                if 'box3d' not in obj:
                    continue
                
                box3d = obj['box3d']
                
                obj_width, obj_height, obj_length = box3d['size']['width'], box3d['size']['height'], box3d['size']['length']
                obj_rotation = np.array(box3d['rotation'], dtype=np.float32).reshape(3, 3)
                tx, ty, tz = box3d['translation']['x'], box3d['translation']['y'], box3d['translation']['z']
                obj_translation = np.array([tx, ty, tz], dtype=np.float32).reshape(3, 1)
                
                # 3D Bounding Box
                vertices = np.array([
                    [obj_length, obj_length, obj_length, obj_length, 0, 0, 0, 0],
                    [0, 0, obj_width, obj_width, 0, 0, obj_width, obj_width],
                    [0, obj_height, obj_height, 0, 0, obj_height, obj_height, 0]
                ])
                
                vertices = vertices - np.array([[obj_length/2], [obj_width/2], [obj_height/2]])
                obj_ext = np.vstack([np.hstack([obj_rotation, obj_translation]), np.array([0, 0, 0, 1])])
                
                vertices = obj_ext @ np.vstack([vertices, np.ones(8)])
                vertices_2d = intrinsic @ vertices[:3, :]
                
                vertices_2d = vertices_2d[:2, :] / vertices_2d[2, :]
                
                # 2D Bounding Box Generation from 3D Bounding Box
                imh, imw = image.shape[:2]
                
                # bool idx that projected in image
                idx = (vertices_2d[0] > 0) & (vertices_2d[0] < imw) & (vertices_2d[1] > 0) & (vertices_2d[1] < imh)
                
                xmin = imw
                ymin = imh
                xmax = 0
                ymax = 0
                
                if sum(idx) == 0:
                    continue
                
                for i in range(8):
                    if idx[i]:
                        xmin = min(xmin, vertices_2d[0, i])
                        ymin = min(ymin, vertices_2d[1, i])
                        xmax = max(xmax, vertices_2d[0, i])
                        ymax = max(ymax, vertices_2d[1, i])
                    else:
                        if (vertices_2d[0, i] > 0) & (vertices_2d[0, i] < imw):
                            if vertices[1, i] > 0:
                                ymax = imh
                            if vertices[1, i] < 0:
                                ymin = 0
                                
                        if (vertices_2d[1, i] > 0) & (vertices_2d[1, i] < imh):
                            if vertices[0, i] > 0:
                                xmax = imw
                            if vertices[0, i] < 0:
                                xmin = 0
                
                for edge in edges_side:
                    x1 = int(vertices_2d[0, edge[0]])
                    y1 = int(vertices_2d[1, edge[0]])
                    x2 = int(vertices_2d[0, edge[1]])
                    y2 = int(vertices_2d[1, edge[1]])
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                for edge in edges_front:
                    x1 = int(vertices_2d[0, edge[0]])
                    y1 = int(vertices_2d[1, edge[0]])
                    x2 = int(vertices_2d[0, edge[1]])
                    y2 = int(vertices_2d[1, edge[1]])
                    cv2.line(image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                
                pass
                
                    
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
            cv2.imshow('image', image)
            cv2.waitKey(0)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:/AIHUB_cits")
    args = parser.parse_args()
    
    main(args.root_dir)