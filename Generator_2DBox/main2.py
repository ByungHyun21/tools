import os
import cv2
import json
import numpy as np
import copy
import time

# 0: 전방 우하단, 1: 전방 우상단, 2: 전방 좌상단, 3: 전방 좌하단
# 4: 후방 우하단, 5: 후방 우상단, 6: 후방 좌상단, 7: 후방 좌하단

def main(root_dir):
    for purpose in ['Train']:
        sensors = os.listdir(os.path.join(root_dir, purpose))    
        for sensor in sensors:
            if 'Image' in os.listdir(os.path.join(root_dir, purpose, sensor)):
                generator(os.path.join(root_dir, purpose, sensor, 'Image'))

def generator(root_dir):
    sub_dirs = os.listdir(root_dir)
    sub_dirs.sort()
    
    triangles = np.array([
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7]
    ])
    
    for sub_dir in sub_dirs:
        image_files = os.listdir(os.path.join(root_dir, sub_dir))
        image_files.sort()
        
        st = time.time()
        cnt = 0
        while True:
            print(f"{sub_dir} - {cnt}/{len(image_files)}")
            image_file = image_files[cnt]
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
            
            vertices3d = []
            n_obj = len(label['objects'])
            
            for obj in label['objects']:
                # if 'box2d' in obj:
                #     continue
                if 'box3d' not in obj:
                    continue
                
                objClass = obj['class']
                
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
                
                vertices3d.append(vertices)
            
            # 2D Bounding Box Generation from 3D Bounding Box
            imh, imw = image.shape[:2]
            
            #####################################################
            # find intersections between 3D bounding box and image plane in 1/4 image
            imh4, imw4 = imh//20, imw//20
            intrinsic4 = copy.deepcopy(intrinsic)
            intrinsic4[:2, :] = intrinsic4[:2, :] / 20
            
            canvas4 = np.zeros((n_obj, imh4, imw4, 3), dtype=np.uint8)
            
            object_inside = [False] * n_obj
            for u in range(0, imw4):
                for v in range(0, imh4):
                    ray4 = np.linalg.inv(intrinsic4) @ np.array([u, v, 1])
                    ray4 = ray4 / np.linalg.norm(ray4)
                    
                    for oidx, p3d in enumerate(vertices3d):
                        for triangle in triangles:
                            A = p3d[:3, triangle[0]]
                            B = p3d[:3, triangle[1]]
                            C = p3d[:3, triangle[2]]
                            
                            AB = B - A
                            BC = C - B
                            CA = A - C
                            
                            #plane: ax + by + cz + d = 0
                            plane_normal = np.cross(AB, BC)
                            d = -np.dot(plane_normal, A)
                            
                            # p = t * ray (from (0, 0, 0))
                            t = -d / np.dot(plane_normal, ray4)
                            p = t * ray4
                            
                            if t < 0:
                                continue
                            
                            pA = A - p
                            pB = B - p
                            pC = C - p
                            
                            cA = np.cross(AB, pA)
                            cB = np.cross(BC, pB)
                            cC = np.cross(CA, pC)
                            
                            b0 = True if np.dot(plane_normal, cA) > 0 else False
                            b1 = True if np.dot(plane_normal, cB) > 0 else False
                            b2 = True if np.dot(plane_normal, cC) > 0 else False
                            
                            if (b0 == b1) and (b1 == b2):
                                canvas4[oidx, v, u] = 1
                                object_inside[oidx] = True
                                break
            
            if not object_inside:
                continue
            
            for oidx, inside in enumerate(object_inside):
                if not inside:
                    continue
                
                # 2D Bounding Box Generation
                temp = cv2.cvtColor(canvas4[oidx], cv2.COLOR_BGR2GRAY)
                _, temp = cv2.threshold(temp, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                
                x1 = max(0, x*20)
                y1 = max(0, y*20)
                x2 = min(imw, (x+w)*20)
                y2 = min(imh, (y+h)*20)
                
                x1 = max(0, x1-20)
                x2 = min(imw, x2+20)
                y1 = max(0, y1-20)
                y2 = min(imh, y2+20)
                
                canvas = np.zeros((imh, imw), dtype=np.uint8)
                for u in range(x1, x2):
                    for v in range(y1, y2):
                        if canvas[v, u] == 1:
                            continue
                        
                        ray = np.linalg.inv(intrinsic) @ np.array([u, v, 1])
                        ray = ray / np.linalg.norm(ray)
                        
                        for triangle in triangles:
                            A = vertices3d[oidx][:3, triangle[0]]
                            B = vertices3d[oidx][:3, triangle[1]]
                            C = vertices3d[oidx][:3, triangle[2]]
                            
                            AB = B - A
                            BC = C - B
                            CA = A - C
                            
                            #plane: ax + by + cz + d = 0
                            plane_normal = np.cross(AB, BC)
                            d = -np.dot(plane_normal, A)
                            
                            # p = t * ray (from (0, 0, 0))
                            t = -d / np.dot(plane_normal, ray)
                            p = t * ray
                            
                            if t < 0:
                                continue
                            
                            pA = A - p
                            pB = B - p
                            pC = C - p
                            
                            cA = np.cross(AB, pA)
                            cB = np.cross(BC, pB)
                            cC = np.cross(CA, pC)
                            
                            b0 = True if np.dot(plane_normal, cA) > 0 else False
                            b1 = True if np.dot(plane_normal, cB) > 0 else False
                            b2 = True if np.dot(plane_normal, cC) > 0 else False
                            
                            if (b0 == b1) and (b1 == b2):
                                canvas[v, u] = 1
                                break
                
                # 2D Bounding Box Generation
                # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                _, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                x, y, w, h = cv2.boundingRect(contours[0])
                
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cx = (x + x + w) // 2
                cy = (y + y + h) // 2
                
                cx = cx / imw
                cy = cy / imh
                w = w / imw
                h = h / imh
                
                label['objects'][oidx]['box2d'] = {}
                box2d = label['objects'][oidx]['box2d']
                box2d['cx'] = cx
                box2d['cy'] = cy
                box2d['w'] = w
                box2d['h'] = h
                
            et = time.time()
            print(f"Time: {et - st}")
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
            cv2.imshow('image', image)
            
            # for Debug
            # key = cv2.waitKey(0)
            # if key == ord('q'):
            #     cnt = max(cnt-1, 0)
            # elif key == ord('e'):
            #     cnt = min(cnt+1, len(image_files))
            # elif key == ord('s'):
            #     cv2.imwrite(f"{cnt}.jpg", image)
            # else:
            #     pass
            
            cv2.waitKey(0)
            
            cnt = min(cnt+1, len(image_files))
            
            # json write
            with open(os.path.join(root_dir.replace('Image', 'Label'), sub_dir, label_file), 'w') as f:
                json.dump(label, f, indent=4)
            
            if cnt == len(image_files):
                break
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:/AIHUB_cits")
    args = parser.parse_args()
    
    main(args.root_dir)