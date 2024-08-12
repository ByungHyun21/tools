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
    
    # suffle sub_dirs
    np.random.shuffle(sub_dirs)
    
    for sub_dir in sub_dirs:
        image_files = os.listdir(os.path.join(root_dir, sub_dir))
        
        cnt = 0
        while True:
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
                
            edges = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [1, 5], [2, 6], [3, 7]
            ]
            
            for obj in label['objects']:
                if 'box2d' in obj:
                    continue
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
                vertices_2d = intrinsic @ vertices[:3, :]
                
                vertices_2d = vertices_2d[:2, :] / vertices_2d[2, :]
                
                # 2D Bounding Box Generation from 3D Bounding Box
                imh, imw = image.shape[:2]
                
                # bool idx that projected in image
                idx = (vertices_2d[0] > 0) & (vertices_2d[0] < imw) & (vertices_2d[1] > 0) & (vertices_2d[1] < imh)
                if sum(idx) == 0:
                    continue
                
                # at least one point is in the front of the image
                zidx = vertices[2] > 0
                if sum(zidx) == 0:
                    continue
                
                # if projected in image
                xmin, ymin, xmax, ymax = imw, imh, 0, 0
                for i in range(8):
                    if idx[i]:
                        xmin = min(xmin, vertices_2d[0, i])
                        ymin = min(ymin, vertices_2d[1, i])
                        xmax = max(xmax, vertices_2d[0, i])
                        ymax = max(ymax, vertices_2d[1, i])
                
                # line: y = ax + b
                
                for edge in edges:
                    p1 = edge[0]
                    p2 = edge[1]
                    
                    if idx[p1] and not idx[p2]:
                        a = (vertices_2d[1, p2] - vertices_2d[1, p1]) / (vertices_2d[0, p2] - vertices_2d[0, p1])
                        b = vertices_2d[1, p1] - a * vertices_2d[0, p1]
                        
                        # left (x < 0)
                        if vertices[0, p2] < 0 and (vertices_2d[1, p2] > 0 and vertices_2d[1, p2] < imh):
                            x = 0
                            y = a * x + b
                            if y > 0 and y < imh:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                                
                        # right (x > 0)
                        if vertices[0, p2] > 0 and (vertices_2d[1, p2] > 0 and vertices_2d[1, p2] < imh):
                            x = imw
                            y = a * x + b
                            if y > 0 and y < imh:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                            
                        # top (y < 0)
                        if vertices[1, p2] < 0 and (vertices_2d[0, p2] > 0 and vertices_2d[0, p2] < imw):
                            y = 0
                            x = (y - b) / a
                            if x > 0 and x < imw:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                                
                        # bottom (y > 0)
                        if vertices[1, p2] > 0 and (vertices_2d[0, p2] > 0 and vertices_2d[0, p2] < imw):
                            y = imh
                            x = (y - b) / a
                            if x > 0 and x < imw:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                                
                        # left and top (x < 0, y < 0)
                        if vertices[0, p2] < 0 and vertices[1, p2] < 0 and (vertices_2d[0, p2] < 0 and vertices_2d[1, p2] < 0):
                            x1 = 0
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = 0
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                        
                        # right and top (x > 0, y < 0)
                        if vertices[0, p2] > 0 and vertices[1, p2] < 0 and (vertices_2d[0, p2] > imw and vertices_2d[1, p2] < 0):
                            x1 = imw
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = 0
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                                
                        # right and bottom (x > 0, y > 0)
                        if vertices[0, p2] > 0 and vertices[1, p2] > 0 and (vertices_2d[0, p2] > imw and vertices_2d[1, p2] > imh):
                            x1 = imw
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = imh
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                                
                        # left and bottom (x < 0, y > 0)
                        if vertices[0, p2] < 0 and vertices[1, p2] > 0 and (vertices_2d[0, p2] < 0 and vertices_2d[1, p2] > imh):
                            x1 = 0
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = imh
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                            
                    if idx[p2] and not idx[p1]:
                        a = (vertices_2d[1, p1] - vertices_2d[1, p2]) / (vertices_2d[0, p1] - vertices_2d[0, p2])
                        b = vertices_2d[1, p1] - a * vertices_2d[0, p1]
                        
                        # left (x < 0)
                        if vertices[0, p1] < 0 and (vertices_2d[1, p1] > 0 and vertices_2d[1, p1] < imh):
                            x = 0
                            y = a * x + b
                            if y > 0 and y < imh:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                                
                        # right (x > 0)
                        if vertices[0, p1] > 0 and (vertices_2d[1, p1] > 0 and vertices_2d[1, p1] < imh):
                            x = imw
                            y = a * x + b
                            if y > 0 and y < imh:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                            
                        # top (y < 0)
                        if vertices[1, p1] < 0 and (vertices_2d[0, p1] > 0 and vertices_2d[0, p1] < imw):
                            y = 0
                            x = (y - b) / a
                            if x > 0 and x < imw:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)
                                
                        # bottom (y > 0)
                        if vertices[1, p1] > 0 and (vertices_2d[0, p1] > 0 and vertices_2d[0, p1] < imw):
                            y = imh
                            x = (y - b) / a
                            if x > 0 and x < imw:
                                xmin = min(xmin, x)
                                ymin = min(ymin, y)
                                xmax = max(xmax, x)
                                ymax = max(ymax, y)   
                                
                        # left and top (x < 0, y < 0)
                        if vertices[0, p1] < 0 and vertices[1, p1] < 0 and (vertices_2d[0, p1] < 0 and vertices_2d[1, p1] < 0):
                            x1 = 0
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = 0
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                                
                        # right and top (x > 0, y < 0)
                        if vertices[0, p1] > 0 and vertices[1, p1] < 0 and (vertices_2d[0, p1] > imw and vertices_2d[1, p1] < 0):
                            x1 = imw
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = 0
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                                
                        # right and bottom (x > 0, y > 0)
                        if vertices[0, p1] > 0 and vertices[1, p1] > 0 and (vertices_2d[0, p1] > imw and vertices_2d[1, p1] > imh):
                            x1 = imw
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = imh
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                                
                        # left and bottom (x < 0, y > 0)
                        if vertices[0, p1] < 0 and vertices[1, p1] > 0 and (vertices_2d[0, p1] < 0 and vertices_2d[1, p1] > imh):
                            x1 = 0
                            y1 = a * x1 + b
                            
                            if y1 > 0 and y1 < imh:
                                xmin = min(xmin, x1)
                                ymin = min(ymin, y1)
                                xmax = max(xmax, x1)
                                ymax = max(ymax, y1)
                            
                            y2 = imh
                            x2 = (y2 - b) / a
                            
                            if x2 > 0 and x2 < imw:
                                xmin = min(xmin, x2)
                                ymin = min(ymin, y2)
                                xmax = max(xmax, x2)
                                ymax = max(ymax, y2)
                    
                    
                
                for edge in edges:
                    x1 = int(vertices_2d[0, edge[0]])
                    y1 = int(vertices_2d[1, edge[0]])
                    x2 = int(vertices_2d[0, edge[1]])
                    y2 = int(vertices_2d[1, edge[1]])
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
                
                # cv2.putText(image, objClass, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                pass
                
                    
            image = cv2.resize(image, (image.shape[1]//2, image.shape[0]//2))
            cv2.imshow('image', image)
            key = cv2.waitKey(0)
            if key == ord('q'):
                cnt = max(cnt-1, 0)
            elif key == ord('e'):
                cnt = min(cnt+1, len(image_files))
            elif key == ord('s'):
                cv2.imwrite(f"{cnt}.jpg", image)
            else:
                pass
            
            if cnt == len(image_files):
                break
            

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:/AIHUB_cits")
    args = parser.parse_args()
    
    main(args.root_dir)