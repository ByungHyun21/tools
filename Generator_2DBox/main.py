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
                
            for obj in label['objects']:
                if 'box2d' in obj:
                    continue
                if 'box3d' not in obj:
                    continue
                
                box3d = obj['box3d']
                
                    
            
            np.save(os.path.join(root_dir, image_file.split('.')[0]), image)
            os.remove(os.path.join(root_dir, image_file))
        
        pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="D:/AIHUB_cits")
    args = parser.parse_args()
    
    main(args.root_dir)