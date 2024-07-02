import os
import cv2

def main(dataset_path):
    train_path = os.path.join(dataset_path, 'Train')
    valid_path = os.path.join(dataset_path, 'Valid')
    test_path = os.path.join(dataset_path, 'Test')
    
    if os.path.exists(train_path):
        sensors = os.listdir(train_path)
        for sensor in sensors:
            sensor_path = os.path.join(train_path, sensor)
            analysis(sensor_path)
            
    if os.path.exists(valid_path):
        sensors = os.listdir(valid_path)
        for sensor in sensors:
            sensor_path = os.path.join(valid_path, sensor)
            analysis(sensor_path)
            
    if os.path.exists(test_path):
        sensors = os.listdir(test_path)
        for sensor in sensors:
            sensor_path = os.path.join(test_path, sensor)
            analysis(sensor_path)
            
def analysis(root_path):
    # list of analysis
    # 1. number of files
    # 2. number of objects
    # 3. number of classes
    # 4. image size
    # 5. number of images
    
    pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='D:/AIHUB_cits', help='dataset path')
    args = parser.parse_args()
    
    main(args.dataset)