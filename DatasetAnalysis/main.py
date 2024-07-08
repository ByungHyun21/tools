import os
import cv2
import json
from tqdm import tqdm

def main(dataset_path):
    # train_path = os.path.join(dataset_path, 'Train')
    # if os.path.exists(train_path):
    #     sensors = os.listdir(train_path)
    #     for sensor in sensors:
    #         sensor_path = os.path.join(train_path, sensor)
    #         analysis(sensor_path)
            
    valid_path = os.path.join(dataset_path, 'Valid')        
    if os.path.exists(valid_path):
        sensors = os.listdir(valid_path)
        for sensor in sensors:
            sensor_path = os.path.join(valid_path, sensor)
            analysis(sensor_path)
    
    # test_path = os.path.join(dataset_path, 'Test')        
    # if os.path.exists(test_path):
    #     sensors = os.listdir(test_path)
    #     for sensor in sensors:
    #         sensor_path = os.path.join(test_path, sensor)
    #         analysis(sensor_path)
            
def analysis(root_path):
    # list of analysis
    # 1. number of files
    # 2. number of objects
    # 3. number of classes
    # 4. image size
    # 5. number of images
    
    # if 'Image' in os.listdir(root_path):
    #     analysed_data = analysis_image(os.path.join(root_path, 'Image'))
    #     print_analysis(analysed_data)
    if 'LiDAR' in os.listdir(root_path):
        analysed_data = analysis_lidar(os.path.join(root_path, 'LiDAR'))
        print_analysis(analysed_data)
    if 'Label' in os.listdir(root_path):
        analysed_data = analysis_label(os.path.join(root_path, 'Label'))
        print_analysis(analysed_data)
    
    pass

def print_analysis(data):
    if data is None:
        return
    # print dict bueautifully
    for key in data:
        print(key)
        if type(data[key]) == dict:
            for k in data[key]:
                print('  ', k, ':', data[key][k])
        else:
            print('  ', data[key])
        print()
        

def analysis_image(image_path):
    print('Analysis Image: ', image_path)
    # 1. image size
    # 2. number of images
    # 3. image format
    
    size = {}
    num_images = 0
    image_format = {}
    
    sub_dirs = os.listdir(image_path)
    for sub_dir in tqdm(sub_dirs):
        files = os.listdir(os.path.join(image_path, sub_dir))
        for file in files:
            image = cv2.imread(os.path.join(image_path, sub_dir, file))
            
            if image is None:
                continue
            
            # number of images
            num_images += 1
            
            # image size
            imh, imw = image.shape[:2]
            size_key = str(imw) + 'x' + str(imh)
            if size_key not in size:
                size[size_key] = 1
            else:
                size[size_key] += 1
                
            # image format
            image_format_key = file.split('.')[-1]
            if image_format_key not in image_format:
                image_format[image_format_key] = 1
            else:
                image_format[image_format_key] += 1

    output = {
        'size': size, 
        'num_images': num_images, 
        'image_format': image_format
        }

    return output
    
def analysis_lidar(lidar_path):
    print('Analysis LiDAR: ', lidar_path)
    
    pass
    
def analysis_label(label_path):
    print('Analysis Label: ', label_path)
    # 1. number of files
    # 2. number of objects
    # 3. number of classes
    # 4. image size
    # 5. number of images
    
    classes = []
    box2d = {}
    box3d = {}
    box3d_width = {}
    box3d_height = {}
    box3d_length = {}
    zdepth = {}
    
    sub_dirs = os.listdir(label_path)
    for sub_dir in tqdm(sub_dirs):
        files = os.listdir(os.path.join(label_path, sub_dir))
        for file in files:
            with open(os.path.join(label_path, sub_dir, file), 'r') as f:
                label = json.load(f)
                
            for obj in label['objects']:
                if 'box2d' in obj:
                    key = obj['class']
                    if key not in classes:
                        classes.append(key)
                    if key not in box2d:
                        box2d[key] = 1
                    else:
                        box2d[key] += 1
                        
                if 'box3d' in obj:
                    key = obj['class']
                    if key not in classes:
                        classes.append(key)
                    if key not in box3d:
                        box3d[key] = 1
                    else:
                        box3d[key] += 1
                        
                    if key not in zdepth:
                        zdepth[key] = []
                    zdepth[key].append(obj['box3d']['translation']['z'])
                    
                    if key not in box3d_width:
                        box3d_width[key] = []
                    box3d_width[key].append(obj['box3d']['size']['width'])
                    
                    if key not in box3d_height:
                        box3d_height[key] = []
                    box3d_height[key].append(obj['box3d']['size']['height'])
                    
                    if key not in box3d_length:
                        box3d_length[key] = []
                    box3d_length[key].append(obj['box3d']['size']['length'])
                    
    # zdepth mu, sigma
    zdepth_mu = {}
    zdepth_sigma = {}
    for key in zdepth:
        zdepth_mu[key] = sum(zdepth[key]) / len(zdepth[key])
        zdepth_sigma[key] = (sum([(z - zdepth_mu[key])**2 for z in zdepth[key]]) / len(zdepth[key]))**0.5
        
    # box3d size mu, sigma
    width_mu = {}
    width_sigma = {}
    for key in box3d_width:
        width_mu[key] = sum(box3d_width[key]) / len(box3d_width[key])
        width_sigma[key] = (sum([(w - width_mu[key])**2 for w in box3d_width[key]]) / len(box3d_width[key]))**0.5
        
    height_mu = {}
    height_sigma = {}
    for key in box3d_height:
        height_mu[key] = sum(box3d_height[key]) / len(box3d_height[key])
        height_sigma[key] = (sum([(h - height_mu[key])**2 for h in box3d_height[key]]) / len(box3d_height[key]))**0.5
        
    length_mu = {}
    length_sigma = {}
    for key in box3d_length:
        length_mu[key] = sum(box3d_length[key]) / len(box3d_length[key])
        length_sigma[key] = (sum([(l - length_mu[key])**2 for l in box3d_length[key]]) / len(box3d_length[key]))**0.5
                
    output = {
        'classes': classes,
        'box2d': box2d,
        'box3d': box3d,
        'zdepth_mu': zdepth_mu,
        'zdepth_sigma': zdepth_sigma,
        'box3d_width_mu': width_mu,
        'box3d_width_sigma': width_sigma,
        'box3d_height_mu': height_mu,
        'box3d_height_sigma': height_sigma,
        'box3d_length_mu': length_mu,
        'box3d_length_sigma': length_sigma
        }
    
    return output

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='D:/KITTI_3D_Detection', help='dataset path')
    args = parser.parse_args()
    
    main(args.dataset)