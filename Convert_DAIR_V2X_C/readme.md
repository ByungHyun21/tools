# DAIR-V2X-C

"DAIR-V2X: A Large-Scale Dataset for Vehicle-Infrastructure Cooperative 3D Object Detection"

https://air.tsinghua.edu.cn/DAIR-V2X/english/cheluduan.html

# Environment
python 3.10.14

opencv

numpy

pyntcloud

open3d

# convert_Infra.py and convert_Vehicle.py

source_root_dir is "Full Dataset (train&val)"

```
Full Dataset (train&val)
├── cooperative-vehicle-infrastructure
│   ├── cooperative-vehicle-infrastructure
│   │   ├── cooperative
│   │   ├── infrastructure-side
│   │   ├── vehicle-side
├── cooperative-vehicle-infrastructure-infrastructure-side-image
│   ├── cooperative-vehicle-infrastructure-infrastructure-side-image
│   │   ├── 000009.jpg
│   │   ├── ...
│   │   ├── 019955.jpg
├── cooperative-vehicle-infrastructure-infrastructure-side-velodyne
│   ├── cooperative-vehicle-infrastructure-infrastructure-side-velodyne
│   │   ├── 000009.pcd
│   │   ├── ...
│   │   ├── 019955.pcd
├── cooperative-vehicle-infrastructure-vehicle-side-image
│   ├── cooperative-vehicle-infrastructure-vehicle-side-image
│   │   ├── 000000.jpg
│   │   ├── ...
│   │   ├── 020513.jpg
├── cooperative-vehicle-infrastructure-vehicle-side-velodyne
│   ├── cooperative-vehicle-infrastructure-vehicle-side-velodyne
│   │   ├── 000000.pcd
│   │   ├── ...
│   │   ├── 020514.pcd
```

convert Infrasturcture data to M-Fast/L-Fast Form

```python
python convert_infra.py --source_root_dir "path/to/dataset" --output_root_dir "path/to/output"
```

convert Vehicle data to M-Fast/L-Fast Form

```python
python convert_vehicle.py --source_root_dir "path/to/dataset" --output_root_dir "path/to/output"
```