# KITTI

"Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite"

https://www.cvlibs.net/datasets/kitti/

# Environment
python 3.10.14

opencv

numpy

pyntcloud

open3d

# convert_kitti.py and convert_kitti_subset.py

source_root_dir is ""

```
Full 
├── 
```

convert KITTI to M-Fast/L-Fast Form

```python
python convert_kitti.py --source_root_dir "path/to/dataset" --output_root_dir "path/to/output"
```

convert KITTI_Subset to M-Fast/L-Fast Form

```python
python convert_kitti_subset.py --source_root_dir "path/to/dataset" --output_root_dir "path/to/output"
```