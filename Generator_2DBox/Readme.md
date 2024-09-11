# Generator_2DBox

## Usage

Generate 2D Box from 3D Box

main.py -> Using projected 8 points
main2.py -> Using 3D Box Rasterization

```python
python main.py --root_dir "path/to/dataset"
```

```
root_dir
├── Train
├── Valid
```

## Result (main.py)

![image1](image1.jpg)

Z < 0 and projected other side of the image

![image2](image2.jpg)

![image4](image4.jpg)

![image5](image5.jpg)

## Result (main2.py)

![image1](image1_main2.jpg)