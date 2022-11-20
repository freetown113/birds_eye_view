# Bird's eye view

This project is about to get *segmentation mask* of one particular class - **Road** in this case. And transform *frontal view* of this object taken from the camera into *topdown (bird's eye)* view.<br>

**Input:** Original image taken from the camera (3 channels image). Camera's parameters(intrinsics, distortion coefficients).<br>
**Output:** Segmentation mask (1 channel image) perspective transformed to topdown view.


### About
> 
> - *Segmentation algorithm* was taken from https://github.com/open-mmlab/mmsegmentation 
> - There are 3 different perspective transformation algorithms in the project:
>  1. `projector_initial.py` contains mix of opencv functions and manual operations with numpy arrays.
>  2. `projector_opencv.py` contains transformation method based purely on opencv functions.
>  3. `projector_alternative.py` contains manual perspective transformation method.
> - Interfaces of all methods are the same, so any of them can be use interchangebly. One just need to chnage import module in line 4 of `main.py` file.  

### Prepare
>
> - It's recommended to build a container based on the **Dockerfile** from the project

### Install
```
git clone https://github.com/freetown113/birds_eye_view.git
cd birds_eye_view
bash setup.sh
```

### Launch
```
bash launch.sh
```

### Result
>
> - After the completion of the programm execution process the `output.jpg` image will be saved to the `result` directory.
