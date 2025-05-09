# CELLmap
This work is the official implementation of "CELLmap: Enhancing LiDAR SLAM through Elastic and Lightweight Spherical Map Representation" accepted by ICRA 2025.


#### [[Video](https://youtu.be/u3f5F_YoFQ4?si=vhvzE56wz8Fon3w_)] [[Preprint Paper](https://arxiv.org/abs/2409.19597)]   

**Author:** [Yifan Duan](https://yjsx.top), University of Science and Technology of China, China


## 1. Prerequisites and build
+ It's similar to FLOAM, please see [FLOAM](https://github.com/wh200720041/floam) for the build details.

## 2. Usage
+ Build the workspace at first!
```python
python build_ws.py **ws_path**
```

+ Prepare the raw trajectory in tum format by any LO/LIO with a rosbag.
+ Move the trajectory to **ws_path**
```
cp raw_trajectory.txt **ws_path/trajectory/odom.txt**
```
+ Run CELLmap to refine the trajectory!
```
roslaunch cellmap plugin_backend.launch bag_file:=**rosbag_path**   ws_path:=**ws_path** topic:=**LiDAR_topic**
```
The optimized trajectory will be saved as **ws_path/trajectory/backend_odom.txt** 
## 3.Acknowledgements
Thanks for [A-LOAM](https://github.com/HKUST-Aerial-Robotics/A-LOAM) and [F-LOAM](https://github.com/wh200720041/floam).



## 4. Citation
If you use this work for your research, you may want to cite
```
@article{duan2024cellmap,
  title={CELLmap: Enhancing LiDAR SLAM through Elastic and Lightweight Spherical Map Representation},
  author={Duan, Yifan and Zhang, Xinran and Li, Yao and You, Guoliang and Chu, Xiaomeng and Ji, Jianmin and Zhang, Yanyong},
  journal={arXiv preprint arXiv:2409.19597},
  year={2024}
}
```
Additionally, the idea of **CELL** is from **TERRA** in 
```
@article{duan2024rotation,
  title={Rotation Initialization and Stepwise Refinement for Universal LiDAR Calibration},
  author={Duan, Yifan and Zhang, Xinran and You, Guoliang and Wu, Yilong and Li, Xingchen and Li, Yao and Chu, Xiaomeng and Peng, Jie and Zhang, Yu and Ji, Jianmin and others},
  journal={arXiv preprint arXiv:2405.05589},
  year={2024}
}
```
