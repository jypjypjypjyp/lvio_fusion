# lvio_fusion

## Introduction

A Self-adaptive Multi-sensor Fusion SLAM Framework Using Actor-critic Method. In 1.0, we built a common framework, and we will focus on the complex environment of vehicles in 2.0. 

## Dependencies

* ros (Kinetic/Melodic/Noetic)
* Eigen3
* Sophus
* opencv
* pcl
* ceres-solver
* libgeographic-dev

## Usage

Complie:
``` bash
catkin_make
```

Run:
``` bash
source devel/setup.bash
roslaunch lvio_fusion_node kitti.launch
```

## Result

kitti:
![](misc/kitti-result.png)
![](misc/lvio1.png)

kaist urban:
![](misc/lvio39.png)

