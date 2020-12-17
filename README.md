# lvio_fusion

## Introduction

## Dependencies

* ros (Kinetic/Melodic/Noetic)
* Eigen3
* Sophus
* opencv
* pcl
* ceres-solver
* DBoW3
* libgeographic-dev

## Usage

Complie:
``` bash
catkin_make -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
```

Run:
``` bash
source devel/setup.bash
roslaunch lvio_fusion_node kitti.launch
```

## Method


## Result

kitti:
![](./misc/kitti-result.png)

