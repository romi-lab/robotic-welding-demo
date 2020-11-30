# Robotic Weling path planning (demo)

## Overview
This script is used for Robotic Weling path planning demo on various workpiece.
It has 4 functions:     
* convert point cloud from ROS To Open3d
* find the corresponding groove
* compute the trajectroy
* mutilayer planning

The result looks this this:

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/1.gif" width="800" alt="">

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/2.gif" width="800" alt="">

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/3.gif" width="800" alt="">  


## Detail
The system is built and tesated on ROS kinetic. Much of the point cloud processing used [open3d 0.9](http://www.open3d.org/docs/0.9.0/introduction.html). 

### 1.convert point cloud from ROS To Open3d
Adopted from [felixchenfy](https://github.com/felixchenfy/open3d_ros_pointcloud_conversion).

### 2.find the corresponding groove
Part of the methods is adopted from our [previous work](https://github.com/romi-lab/weldingRobot-CNERC) on robotic welding.
We used a different geometry featrue (asymmetry) this time and optimised the processing.
The seam can be detected on a real-time basis.

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/detect.gif" width="600" alt="">  

### 3.compute the trajectroy
After getting the groove, we find the centerline of the region and then ordered it.
The detail of can be found at [3D-Point-Cloud-Curve-Extraction](https://github.com/aliadnani/3D-Point-Cloud-Curve-Extraction).

### 4.mutilayer planning
For detail, please check [Mutilayer weld path planning](https://github.com/romi-lab/mutilayer-weld-path-planning).
