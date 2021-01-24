# Robotic Welding path planning (demo)

## Overview
This script is used for Robotic Weling path planning demo on various workpiece.
It has 4 functions:     
* convert point cloud from ROS To Open3d
* find the corresponding groove
* compute the trajectroy
* mutilayer planning

The result looks this this:

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/1.gif" alt="">

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/2.gif" alt="">

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/3.gif" alt="">  


## Detail
The system is built and tesated on ROS kinetic. Much of the point cloud processing used [open3d 0.9](http://www.open3d.org/docs/0.9.0/introduction.html). 

#### 1.convert point cloud from ROS To Open3d
Adopted from [felixchenfy](https://github.com/felixchenfy/open3d_ros_pointcloud_conversion).

#### 2.find the corresponding groove
Part of the methods is adopted from our [previous work](https://github.com/romi-lab/weldingRobot-CNERC) on robotic welding.
We used a different geometry featrue (asymmetry) this time and optimised the processing.
The seam can be detected on a real-time basis.

<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/detect.gif" width="500" alt="">  

#### 3.compute the trajectroy
After getting the groove, we find the centerline of the region and then ordered it.

<img src="https://github.com/aliadnani/3D-Point-Cloud-Curve-Extraction/blob/main/assets/2020-11-10-11-16-57.png" width="400" alt=""><img src="https://github.com/aliadnani/3D-Point-Cloud-Curve-Extraction/blob/main/assets/2020-11-10-11-17-08.png" width="400" alt=""> 

The detail can be found at [3D-Point-Cloud-Curve-Extraction](https://github.com/aliadnani/3D-Point-Cloud-Curve-Extraction).

#### 4.mutilayer planning
<img src="https://github.com/romi-lab/robotic-welding-demo/blob/main/demo/mutilayer.gif" alt="">  

For detail, please check [Mutilayer weld path planning](https://github.com/romi-lab/mutilayer-weld-path-planning).

## Related links
#### Publication
P. Zhou, R. Peng, M. Xu, V. Wu and D. Navarro-Alarcon.  Path Planning with Automatic Seam Extraction over Point Cloud Models for Robotic Arc Welding, IEEE Robotics and Automation Letters (RA-L) (under review), 2020. [[pdf]](https://601318ff-63df-413d-983a-b8c13c4c1e60.filesusr.com/ugd/49b3f5_6ff63735794a4965b76fb5c6de7edc53.pdf)

Video: [Path Planning with Automatic Seam Extraction over Point Cloud Models for Robotic Arc Welding](https://vimeo.com/453908150)

#### Gesture-based control on UR
[Leap motion UR](https://github.com/Jeffery-Zhou/URLeapMotion)
