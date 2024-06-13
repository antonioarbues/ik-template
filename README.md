# Inverse Kinematics for leader-follower arms in ROS2
ROS2 template for inverse kinematics for leader-follower teleoperation using [Pinocchio](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/).

Inverse kinematics allows to have leader and follower arms with different kinematic structures - [example here](https://x.com/arbwes/status/1799498155219734620).

## Setup
Copy-paste this package in your ros workspace `src/` folder, modify it with your system interfaces 
and build with `colcon build`.

## Run
```
ros2 run ik ik.py
```