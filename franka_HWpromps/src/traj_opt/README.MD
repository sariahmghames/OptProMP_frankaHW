Requirements:
1. MoveIt!  https://github.com/LCAS/CLAS_wiki/wiki/Moveit
2. Franka ROS  https://github.com/LCAS/CLAS_wiki/wiki/Franka



* launch traj_opt demo_promp.launch (franka in moveit)

* in catkin_ws
python src/TrajOpt/promp_trajopt/traj_opt/src/trajectory_optimize.py  
(for CHOMP + ProMP)

&
 
python src/TrajOpt/promp_trajopt/traj_opt/src/franka_promp.py  
(for ProMP)


* Update the path to the demos recorded .npz folder in the franka_promp.py script
