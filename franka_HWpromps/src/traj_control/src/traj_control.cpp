/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2013, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** \author Sariah Mghames. */
/** reference for goal from camera : https://github.com/pal-robotics/tiago_tutorials/blob/kinetic-devel/look_to_point/src/look_to_point.cpp */



// C++ standard headers
#include <iostream>
#include <vector>
#include <exception>
#include <string>
#include <fstream>
#include <eigen3/Eigen/Eigen>
#include <stdio.h>      
#include <stdlib.h> 
//#include <cstdlib>

// Boost headers
#include <boost/shared_ptr.hpp>


// ROS headers
#include <ros/ros.h>
#include <actionlib/client/simple_action_client.h>
#include <control_msgs/FollowJointTrajectoryAction.h>//the controller exposes this interface in the follow_joint_trajectory namespace of the controller
#include <ros/topic.h>
//#include <ros/package.h>
#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/MultiArrayDimension.h>

using namespace std ; 

// variables definition
//vector<double> Pj0, Pj1, Pj2, Pj3, Pj4, Pj5, Pj6;
const int nb_joint = 7 ;
//const int wpts = 18 ;
//int lp = 18 ;
//int lj1 = 2*lp ;
//int lj2 = 3*lp ;
double T = 2.0 ;


std::vector<Eigen::VectorXd> robot_trajectory;
std::string addr;
std::string pkg_addr;

//A function that linearly divides a given interval into a given number of
//  points and assigns them to a vector

vector<double> linspace(double a, double b, int num)
{
  // create a vector of length num
  vector<double> v(num);

  // now assign the values to the vector
  for (int i = 0; i < num; i++){
    v.push_back(a + i * ( (b - a) / num ));
     }
  return v;
}


// Our Action interface type for moving TIAGo's head, provided as a typedef for convenience
typedef actionlib::SimpleActionClient<control_msgs::FollowJointTrajectoryAction> arm_control_client;
typedef boost::shared_ptr< arm_control_client>  arm_control_client_Ptr; // arm_control_client_Ptr is a type and is a shared ptr btw elements/joints each of type arm_control_client
control_msgs::FollowJointTrajectoryGoal arm_goal ;



// Create a ROS action client to move arm
void createArmClient(arm_control_client_Ptr& actionClient)
{
  ROS_INFO("Creating action client to arm controller ...");

  actionClient.reset( new arm_control_client("effort_joint_trajectory_controller/follow_joint_trajectory") ); // action client on topic XX ... (/scara_arm/arm_traj_controller of type position_controllers/JointTrajectoryController or effort_controllers/JointTrajectoryController or velocity_controllers/JointTrajectoryController

  int iterations = 0, max_iterations = 3;
  // Wait for arm controller action server to come up
  while( !actionClient->waitForServer(ros::Duration(2.0)) && ros::ok() && iterations < max_iterations )
  {
    ROS_DEBUG("Waiting for the arm_controller_action server to come up");
    ++iterations;
  }

  if ( iterations == max_iterations )
    throw std::runtime_error("Error in createArmClient: arm controller action server not available");
}


// Generates a simple trajectory with two waypoints to move franka 
void waypoints_arm_goal(control_msgs::FollowJointTrajectoryGoal& goal)
{
  ROS_INFO("Starting to store joint names ...");
  // The joint names, which apply to all waypoints
  goal.trajectory.joint_names.push_back("panda_joint1");
  goal.trajectory.joint_names.push_back("panda_joint2");
  goal.trajectory.joint_names.push_back("panda_joint3");
  goal.trajectory.joint_names.push_back("panda_joint4");
  goal.trajectory.joint_names.push_back("panda_joint5");
  goal.trajectory.joint_names.push_back("panda_joint6");
  goal.trajectory.joint_names.push_back("panda_joint7");

  int len_traj = robot_trajectory.size() ;

  vector<double> dt = linspace(0, T, len_traj) ;
  ROS_INFO("Get time vector ...");
  for(int i=0; i<dt.size(); ++i)
    cout << dt[i] << ' ';
    

  goal.trajectory.points.resize(len_traj);
 
  // First trajectory point
  // Positions

  for (int index = 0; index < len_traj; ++index) {
    ROS_INFO("Starting loop ...");
    
    //goal.trajectory.points[index].velocities.resize(nb_joint);

    ROS_INFO("resized done ...");
    //ROS_INFO("P0 %d = %f",index,P0.data[index]);
    goal.trajectory.points[index].positions.resize(nb_joint);
  	goal.trajectory.points[index].positions[0] = robot_trajectory[index][0];
    goal.trajectory.points[index].positions[1] = robot_trajectory[index][1];
    goal.trajectory.points[index].positions[2] = robot_trajectory[index][2];
    goal.trajectory.points[index].positions[3] = robot_trajectory[index][3];
    goal.trajectory.points[index].positions[4] = robot_trajectory[index][4];
    goal.trajectory.points[index].positions[5] = robot_trajectory[index][5];
    goal.trajectory.points[index].positions[6] = robot_trajectory[index][6];

    ROS_INFO("stored pos ...%i", index);
  
    // To be reached 2 second after starting along the trajectory
    //if (index == 0)
    goal.trajectory.points[index].time_from_start = ros::Duration(dt[index]+0.1); //index/10+0.1
    //if (index >=1 and index <=6)
    //  goal.trajectory.points[index].time_from_start = ros::Duration(index+10);

}
}


void unpack_traj(){

// ------------------------------------------------------------ get desired robot trajectory
  printf("pkg dir= %s", pkg_addr.c_str()) ;
  addr = pkg_addr + "traj_opt/scripts/data/promp_goal.csv";
  std::ifstream stream(addr.c_str()); // Get C string equivalent, Returns a pointer to an array that contains a null-terminated sequence of characters
  std::string line;
  while(std::getline(stream, line)){ //getline reads characters from an input stream and places them into a string, with delimiter default \n, so gets 1 line of excel table per time
    std::istringstream s(line); // s takes a string and is of type istringstream: construct an Input stream class to operate on strings
    std::string field;
    //char* field ;
    Eigen::VectorXd pose(7);
    int i = 0;
    while (getline(s, field,',')){
      pose[i] = std::stold(field); // convert string to double: 
      i++ ;
    }

    robot_trajectory.push_back(pose); 
  }
    

  //Create an arm controller action client to move franka arm
  arm_control_client_Ptr ArmClient;
  createArmClient(ArmClient);

  waypoints_arm_goal(arm_goal);
  //Sends the command to start the given trajectory 1s from now
  arm_goal.trajectory.header.stamp = ros::Time::now() + ros::Duration(1.0);
  ArmClient->sendGoal(arm_goal);

}


// Entry point
int main(int argc, char** argv)
{
  // Init the ROS node
  ros::init(argc, argv, "joint_trajectory_action");

  ROS_INFO("Starting franka_traj_control application ...");
 
  // Precondition: Valid clock
  ros::NodeHandle nh;
  if (!ros::Time::waitForValid(ros::WallDuration(10.0))) // NOTE: Important when using simulated clock
  {
    ROS_FATAL("Timed-out waiting for valid time.");
    return EXIT_FAILURE;
  }
  ros::Rate loop_rate = ros::Rate(10000);
  //pkg_addr = "/home/sariah/franka_HWpromps/src" ;
  nh.getParam("pkg_addr", pkg_addr);
  // Define a subscriber to subscribe to array of data from python node 
  //ros::Subscriber sub0_traj = nh.subscribe("get_Joint0Traj", 1000, prompCallback0);
  unpack_traj() ;
  ros::spin() ;

  return 0 ;
}
