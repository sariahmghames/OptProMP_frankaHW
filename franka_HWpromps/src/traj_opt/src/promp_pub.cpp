// ROS
#include <ros/ros.h>
// ROS service
#include <chrono>
#include <fstream>
#include <random>

// OTC service message converter fcns
//#include <otc_services/otc_services_msg.h>

using namespace std;



int main(int argc, char** argv){
	ros::init(argc, argv, "promp_pub");
	ros::NodeHandle n;

 	//ros::ServiceClient srvCA = n.serviceClient<otc_services::collisionAvoidanceObjective>("compute_collision_avoidance");

  	// [INFO] Chose the frequency of the loop
 	ros::Rate loop_rate = ros::Rate(1000000);
	// Grasping Configurations
	std::vector <std::array<double, 7> > GCs; 
	std::vector<Eigen::VectorXd> robot_trajectory;
	std::string addr;
	std::string pkg_addr;

	n.param("pkg_addr", pkg_addr, std::string());

// ------------------------------------------------------------ get Object's Trajectory
	{
		addr = pkg_addr + "../traj_opt/src/data/promp_pub.csv";
		std::ifstream stream(addr.c_str());
		std::string line;
		while(std::getline(stream, line)){
			std::istringstream s(line);
			std::string field;
			Eigen::VectorXd pose(7);
			int i = 0;
			while (getline(s, field,',')){
				pose[i] = std::stod(field);
				i++;
			}

			robot_trajectory.push_back(pose);
		}
	}