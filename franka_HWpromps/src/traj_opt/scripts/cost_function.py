#import rospy
import sys
#import moveit_msgs.msg
#import moveit_commander
from geometry_msgs.msg import Pose, PoseStamped
from franka_kinematics import FrankaKinematics
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np


class CostFunction(object):
    def __init__(self):
        super(CostFunction, self).__init__()
        #moveit_commander.roscpp_initialize(sys.argv)
        self.collision_threshold = 0.07
        #self.robot_state = moveit_commander.RobotCommander()
        #self.scene = moveit_commander.PlanningSceneInterface()
        self.franka_k = FrankaKinematics()

        def show_objects(self):
            for k, v in self.scene.get_objects().items():
                print("Pos:", v.primitive_poses[0].position.x, v.primitive_poses[0].position.y,\
                    v.primitive_poses[0].position.z)
                print("Ori:", v.primitive_poses[0].orientation.x, v.primitive_poses[0].orientation.y,\
                    v.primitive_poses[0].orientation.z, v.primitive_poses[0].orientation.w)
                print("Size:", v.primitives[0].dimensions)

    def setup_planner(self):
        self.group = moveit_commander.MoveGroupCommander("panda_arm")
        self.group.set_end_effector_link("panda_hand")    # planning wrt to panda_hand or link8
        self.group.set_max_velocity_scaling_factor(0.05)  # scaling down velocity
        self.group.set_max_acceleration_scaling_factor(0.05)  # scaling down velocity
        self.group.allow_replanning(True)
        self.group.set_num_planning_attempts(5)
        self.group.set_goal_position_tolerance(0.0002)
        self.group.set_goal_orientation_tolerance(0.01)
        self.group.set_planning_time(5)
        self.group.set_planner_id("FMTkConfigDefault")

    def get_robot_discretised_points(self, joint_values=None, step_size=0.2, with_joint_index=False):

        if joint_values is None:
            joint_values = list(self.robot_state.get_current_state().joint_state.position)[:7]

        _, t_joints = self.franka_k.fwd_kin(joint_values) # t_joints is transformation matrix up to eac joint
        fwd_k_j_positions = np.vstack(([0.,0.,0.], t_joints[:, :3, 3])) # it stacks the translation vector of each trasnformation matrix
        discretised = list()
        j_index = list()
        for j in range(len(fwd_k_j_positions) - 1):
            w = fwd_k_j_positions[j+1] - fwd_k_j_positions[j]
            if len(w[w != 0.]):
                step = step_size * w / np.linalg.norm(w) # scaled normalized distance btw  eac successive 2 joints
                n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
                discretised.extend(np.outer(np.arange(1, n), step) + fwd_k_j_positions[j]) # extend te list of discretized body points wit discretized btw oin from 1 joint to anoter
            j_index.append(len(discretised))
        return (np.array(discretised), j_index) if with_joint_index else np.array(discretised)


    def get_object_collision_spheres(self):
        return [(v.primitive_poses[0].position.x, v.primitive_poses[0].position.y, v.primitive_poses[0].position.z,\
                    v.primitives[0].dimensions[0]) for v in self.scene.get_objects().values()]


    def compute_minimum_distance_to_objects(self, robot_body_points, object_list=None, ):
        if object_list is None:
            object_list = self.get_object_collision_spheres()
        D = list()
        if len(robot_body_points.shape) == 1:
            robot_body_points = robot_body_points.reshape(1, 3) # x, y , z vert stacked

        for r in robot_body_points:  # expects robot_body_points as an array of dimension 1 x n
            dist = []
            for o in object_list:
                ro = r - o[:3]
                norm_ro = np.linalg.norm(ro)
                dist.append(norm_ro - o[3]) # S: 0.15m is a safety factor, - 0.15
            D.append(np.min(np.array(dist)))
        return D

    def cost_potential(self, D): # D is distance form body points to obstacles, Eq (21) in CHOMP paper
        c = list()
        for d in D:
            if d < 0.:
                c.append(-d + 0.5 * self.collision_threshold)
            elif d <= self.collision_threshold:
                c.append((0.5 * (d-self.collision_threshold)**2) / self.collision_threshold)
            else:
                c.append(0)
        return c


    def calculate_normalised_workspace_velocity(self, trajectory):
        # We have not divided by  time as this has been indexed and is thus not available
        position_vectors = np.array([self.get_robot_discretised_points(joint_values) for joint_values in trajectory])
        velocity = np.gradient(position_vectors, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=2)
        vel_normalised = np.divide(velocity, vel_magnitude[:, :, None], out=np.zeros_like(velocity), where=vel_magnitude[:, :, None] != 0)
        return vel_normalised, vel_magnitude, velocity


    def calculate_jacobian(self, robot_body_points, joint_index, joint_values=None):
        class ParentMap(object):
            def __init__(self, num_joints):
                self.joint_idxs = [i for i in range(num_joints)]
                self.p_map = np.zeros((num_joints, num_joints))
                for i in range(num_joints): # child frame ind or joint ind
                    for j in range(num_joints): # parent frame ind or joint ind
                        if j <= i:
                            self.p_map[i][j] = True

            def is_parent(self, parent, child):
                if child not in self.joint_idxs or parent not in self.joint_idxs:
                    return False
                return self.p_map[child][parent]

        parent_map = ParentMap(7)

        if joint_values is None:
            joint_values = list(self.robot_state.get_current_state().joint_state.position)[:7]

        _, t_joints = self.franka_k.fwd_kin(joint_values)
        joint_axis = t_joints[:, :3, 2] # 1st dim is a transformation mat for each joint in the world frame, 2 is the z-axis for that joint in the world frame
        joint_positions = t_joints[:7, :3, 3]  # Excluding the fixed joint at the end
        jacobian = list()
        for i, points in enumerate(np.split(robot_body_points, joint_index)[:-1]): # i for index and points is for values (values of body points up to each joint + pts on link after it)
            # print i, len(points)
            if i == 0:
                for point in points:
                    jacobian.append(np.zeros((3, 7)))
            else:
                for point in points:
                    jacobian.append(np.zeros((3, 7))) # 3 associated to position in 3D , 7 for 7 joints
                    for joint in parent_map.joint_idxs:
                        if parent_map.is_parent(joint, i-1): # parent shld be less than child, i -1 to account for all the body points up to link before the joint 
                            # find cross product
                            col = np.cross(joint_axis[joint, :], point-joint_positions[joint])
                            jacobian[-1][0][joint] = col[0] #replaces the zeros in the latest (-1) appended matrix
                            jacobian[-1][1][joint] = col[1]
                            jacobian[-1][2][joint] = col[2]
                        else:
                            jacobian[-1][0][joint] = 0.0
                            jacobian[-1][1][joint] = 0.0
                            jacobian[-1][2][joint] = 0.0
        return np.array(jacobian)


    def calculate_curvature(self, trajectory):
        vel_normalised, vel_magnitude, velocity = self.calculate_normalised_workspace_velocity(trajectory)
        temp = (np.eye(len(vel_normalised)) - vel_normalised.dot(vel_normalised.T))
        curvature = np.dot(temp, np.gradient(velocity))/vel_magnitude**2
        return curvature


def main():
   #with open('/home/ash/Ash/Repo/MPlib/src/ProMPlib/traject_task_conditioned_very_good.npz', 'r') as f:
   with open('/home/sariah/ws_moveit/src/TrajOpt/promp_trajopt/traj_opt/src/traject_task_conditioned1.npz', 'r') as f:  
   #with open('/home/sariah/ws_moveit/src/TrajOpt/promp_trajopt/traj_opt/src/100demos.npz', 'r') as f:
        trajectories = np.load(f)[:, :, 0:5]

   cf = CostFunction()  # create an object of the class CostFunction, dont pass arguments since the init function doesnt have args

   for ti in range(trajectories.shape[2]):
        vel_normalised, vel_magnitude, velocity = cf.calculate_normalised_workspace_velocity(trajectories[:, :, ti])
        c = cf.gradient_cost_potential(trajectories[:, :, ti])
        obstacle_cost = np.sum(c * vel_magnitude)
        jacobian = cf.calculate_jacobian()
        # curvature = cf.calculate_curvature(trajectories[:, :, ti])


if __name__ == '__main__':
    rospy.init_node('someName')
    try:
        main()
    except Exception as e:
        print(e)
    finally:
        moveit_commander.roscpp_shutdown()
        moveit_commander.os._exit(0)
