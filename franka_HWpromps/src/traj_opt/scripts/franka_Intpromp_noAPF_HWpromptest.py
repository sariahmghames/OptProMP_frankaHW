#!/usr/bin/env python

# import sys
# class KineticImportsFix:
# 	def __init__(self, kinetic_dist_packages="/opt/ros/kinetic/lib/python2.7/dist-packages"):
# 		self.kinetic_dist_packages = kinetic_dist_packages

# 	def __enter__(self):
# 		sys.path.remove(self.kinetic_dist_packages)

# 	def __exit__(self, exc_type, exc_val, exc_tb):
# 		sys.path.append(self.kinetic_dist_packages)


#with KineticImportsFix():   
import os
import sys
import rospy
import numpy as np
from operator import itemgetter


from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseArray
import basis as basis
import promps as promps
import phase as phase
import tf.transformations as tf_tran
import matplotlib.pyplot as plt
import franka_kinematics
from mpl_toolkits.mplot3d import Axes3D
import Franka_pushing_clusters as franka_decluster
import formulations as fm
#import traj_opt    
from matplotlib import animation
from matplotlib.animation import FuncAnimation


franka_kin = franka_kinematics.FrankaKinematics()
franka_actions = franka_decluster.actions()

# This is the latest script in use to generate conditioned promp for franka, then the push_cost will create pushing actions
# the franka_intprompV2Opt.py has the pushing actions generated as IROS paper for Scara, pushing actions are not well generated as demos need to be collected 
# with franka for the specific picking task


class promp_pub(object):

	def __init__(self):
		rospy.init_node('franka_intprompV2_publisher', anonymous=True)
		self.rate = rospy.Rate(50)
		self.q0 = np.array([])
		self.q0 = [0.0026156825002733833, -0.7471497647631505, -0.01701382033098254, -2.641352130354944, -0.009966759979778217, 1.8291950980684883, 0.7558905248190247] # HW
		#self.q0 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]  # comment out wen running sim
		#self.q0 = [0.0, (-np.pi)/4, 0.0, (np.pi)/2, 0.0, (-np.pi)/3, 0.0]  ## works for end effector facing down
		self.q0_sim = [0.0, (-60*np.pi)/180, 0.0, (120*np.pi)/180, 0.0, (-np.pi)/3, 0.0] ## works for end effector facing down and cluster right above franka head
		self.qf = [0.0, (-70*np.pi)/180, 0.0, (160*np.pi)/180, 0.0, (np.pi)/2, 0.0]   # works well for final config desired 
		self.Tf = 0.3 # originally set to 0.3 --> chnage with task and with position of robot wrt cluster
		self.T0 = 0
		self.nDof = 7
		self.t1samples = 200
		self.t2samples = 200
		self.time_normalised1 = np.linspace(self.T0, self.Tf, self.t1samples)
		self.t_cam2 = self.time_normalised1[-1]-0.2
		self.init_cov = np.eye(len(self.q0)) * 0.000001
		self.Rgrip = 0.1
		self.Rcluster = 0.1
		self.k = 0.002
		self.Xi = 0.002
		self.pot_power = 2
		self.Q_star = 0.01 # change later
		self.stem_ndiscrete = 100
		self.constraints = {}
		self.goal = np.array([])
		self.neighbours = []
		#self.traj_opt = traj_opt.trajectory_optimization(goal = self.goal,const =self.constraints, tcam2 = self.t_cam2, tsamples = self.t1samples, tf = self.Tf)
		self.cost = []
		self.iter = []	
		self.ninter0 = 0
		self.ninter1 = 0
		self.tcam_samp = int((str(np.round(self.t_cam2*(self.t1samples)/self.Tf,0))).rstrip('0').rstrip('.') if '.' in (str(np.round(self.t_cam2*(self.t1samples)/self.Tf,0))) else (str(np.round(self.t_cam2*(self.t1samples)/self.Tf,0))))

		self.pubTrajD = rospy.Publisher('/franka_controller/command', JointTrajectory, queue_size=10)
		#self.subJState  = rospy.Subscriber("/joint_states", JointState, callback=self.RobotStateCallback)
		#self.subSPose  = rospy.Subscriber("/straws_states", PoseArray, callback=self.ObjCallback)
		self.connections_debug = False
		self.plot_connections = False
		self.animation = False
		self.plot_cost = False
		self.plot_learntTraj = True
		self.plot_cond_traj = True
		self.plot_demos = False 
		self.plot_joint_cond_promp = False



	def RobotStateCallback(self,msg):
		name = msg.name
		print('msg state=', msg)
		self.q0 = msg.position[:7]
		print('q0=', self.q0)
		self.qdot0 = msg.velocity[:7]
		if (len(self.q0) != 0):
			self.subJState.unregister()



	def ObjCallback(self,msg):
		cluster = msg.poses
		cluster_frame = msg.header.frame_id
		self.goal = np.array([cluster[0].position.x, cluster[1].position.y, cluster[2].position.z]) 
		self.neighbours = [np.array([cluster[1].position.x, cluster[1].position.y, cluster[1].position.z]), np.array([cluster[2].position.x, cluster[2].position.y, cluster[2].position.z])]
		print(self.neighbours)

		if ((cluster_frame) and len(self.neighbours) != 0):
			print('started unregistering')
			self.subSPose.unregister()



	def cluster_specifications(self,): 
		#straw81 = np.array([0.7000167482108263, 0.35000114831665186, 1.059899192900648, 4.694414635190038e-06])
		#straw82 = np.array([0.7000147967952163, 0.32817129326930183, 0.9254237487507031, 0.31141583278076324])
		#straw83 = np.array([0.700015487710529, 0.334161673495151, 0.9974340790662544, -0.2568663122771039])
		straw81 = np.array([-0.04998666399793127, 0.10000034078139142, 1.059910236708741, 4.516664479699469e-06])
		straw82 = np.array([-0.04998839523886116, 0.08264126219374765, 0.9239243388589038, 0.3136100853490663])
		straw83 = np.array([-0.04998755320876535, 0.08827520283646068, 0.9986486172601952, -0.33023169599299057])
		bunch_xyz = []
		bunch_col = []
		bunch_orient = []

		goal8c = True
		goal8_n1c = False
		goal8_n2c = False

		# orientation: rotation about y-axis of camera2 (mobile wrt arm base) , or about z-axis of camera1 (fixed wrt arm base) 
		# the following are preset orientation , call form.get_orientation when importing pcls
		config = 8
		theta8 = 0 

		goal8o = np.array([0,0,1])  # important goal orientation definition
		goal8_n1o = np.array([0,0,1]) 
		goal8_n2o = np.array([0,0,1])


		g8_n1_so = (np.pi)/20 
		g8_n2_so = -(np.pi)/20


		#bunch_xyz.append(self.goal)
		#bunch_xyz.append(self.neighbours[0]) 
		#bunch_xyz.append(self.neighbours[1])
		# override the above 3 commented lines, until you solve the PoseArray() poses.append() in state_sub.py.. cz so far we are eliminating the last appended, and replaces it with the new one, so all elements get the same height
		bunch_xyz.append(straw81[0:3])
		bunch_xyz.append(straw82[0:3]) 
		bunch_xyz.append(straw83[0:3])

		bunch_col.append(goal8c)
		bunch_col.append(goal8_n1c)
		bunch_col.append(goal8_n2c)
		bunch_orient.append(goal8o)
		bunch_orient.append(goal8_n1o)
		bunch_orient.append(goal8_n2o)
		spherical_angles = [[g8_n1_so, 0], [g8_n2_so, 0]]
		Lstems = [0.28, 0.2, 0.13]

		return bunch_xyz, bunch_col, bunch_orient, spherical_angles, Lstems, config



	def jointTrajectoryCommand(self,traj, t=1):
		jt = JointTrajectory()
		jt.header.stamp = rospy.Time.now()
		jt.header.frame_id = "panda_arm"
		jt.joint_names.append("panda_arm_joint1")
		jt.joint_names.append("panda_arm_joint2")
		jt.joint_names.append("panda_arm_joint3")
		jt.joint_names.append("panda_arm_joint4")
		jt.joint_names.append("panda_arm_joint5")
		jt.joint_names.append("panda_arm_joint6")
		jt.joint_names.append("panda_arm_joint7")
		J1 = traj[:, 0]
		J2 = traj[:, 1]
		J3 = traj[:, 2]
		J4 = traj[:, 3]
		J5 = traj[:, 4]
		J6 = traj[:, 5]
		J7 = traj[:, 6]

		n = len(J1)
		dt = np.linspace(float(t)/n, t+float(t)/n, n) #added float before t to avoid future division

		for i in range (n):
			p = JointTrajectoryPoint()
			p.positions.append(J1[i])
			p.positions.append(J2[i])
			p.positions.append(J3[i])
			p.positions.append(J4[i])
			p.positions.append(J5[i])
			p.positions.append(J6[i])
			p.positions.append(J7[i])

			p.time_from_start = rospy.Duration.from_sec(dt[i])  # time_from_start is the point in time at which that TrajectoryPoint should be executed.

			jt.points.append(p)
		self.pubTrajD.publish(jt)
		#time.sleep(1)
		del p.positions[:]
		del jt.points[:]



	def load_demos(self,): 
		# Refer to simple_example.py in folder promp/examples/python_promp/simple_example.py, for comments on functions (simple_example.py and franka_promp.py are similar scripts)
		with open('/home/sariah/franka_HWpromps/src/traj_opt/scripts/100demos.npz', 'r') as f:
		    data = np.load(f, allow_pickle=True)
		    self.Q = data['Q']#[:97]
		    self.time = data['time']#[:97] # demo 98 has less than 30 samples
		Q_row = self.Q.shape


		################################################
		# To plot demonstrated end-eff trajectories
		if self.plot_demos == True:
			fig = plt.figure()
			ax = fig.add_subplot(111, projection='3d')
			for i in range(len(self.Q)):
			    endEffTraj = franka_kin.fwd_kin_trajectory(self.Q[i])
			    ax.scatter(endEffTraj[:,0], endEffTraj[:,1], endEffTraj[:,2], c='b', marker='.')
			plt.title('EndEff')
			plt.show()

		return self.Q, self.time
		######################################


	def promp_generator(self, IC, goal):

		phaseGenerator = phase.LinearPhaseGenerator()
		# best to change design parameters (basis nb, width, and num basis outside) choice with nb of neighbours
		# First chosen combiantion was: num basis=4, width = 5, outside =1)
		basisGenerator1 = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=2, duration=self.Tf, basisBandWidthFactor=3, # check duration
		                                                   numBasisOutside=1) 


	
		proMP1 = promps.ProMP(basisGenerator1, phaseGenerator, self.nDof)
		plotDof = 2

		################################################################
		# Conditioning in JointSpace

		desiredTheta = np.array([0.5, 0.7, 0.5, 0.2, 0.6, 0.8, 0.1])
		desiredVar = np.eye(len(desiredTheta)) * 0.0001
		meanTraj, covTraj = proMP1.getMeanAndCovarianceTrajectory(self.time_normalised1)
		newProMP1 = proMP1.jointSpaceConditioning(self.Tf, desiredTheta=desiredTheta, desiredVar=desiredVar)
		traj = proMP1.getTrajectorySamples(self.time_normalised1, 1)
		print('traj =', traj.shape)

		#plt.figure()
		#plt.plot(time_normalised, trajectories[:, plotDof, :])
		#plt.xlabel('time')
		#plt.title('Joint-Space conditioning for joint 2')

		if self.plot_joint_cond_promp == True:
			proMP1.plotProMP(self.time_normalised1, [3, 4]) # refer to plotter.py, it plots means ans 2*std filled curves of newpromp, indices = [3, 4] refer to joints to plot


		##################################################
		# Conditioning in Task Space

		learnedProMP1 = promps.ProMP(basisGenerator1, phaseGenerator, self.nDof) # regression model initialization
		learner1 = promps.MAPWeightLearner(learnedProMP1) # weights learning model
		#print('q', Q[:][:120])
		learner1.learnFromData(self.Q, self.time) # get weights
		learned_promp1 = learnedProMP1.getTrajectorySamples(self.time_normalised1, 1)
		ttrajectories_learned = franka_kin.fwd_kin_trajectory(learned_promp1)
		mu_theta, sig_theta = learnedProMP1.getMeanAndCovarianceTrajectory(np.array([self.Tf])) # get mean cov of the learnt promp in joint space at time T = 1s
		#print('mu_theta=', np.squeeze(mu_theta))
		#print('sig_theta=', np.squeeze(sig_theta))
		sig_theta = np.squeeze(sig_theta)
		mu_x = goal # desired mean ee pose/position at T = 1s, old= [0.6, 0.5, 0.8]
		sig_x = np.eye(3) * 0.00000001 # desired cov of ee pose/position at T = 1s
		q_home = self.q0 #[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
		T_desired, tmp = franka_kin.fwd_kin(self.qf)
		mu_ang_euler_des = tf_tran.euler_from_matrix(T_desired, 'szyz')
		sig_euler = np.eye(3) * 0.0002
		print('starting IK')
		post_mean_tf, post_cov = franka_kin.inv_kin_ash_pose(np.squeeze(mu_theta), sig_theta, mu_x, sig_x, mu_ang_euler_des, sig_euler) # output is in joint space, desired mean and cov
		#print('post_mean =',post_mean_Ash.shape) 7x1
		#print('post_cov =',post_cov.shape) 7x7
		print('finishing IK')
		newProMP0 = learnedProMP1.jointSpaceConditioning(self.T0, desiredTheta= q_home, desiredVar=self.init_cov)
		newProMP1 = newProMP0.jointSpaceConditioning(self.Tf, desiredTheta= post_mean_tf, desiredVar=post_cov)
		trajectories_task1_conditioned = newProMP1.getTrajectorySamples(self.time_normalised1, 1)
		ttrajectories_task_conditioned = franka_kin.fwd_kin_trajectory(trajectories_task1_conditioned)

		# Get orientation of EE at T = 1s from demonstrations
		q_final = mu_theta
		q_final = np.squeeze(q_final)
		print('q_final=', q_final)
		T_final, tmp_final = franka_kin.fwd_kin(q_final)
		mu_ang_euler_final = tf_tran.euler_from_matrix(T_final, 'szyz')
		with open('promp_goal.npz', 'w') as f:
			np.save(f, trajectories_task1_conditioned)
		with open('promp_goal.csv', 'w') as f:
			#np.save(f, trajectories_task1_conditioned)
			np.savetxt(f, trajectories_task1_conditioned, delimiter=',', fmt='%f')

		return newProMP1, ttrajectories_task_conditioned, mu_ang_euler_final, sig_euler, ttrajectories_learned




	def pushing_generator(self, GoalCond_promp, Goalpromp_Sampled, cond_pts, mu_ang_euler_final, sig_euler):
		# Get the min Z in the cluster 
		Goal = Goalpromp_Sampled[-1]
		minZ = Goal[2]
		for lowest in cond_pts:
			if lowest[2] < minZ:
				minZ = lowest[2]
		print('minZ=', minZ)
		coord_s1 = []
		print('cond_pts=', cond_pts)

		## random time vector, since we didn't collected data
		#tff1 = np.linspace(0,1, 152)
		#tff1 = np.repeat(np.array([tff1]), sdemo, axis = 0)

		#tff2 = np.linspace(0,1, 10)
		#tff2 = np.repeat(np.array([tff2]), sdemo, axis = 0)

		t0 = self.time_normalised1[0]
		tf = self.time_normalised1[-1]


		cond_pts_reduced = []

		## approach 1: to check if there is topset for the goal
		for i in range(0,len(cond_pts), 2):
			cond_pts_reduced.append(cond_pts[i])

		goal = np.array([Goal[0], Goal[1], Goal[2]+0.016])
		cluster_init = np.array([Goal[0], Goal[1], Goal[2]-0.016])


		start = []
		start.append(IC0)
		start.append(cluster_init)

		mu_x_IC = start[0]  

		# Conditioning at 1st goal point (same performance if cond at tf comes at end)
		print('cond_pts_reduced=', cond_pts_reduced)
		j = []

		if len(goal_update)!= 0:
			mu_x_tf = goal_update

		elif len(j) == 0 and len(goal_update)== 0: 
			print('i didnt get goal below nb')
			mu_x_tf = cluster_init
		else:
			inex = np.where(np.sum(np.abs(np.asarray(cond_pts) - np.asarray(cond_pts_reduced)[j[-1]]), axis = -1) == 0)
			inex = inex[-1][0]
			mu_x_tf = cond_pts[inex+1]


		#######################################################################################################################################

		# Conditioning at cluster bottom point :
		mu_theta1, sig_theta1 = GoalCond_promp.getMeanAndCovarianceTrajectory(np.array([self.t_cam2])) # get mean cov of the learnt promp in joint space at time T = 1s
		sig_theta1 = np.squeeze(sig_theta1)
		mu_x_t1g1 = [cluster_init[0], cluster_init[1], (minZ-0.1)]  # desired mean ee pose/position at T = 1s
		sig_x_t1g1 = np.eye(3) * 0.0000002 # desired cov of ee pose/position at T = 1s
		post_mean_push, post_cov_push = franka_kin.inv_kin_ash_pose(np.squeeze(mu_theta1), sig_theta1, mu_x_t1g1, sig_x_t1g1, mu_ang_euler_final, sig_euler) # output is in joint space, desired mean and cov
		print('finishing IK')
		newProMP1 = GoalCond_promp.jointSpaceConditioning(self.t_cam2, desiredTheta=post_mean_push, desiredVar=post_cov_push)
		trajectories_task1_conditioned = newProMP1.getTrajectorySamples(self.time_normalised1, 1)

		#######################################################################################################################################


		## Pushing close neighbours
		mu_x_tfn = []
		if len(j) == 0 and len(goal_update)==0:
			for wpt in cond_pts:
				mu_x_tfn.append(wpt)
		elif len(j) == 0 and len(goal_update)!=0:
			for wpt in cond_pts:
				mu_x_tfn.append(wpt)
			mu_x_tfn.append(cluster_init)
		else:
			mu_x_tfn.append(cluster_init)
			cond_pts_sub = np.delete(np.asarray(cond_pts),-1, axis = 0) # to change the -1 with a systematic code
			print('cond_pts_sub=', cond_pts_sub)
			for el in cond_pts_sub:
				mu_x_tfn.append(el)

			
		t_discrete = np.linspace(self.t_cam2, tf, len(mu_x_tfn)+1,endpoint=False) # len(mu_x_tfn)
		print('t_discrete=', t_discrete)
		print('mu_x_tfn=', len(mu_x_tfn))
		t_effec = []
		mean_push = []
		cov_push = []

		traj_cond = []
		traj_cond.append(newProMP1)

		for k in range(1,len(mu_x_tfn)+1): # range(0,len(mu_x_tfn))
			## Nonlinear scale of t_discrete : exponential one
			tn = t_discrete[k]
			if (k % 2) != 0:
				tn = t_discrete[k] - t_discrete[k] * 0.0 # Tune those for trajectory variation minimization, + t_discrete[k+1] * 0.1 only add it in simulation to increase time to reach 1st cond point
			elif k==(len(mu_x_tfn)):
				tn = t_discrete[k] - t_discrete[k] * 0.0 # not much diff if 0.0
			else:
				tn = t_discrete[k] - t_discrete[k] * 0.0 # not much diff if 0.0
			t_effec.append(tn)
			print('tn=', tn)
			mu_thetan, sig_thetan = traj_cond[k-1].getMeanAndCovarianceTrajectory(np.array([tn])) # get mean cov of the learnt promp in joint space at time T = 1s
			sig_thetan = np.squeeze(sig_thetan)
			mu_x_tn = mu_x_tfn[k-1] 
			#print('mu_x=',mu_x_tng1)
			sig_x_tn = np.eye(3) * 0.0000002 # desired cov of ee pose
			post_mean_push, post_cov_push = franka_kin.inv_kin_ash_pose(np.squeeze(mu_thetan), sig_thetan, mu_x_tn, sig_x_tn, mu_ang_euler_final, sig_euler) # output is in joint space, desired mean and cov
			mean_push.append(post_mean_push)
			cov_push.append(post_cov_push)
			newProMPn = traj_cond[k-1].jointSpaceConditioning(tn, desiredTheta=post_mean_push, desiredVar=post_cov_push)
			trajectories_taskn_conditioned = newProMPn.getTrajectorySamples(self.time_normalised1, 1)

			traj_cond.append(newProMPn)

		## Save data for testing
	  	jtrajectories_pushed = trajectories_taskn_conditioned
		print('traj pushed jt=', jtrajectories_pushed.shape) 
		ttrajectories_pushed = franka_kin.fwd_kin_trajectory(np.squeeze(jtrajectories_pushed))
		print('traj pushed task=', ttrajectories_pushed.shape) 

		with open('jintprompV2_Pushing.npz', 'w') as f:
			np.save(f, jtrajectories_pushed)
		return jtrajectories_pushed, ttrajectories_pushed, cond_pts, t_effec, mu_x_t1g1, mean_push, cov_push





	def plot_connections(self, connections, cond_pts):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		for Ip in range(len(cond_pts)):
			#if counting == 0:
			pt = cond_pts[Ip]
			ax.scatter(pt[0], pt[1], pt[2], s = 50, c='g', marker='o')
			#X, Y, Z = fm.plot_3dseg(pt, sph[Ip], lstems[Ip])
			#ax.plot3D(X, Y, Z, 'green')
		#Xg, Yg, Zg = fm.plot_3dseg(xf, [0,0], lstems[-1])
		#ax.plot3D(Xg, Yg, Zg, 'green')
		for conn0 in connections:
			for conn1 in conn0:
				ax.scatter(conn1[0], conn1[1], conn1[2], s = 2, c='r', marker='o')
		        plt.pause(.001)

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		#plt.title('ProMP Conditioning at selected unripe pushable obstacles:\nCase of fruits')
		plt.show()


	def	plot_learnt_traj(self,traj, cond_pts, x0, xf):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(traj[:,0], traj[:,1], traj[:,2], c='b', marker='.')
		ax.scatter(x0[0], x0[1], x0[2], s = 100, c='y', marker='o')
		ax.scatter(xf[0], xf[1], xf[2], s = 100, c='r', marker='o')
		#counting = 0
		for Ip in range(len(cond_pts)):
			#if counting == 0:
			pt = cond_pts[Ip]
			ax.scatter(pt[0], pt[1], pt[2], s = 50, c='g', marker='o')

		# for obs in self.traj_opt.object_list:
		# 	x, y, z, alpha = fm.plot_sphere(obs[0:3], obs[3:])
		# 	sphere = ax.plot_surface(x, y, z, color='b', alpha=alpha)
	 #        plt.pause(.001)

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		#plt.title('ProMP Conditioning at selected unripe pushable obstacles:\nCase of fruits')
		plt.show()   


	def plot_traj(self, traj, cond_pts, x0, xf, sph, lstems, cam =None):

		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(traj[:,0], traj[:,1], traj[:,2], c='b', marker='.')
		ax.scatter(x0[0], x0[1], x0[2], s = 100, c='y', marker='o')
		ax.scatter(xf[0], xf[1], xf[2], s = 100, c='r', marker='o')
		if cam != None:
			ax.scatter(cam[0], cam[1], cam[2], s = 100, c='c', marker='o')
		#counting = 0
		for Ip in range(len(cond_pts)):
			#if counting == 0:
			pt = cond_pts[Ip]
			ax.scatter(pt[0], pt[1], pt[2], s = 50, c='g', marker='o')
			X, Y, Z = fm.plot_3dseg(pt, sph[Ip], lstems[Ip])
			ax.plot3D(X, Y, Z, 'green')
		Xg, Yg, Zg = fm.plot_3dseg(xf, [0,0], lstems[-1])
		ax.plot3D(Xg, Yg, Zg, 'green')
		for obs in self.traj_opt.object_list:
			x, y, z, alpha = fm.plot_sphere(obs[0:3], obs[3:])
			sphere = ax.plot_surface(x, y, z, color='b', alpha=alpha)
	        plt.pause(.001)

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		#plt.title('ProMP Conditioning at selected unripe pushable obstacles:\nCase of fruits')
		plt.show()



def promp_plotter(time_normalised, trajectories_task_conditioned):
	plt.figure()
	plt.plot(time_normalised, trajectories_task_conditioned[:, plotDof, :]) # 10 samples in joint space for joint : plotDof
	plt.xlabel('time')
	plt.title('Task-Space conditioning for joint 2')

	##############################################
	# Plot of end-effector trajectories

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for i in range(trajectories_task_conditioned.shape[2]): # shape[2] is time samples
	    endEffTraj = franka_kin.fwd_kin_trajectory(trajectories_task_conditioned[:, :, i])
	    ax.scatter(endEffTraj[:, 0], endEffTraj[:, 1], endEffTraj[:, 2], c='b', marker='.')
	plt.xlabel('X')
	plt.ylabel('Y')
	plt.title('EE trajectories after task space Conditioning')
	##############################################
	plt.show()
	print('Finished')



##################################################
def promp_saver(trajectories_task_conditioned):
	# To save the task conditioned trajectories for playing back on robot

	with open('traject_task_conditioned1.npz', 'w') as f:
	    np.save(f, trajectories_task_conditioned)


if __name__ == "__main__":
	#initialization = promp_pub()
	#while (initialization.q0.shape[0] == 0):
	#	rospy.loginfo('init state subscriber didnt return yet')
	#	time.sleep(1)
	intprompV2_generator = promp_pub()
	while len(intprompV2_generator.q0) == 0:
		rospy.loginfo('I am still waiting to get robot init state')

	#while (len(intprompV2_generator.goal) == 0 and len(intprompV2_generator.neighbours) == 0):
	# 	rospy.loginfo('I am still waiting to get the goal')

	data, time = intprompV2_generator.load_demos()
	Q = data 
	bunch_xyz, bunch_col, bunch_orient, spherical, Lstems, conf = intprompV2_generator.cluster_specifications()
	IC0, cluster_init, cond_pts, min_wpts2, subset_Ncond, goal_update = franka_actions.PushOpt_planner(bunch_xyz, bunch_col, bunch_orient, conf)
	print('goal=', cluster_init)
	GoalCond_promp, Goalpromp_Sampled, mu_ang_euler_final, sig_euler, ttrajectories_learned = intprompV2_generator.promp_generator(IC0, cluster_init)
	if intprompV2_generator.plot_learntTraj == True:
		intprompV2_generator.plot_learnt_traj(Goalpromp_Sampled, cond_pts, IC0, cluster_init)   
	#intprompV2_joint, intprompV2_task, cond_pts, tcond, x_cam, mean_jcondpts, cov_jcondpts = intprompV2_generator.pushing_generator(GoalCond_promp, Goalpromp_Sampled, min_wpts2, mu_ang_euler_final, sig_euler)


	
	#intprompV2_generator.plot_traj(intprompV2_task, cond_pts, IC0, cluster_init, spherical, Lstems, cam =x_cam)
	#intprompV2_Opt = intprompV2_generator.intprompV2_opt_sariah(intprompV2_joint, cluster_init, cond_pts, tcond, spherical, Lstems, mean_jcondpts, cov_jcondpts)
	#intprompV2_generator.jointTrajectoryCommand(intprompV2_Opt, t=100)
