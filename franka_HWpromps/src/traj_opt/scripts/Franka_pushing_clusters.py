import numpy as np
import phase as phase
import basis as basis
import promps as promps
import tf.transformations as tf_tran
import matplotlib.pyplot as plt
import franka_kinematics
import clustering
import formulations as form
#import request
from mpl_toolkits.mplot3d import Axes3D
from numbers import Number
from matplotlib.lines import Line2D
from scipy.interpolate import Rbf
import scipy
from operator import itemgetter
#from sympy.geometry import Point, Line
import math
from itertools import cycle



class actions():

	def __init__(self, ):
		self.franka_kin = franka_kinematics.FrankaKinematics()
		self.pi = math.pi
		self.tabletop_height = 0.4
		self.rmax_straw = 0.02
		self.Dg_open_max = 0.04

		self.shiftX_Plink1_plate = 0
		self.shiftY_Plink1_plate = 0.0377
		self.shiftZ_Plink1_plate = 0
		self.shiftX_plate_R1link2 = 0.0347
		self.shiftY_plate_R1link2 = 0.067
		self.shiftZ_plate_R1link2 = 0
		self.Zgripper = 0.1016 + 0.01
		self.Push_above = False
		self.Push_plane = False
		self.q00 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
		self.q0_sim = [0.0, (-60*np.pi)/180, 0.0, (120*np.pi)/180, 0.0, (-np.pi)/3, 0.0]
		self.q0 = [0.0026156825002733833, -0.7471497647631505, -0.01701382033098254, -2.641352130354944, -0.009966759979778217, 1.8291950980684883, 0.7558905248190247] 


	def push_promp(self, IC0, init, cond_pts, Ncond_pts, conf, goal_update, push_above, push_plane):
		Xee = list()
		Yee = list()
		Zee = list()

		# Get the min Z in the cluster 
		minZ = init[2]
		for lowest in cond_pts:
			if lowest[2] < minZ:
				minZ = lowest[2]

		coord_s1 = []
		print('State promp with IC=', IC0)
		print('State promp with goal=', init)

		# cd to promp folder
		with open('./PRR_demos_new.npz', 'r') as f:
		    Q = np.load(f) # Q shape is (121, 162, 7), 3D array, i = 121 is demo length, 162 is samples length and 7 is dof length
		    Q = Q['data']
		    #print(Q[0])
		    print('Q length:',len(Q))


		print('Q=', Q.shape)
		sdemo = Q.shape[0]
		#sdemo = np.squeeze(sdemo)
		ssamples = Q.shape[1]
		print('ssamples', ssamples)
		sdof = Q.shape[2]
		print('sdof', sdof)


		## random time vector, since we didn't collected data
		tff1 = np.linspace(0,1, 152)
		tff1 = np.repeat(np.array([tff1]), sdemo, axis = 0)

		tff2 = np.linspace(0,1, 10)
		tff2 = np.repeat(np.array([tff2]), sdemo, axis = 0)

		Xj_s1 = np.zeros((sdemo, ssamples))
		Yj_s1 = np.zeros((sdemo, ssamples))
		Zj_s1 = np.zeros((sdemo, ssamples))
		Xj_s2 = np.zeros((sdemo, ssamples))
		Yj_s2 = np.zeros((sdemo, ssamples))
		Zj_s2 = np.zeros((sdemo, ssamples))

		# store not valid solutions
		nValid_sol1 = []
		nValid_sol2 = []


		# Get joint trajectories
		joint_data = np.transpose(np.array([np.transpose(Xj_s1), np.transpose(Yj_s1), np.transpose(Zj_s1)])) # 121 x 162 x 3



		################################################################
		phaseGenerator = phase.LinearPhaseGenerator() # generates z = z_dot *time, a constructor of the class LinearPhaseGenerator
		basisGenerator1 = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=4, duration=1, basisBandWidthFactor=1,
		                                                   numBasisOutside=1)  # passing arguments of the __init__() method, best nb of basis: 4, 5

		basisGenerator2 = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=6, duration=1, basisBandWidthFactor=1, 
		                                                   numBasisOutside=1)  # smaller numBasis --> less variation in the promp, change numBasis depending on nb of pushable fruits

		t0 = 0.0
		tf = 1.0
		# To* design a systematic approach for the time vector to condition at
		if len(cond_pts)/2 >= 2:
			t_cam2 = 0.5/(len(cond_pts)/2)
		elif (len(cond_pts)/2) < 2 and (len(cond_pts)/2) != 0:
			t_cam2 = 0.5/(len(cond_pts)/2)
		else:
			t_cam2 = 0.98
		print('t_cam2=', t_cam2)
		time_normalised1 = np.linspace(0, t_cam2, (2/2)*100)  # 1sec duration 
		time_normalised2 = np.linspace(t_cam2, tf, (2/2)*100)  # 1sec duration 
		#time_normalised1 = np.linspace(0, t_cam2, (len(cond_pts)/2)*100)  # 1sec duration 
		#time_normalised2 = np.linspace(t_cam2, tf, (len(cond_pts)/2)*100)  # 1sec duration 
		nDof = 1
		plotDof = 1
		cond_pts_reduced = []

		## approach 1: to check if there is topset for the goal
		for i in range(0,len(cond_pts), 2):
			cond_pts_reduced.append(cond_pts[i])

		goal = np.array([init[0], init[1], init[2]+0.016])
		cluster_init = np.array([init[0], init[1], init[2]-0.016])

		print('clusterinit=', cluster_init)


		start = []
		start.append(IC0)
		start.append(cluster_init)

		##################################################
		# Learnt promp in Task Space 

		learnedProMP1 = promps.ProMP(basisGenerator1, phaseGenerator, nDof)
		learnedProMP2 = promps.ProMP(basisGenerator2, phaseGenerator, nDof)
		learner1 = promps.MAPWeightLearner(learnedProMP1)  
		learner2 = promps.MAPWeightLearner(learnedProMP2)  

		learntTraj1Xee = learner1.learnFromXDataTaskSapce(Q[:,0:152,0]/1000, tff1)
		traj1_Xee = learnedProMP1.getTrajectorySamplesX(time_normalised1, 1) # get samples from the ProMP in the joint space 
		learntTraj1Yee = learner1.learnFromYDataTaskSapce(Q[:,0:152,1]/1000, tff1)
		traj1_Yee = learnedProMP1.getTrajectorySamplesY(time_normalised1, 1) 
		learntTraj1Zee = learner1.learnFromZDataTaskSapce(Q[:,0:152,2]/1000, tff1)
		traj1_Zee = learnedProMP1.getTrajectorySamplesZ(time_normalised1, 1)


		learntTraj2Xee = learner2.learnFromXDataTaskSapce(Q[:,152:162,0]/1000, tff2)
		traj2_Xee = learnedProMP2.getTrajectorySamplesX(time_normalised2, 1) # get samples from the ProMP in the joint space 
		learntTraj2Yee = learner2.learnFromYDataTaskSapce(Q[:,152:162,1]/1000, tff2)
		traj2_Yee = learnedProMP1.getTrajectorySamplesY(time_normalised2, 1) 
		learntTraj2Zee = learner2.learnFromZDataTaskSapce(Q[:,152:162,2]/1000, tff2)
		traj2_Zee = learnedProMP1.getTrajectorySamplesZ(time_normalised2, 1)


		mu_x_IC = start[0]  

		#####################################################################################################################################
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(traj1_Xee, traj1_Yee, traj1_Zee, c='b', marker='.')
		ax.scatter(mu_x_IC[0], mu_x_IC[1], mu_x_IC[2], s = 100, c='y', marker='o')
		ax.scatter(traj1_Xee[-1], traj1_Yee[-1], traj1_Zee[-1], s = 100, c='r', marker='o')
		ax.scatter(traj2_Xee, traj2_Yee, traj2_Zee, s = 100, c='b', marker='.')
		ax.scatter(traj2_Xee[-1], traj2_Yee[-1], traj2_Zee[-1], s = 100, c='r', marker='o')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.title('Task space learnt promp1')
		plt.show()


		# Conditioning at 1st goal point (same performance if cond at tf comes at end)
		print('cond_pts_reduced=', cond_pts_reduced)
		j = []
		if self.push_above == True:
			for j1, val in enumerate(cond_pts_reduced):
				if val[2] > goal[2]:
					print('i got goal below nb')
					j.append(j1)
		elif len(goal_update)!= 0:
			mu_x_tf = goal_update

		elif len(j) == 0 and len(goal_update)== 0: 
			print('i didnt get goal below nb')
			mu_x_tf = cluster_init
		else:
			inex = np.where(np.sum(np.abs(np.asarray(cond_pts) - np.asarray(cond_pts_reduced)[j[-1]]), axis = -1) == 0)
		 	inex = inex[-1][0]
		 	mu_x_tf = cond_pts[inex+1]
		sig_x_tf = np.eye(3) * 0.0
		print('i got mu_x_tf=', mu_x_tf)
		print('i got goal update=', goal_update)
		traj_conditioned_tf = learnedProMP2.taskSpaceConditioning_Sariah(tf, mu_x_tf, sig_x_tf)
		traj_Xee_condT = traj_conditioned_tf.getTrajectorySamplesX(time_normalised2, 1)
		traj_Yee_condT = traj_conditioned_tf.getTrajectorySamplesY(time_normalised2, 1)
		traj_Zee_condT = traj_conditioned_tf.getTrajectorySamplesZ(time_normalised2, 1)

		#### cond at t0
		mu_x_t0 = start[0] 
		sig_x_t0 = np.eye(3) * 0.0
		traj_conditioned_t0 = learnedProMP1.taskSpaceConditioning_Sariah(t0, mu_x_t0, sig_x_t0)
		traj_Xee_condt0 = traj_conditioned_t0.getTrajectorySamplesX(time_normalised1, 1)
		traj_Yee_condt0 = traj_conditioned_t0.getTrajectorySamplesY(time_normalised1, 1)
		traj_Zee_condt0 = traj_conditioned_t0.getTrajectorySamplesZ(time_normalised1, 1)


		#######################################################################################################################################

		# Conditioning at cluster bottom point :

		mu_x_t1g1 = np.array([cluster_init[0], cluster_init[1]-0.015, (minZ-0.05)]) # in y= ... -0.015 , 0.015 is radius of strawberry , included for simulation reasons , to enter the cluster with pushing NN and not with direct cross to avid NN sliding into the gripper, return it to -0.1 after simulation
		print('mu_x_t1g1=', mu_x_t1g1)
		sig_x_t1g1 = np.eye(3) * 0.0
		traj_conditioned_t1g1 = traj_conditioned_t0.taskSpaceConditioning_Sariah(t_cam2, mu_x_t1g1, sig_x_t1g1)
		traj_Xee_condt1g1 = traj_conditioned_t1g1.getTrajectorySamplesX(time_normalised1, 1)
		traj_Yee_condt1g1 = traj_conditioned_t1g1.getTrajectorySamplesY(time_normalised1, 1)
		traj_Zee_condt1g1 = traj_conditioned_t1g1.getTrajectorySamplesZ(time_normalised1, 1)

		##################


		traj_conditioned_t12g1 = traj_conditioned_tf.taskSpaceConditioning_Sariah(t_cam2, mu_x_t1g1, sig_x_t1g1)
		traj_Xee_condt12g1 = traj_conditioned_t12g1.getTrajectorySamplesX(time_normalised2, 1)
		traj_Yee_condt12g1 = traj_conditioned_t12g1.getTrajectorySamplesY(time_normalised2, 1)
		traj_Zee_condt12g1 = traj_conditioned_t12g1.getTrajectorySamplesZ(time_normalised2, 1)


		###################

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

			
		t_discrete = np.linspace(t_cam2, tf, len(mu_x_tfn)+1 ,endpoint=False)
		print('t_discrete=', t_discrete)
		#mu_x_tfn = sorted(mu_x_tfn, key=itemgetter(2)) # sorted along y axis of cam (not needed since cond_pts are already sorted)
		#print('mu_x_tfn=', mu_x_tfn)
		traj_Xee_condtn = []
		traj_Yee_condtn = []
		traj_Zee_condtn = []
		traj_cond = []
		traj_cond.append(traj_conditioned_t12g1)

		for k in range(0,len(mu_x_tfn)):
			mu_x_tng1 = mu_x_tfn[k] 
			#print('mu_x=',mu_x_tng1)
			sig_x_tng1 = np.eye(3) * 0.0
			## Nonlinear scale of t_discrete : exponential one
			if k == 0:
				tng1 = t_discrete[k+1] + t_discrete[k+1] * 0.1 # Tune those for trajectory variation minimization, + t_discrete[k+1] * 0.1 only add it in simulation to increase time to reach 1st cond point
			else:
				tng1 = t_discrete[k+1] - t_discrete[k+1] * 0.04
			## Solution 2: time discretised with exponential function
			#tng1 = t_discrete[0] + (tf - t_discrete[0]) / np.exp((len(mu_x_tfn)-(k*1))-(tf - t_discrete[0])**((len(mu_x_tfn)-(k*1))/10)) 
			print('tng1=', tng1)
			traj_conditioned_tng1 = traj_cond[k].taskSpaceConditioning_Sariah(tng1, mu_x_tng1, sig_x_tng1)
			traj_Xee_condtng1 = traj_conditioned_tng1.getTrajectorySamplesX(time_normalised2, 1)
			traj_Yee_condtng1 = traj_conditioned_tng1.getTrajectorySamplesY(time_normalised2, 1)
			traj_Zee_condtng1 = traj_conditioned_tng1.getTrajectorySamplesZ(time_normalised2, 1)
			if (len(j) != 0 and k ==0): # save conditioned traj for only the real goal, in case there is wpts above the goal
				normal_goalTraj = np.array([np.concatenate((traj_Xee_condt1g1 ,traj_Xee_condtng1), axis = None), np.concatenate((traj_Yee_condt1g1 ,traj_Yee_condtng1), axis = None), np.concatenate((traj_Zee_condt1g1 ,traj_Zee_condtng1), axis = None)])
				with open('GRASPberry_Config{}_NormalTraj.npz'.format(conf), 'w') as f:
					np.save(f, normal_goalTraj)

			traj_cond.append(traj_conditioned_tng1)
			traj_Xee_condtn.append(traj_Xee_condtng1)
			traj_Yee_condtn.append(traj_Yee_condtng1)
			traj_Zee_condtn.append(traj_Zee_condtng1)

		## Save data for testing
	  	trajectories_Pushed = np.array([np.concatenate((traj_Xee_condt1g1 ,traj_Xee_condtng1), axis = None), np.concatenate((traj_Yee_condt1g1 ,traj_Yee_condtng1), axis = None), np.concatenate((traj_Zee_condt1g1 ,traj_Zee_condtng1), axis = None)])
	  	print('traj pushed=', trajectories_Pushed.shape) #3x400
	  	#print('first learnt promp shape=', traj_Xee_condt1g1.shape)
	  	#print('second learnt promp shape=', traj_Xee_condtng1.shape)
	  	with open('GRASPberry_Config{}_Pushing.npz'.format(conf), 'w') as f:
			np.save(f, trajectories_Pushed)


		##################
		loop = 0
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(traj_Xee_condtn[-1], traj_Yee_condtn[-1], traj_Zee_condtn[-1], c='b', marker='.')
		if (len(Ncond_pts) != 0): # plot subpts that are not conditioned (by optimization)
			for i in range(len(Ncond_pts)):
				Ncond_pt = Ncond_pts[i]
				print('Ncond_pt =', Ncond_pt )
				ax.scatter(Ncond_pt[0], Ncond_pt[1], Ncond_pt[2], s = 100, c='r', marker='o')
				ax.scatter(np.repeat(Ncond_pt[0], 80), np.repeat(Ncond_pt[1], 80), np.linspace(Ncond_pt[2], Ncond_pt[2]+0.2, 80), c='g', marker='.')
		for l in range(0, len(mu_x_tfn)):
			mu_x_tng1 = mu_x_tfn[l]
			print('mu_x_tng1=',mu_x_tng1)
			if len(j) == 0 and len(goal_update)==0:
				if (loop % 2 != 0):
					ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='g', marker='o')
				else:
					ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='r', marker='o')
				##### STEMS DRAWINGS
				#ST_g1n
				if (loop % 2 == 0):
					print('loop is even')
					ax.scatter(np.repeat(mu_x_tng1[0], 80), np.repeat(mu_x_tng1[1], 80), np.linspace(mu_x_tng1[2], mu_x_tng1[2]+0.2, 80), c='g', marker='.')
				else:
					print('loop is odd')
					mu_x_tng1_prev = mu_x_tfn[l-1]
					root = np.array([mu_x_tng1_prev[0], mu_x_tng1_prev[1], mu_x_tng1_prev[2]+0.2])
					update_pose = np.array([mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2]])
					ax.scatter(np.linspace(update_pose[0], root[0], 30), np.linspace(update_pose[1], root[1], 30), np.linspace(update_pose[2], root[2], 30), c='g', marker='.') # uncomment to not plot the stem of teh updated pose
				loop = loop + 1
			elif len(j) == 0 and len(goal_update)!=0:
				if (loop % 2 == 0):
					ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='r', marker='o')
				else:
					ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='g', marker='o')
				##### STEMS DRAWINGS 
				#ST_g1n
				if (loop % 2 != 0):
					print('loop is even')
					#mu_x_tng1_prev = mu_x_tfn[l-1]
					#root = np.array([mu_x_tng1_prev[0], mu_x_tng1_prev[1], mu_x_tng1_prev[2]+0.2])
					#update_pose = np.array([mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2]])
					#ax.scatter(np.repeat(mu_x_tng1[0], 50), np.repeat(mu_x_tng1[1], 50), np.linspace(mu_x_tng1[2], mu_x_tng1[2]+0.2, 50), c='g', marker='.')
				else:
					print('loop is odd')
					ax.scatter(np.repeat(mu_x_tng1[0], 80), np.repeat(mu_x_tng1[1], 80), np.linspace(mu_x_tng1[2], mu_x_tng1[2]+0.2, 80), c='g', marker='.')
				loop = loop + 1
				ax.scatter(np.linspace(mu_x_tf[0], cluster_init[0], 80), np.linspace(mu_x_tf[1], cluster_init[1], 80), np.linspace(mu_x_tf[2], cluster_init[2]+0.2, 80), c='g', marker='.')
			else:
				# first is the goal (done out of loop)
				if l == 0:
					#ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='g', marker='o')
					loop = loop + 1
				else:
					if (loop % 2 == 0):
						ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='g', marker='o')
					else:
						ax.scatter(mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2], s = 100, c='r', marker='o')
					##### STEMS DRAWINGS
					#ST_g1n
					if (loop % 2 != 0):
						print('loop is odd')
						ax.scatter(np.repeat(mu_x_tng1[0], 80), np.repeat(mu_x_tng1[1], 80), np.linspace(mu_x_tng1[2], mu_x_tng1[2]+0.2, 80), c='g', marker='.')
					else:
						print('loop is even')
						mu_x_tng1_prev = mu_x_tfn[l-1]
						root = np.array([mu_x_tng1_prev[0], mu_x_tng1_prev[1], mu_x_tng1_prev[2]+0.2])
						update_pose = np.array([mu_x_tng1[0], mu_x_tng1[1], mu_x_tng1[2]])
						ax.scatter(np.linspace(update_pose[0], root[0], 80), np.linspace(update_pose[1], root[1], 80), np.linspace(update_pose[2], root[2], 80), c='g', marker='.')
					loop = loop + 1
				ax.scatter(np.linspace(mu_x_tf[0], mu_x_tfn[-1][0], 80), np.linspace(mu_x_tf[1], mu_x_tfn[-1][1], 80), np.linspace(mu_x_tf[2], mu_x_tfn[-1][2]+0.2, 80), c='g', marker='.')

		# ST_g1
		ax.scatter(traj_Xee_condt1g1, traj_Yee_condt1g1, traj_Zee_condt1g1, c='b', marker='.')
		ax.scatter(np.repeat(cluster_init[0], 80), np.repeat(cluster_init[1], 80), np.linspace(cluster_init[2], cluster_init[2]+0.2, 80), c='g', marker='.')
		##
		if len(goal_update)==0:
			ax.scatter(mu_x_tf[0], mu_x_tf[1], mu_x_tf[2], s = 100, c='r', marker='o')
		else:
			ax.scatter(mu_x_tf[0], mu_x_tf[1], mu_x_tf[2], s = 100, c='g', marker='o')
		ax.scatter(mu_x_t1g1[0], mu_x_t1g1[1], mu_x_t1g1[2], s = 100, c='c', marker='o')
		ax.scatter(mu_x_t0[0], mu_x_t0[1], mu_x_t0[2], s = 100, c='y', marker='o')
		ax.scatter(cluster_init[0], cluster_init[1], cluster_init[2]+0.016, s = 100, c='r', marker='o')  # we conditioned below th goal by 0.016 but we want to plot the goal, so we reput 0.016
		ax.text(mu_x_t0[0]-0.015, mu_x_t0[1], mu_x_t0[2]+0.03, 'IC', style='italic', weight = 'bold')
		ax.text(cluster_init[0]-0.015, cluster_init[1], cluster_init[2]+0.25, 'Goal', style='italic', weight = 'bold')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		#plt.title('Task space cond @ goal with pushing @ NN')
		plt.show()


	#####################################################################################################################################################


	## Caging Planner: generates modified primitives
	def PushOpt_planner(self, bunch_xyz, bunch_col, bunch_orient, conf):

		cluster_init = []
		cluster = bunch_xyz
		cluster_col = bunch_col
		cluster_orient = bunch_orient
		Lstem = []
		dtheta = []
		root = []
		cluster_col_ord = []
		cluster_orient_ord = []
		cluster_Lstem_ord = []
		cluster_dtheta_ord = []
		cluster_root_ord = []

		for p in range(0,len(cluster)):  
			theta0 = np.arccos(np.array([0,0,1]).dot(cluster_orient[p])) # projection on zaxis of camera2
			L , inter = form.get_StemLength(cluster[p], cluster_orient[p], self.tabletop_height, theta0)
			Lstem.append(L)
			theta = np.arcsin((self.Dg_open_max/2) / L)
			delta_theta0 = theta - theta0
			dtheta.append(delta_theta0)
			root.append(inter)

		# dict
		collection = {'points':cluster, 'color':cluster_col, 'orient': cluster_orient, 'Lstem':Lstem, 'dtheta': dtheta, 'root': root}
		#cluster_ord = cluster.sort(key= lambda x: x[0], reverse = False) # or key=itemgetter(0)
		ord_ind, cluster_ord = zip(*sorted(enumerate(cluster), key=lambda x: x[1][0])) # enumerate(a) returns an enumeration over tuples consisting of the indexes and values in the original list: ---  x[1][0]: item 0 of item 1 of each tuple, a tuple = (index, value) of the list
		print('cluster=', cluster)
		print('cluster ord=', cluster_ord)
		print('ind=', ord_ind)
		for index in ord_ind:
			cluster_col_ord.append(cluster_col[index]) 
			cluster_orient_ord.append(cluster_orient[index])
			cluster_Lstem_ord.append(Lstem[index])
			cluster_dtheta_ord.append(dtheta[index])
			cluster_root_ord.append(root[index])
		collection_ord = {'points': cluster_ord, 'color': cluster_col_ord, 'orient': cluster_orient_ord, 'Lstem': cluster_Lstem_ord, 'dtheta': cluster_dtheta_ord, 'root': cluster_root_ord}
		

		## Start picking goals from the bunch of pt clouds
		for pts in collection_ord['points']:
			#print('pts=', pts)
			ii = np.where(np.sum(np.abs(np.asarray(collection_ord['points'])-pts), axis = -1) == 0)
			ii = ii[0][0]
			#ii = collection_ord['points'].index(pts)
			print('ii=', ii)
			print('collection_ord=', collection_ord['color'][ii])
			if (collection_ord['color'][ii] == True):
				cluster_init = pts
				if (ii == 0 or all(collection_ord['color'][j] == False for j in range(0,ii-1))):
					IC0 = self.franka_kin.fwd_kin(self.q0)
				else:
					for j in range(ii-1,0):
						if (collection_ord['color'][j] == True):
							IC0 = collection_ord['points'][j]


				## override IC0 for simulation reasons only

				T0 , _ = self.franka_kin.fwd_kin(self.q0)
				IC0 = T0[0:3, 3]

				# find nearest neighboured fruits to cluster_init
				ind = np.where(np.sum(np.abs(np.asarray(collection_ord['points'])-cluster_init), axis = -1) == 0)
				ind = ind[0][0]
				print('new ind=', ind)
				lis = list(collection_ord['points']) # collection_ord['points']is a tuple --> (array([]), array([]),..)
				del lis[ind]
				goal_NN = clustering.rNN(lis,cluster_init, r = 0.2) # to add to paper for clustering pt clouds, 0.2 only for sim reasons -- keep it then 0.1

				# from the set of fruits in one cluster select the ones that need to be pushed
				cluster_subset_down = []

				for pt_in_cluster in goal_NN:
					if pt_in_cluster[2] < cluster_init[2]:
						cluster_subset_down.append(pt_in_cluster)

				print('cluster subset down=', cluster_subset_down)
				#D_plane = ((Dg_open_max/2 + rmax_straw) *1.1)
				D_plane = ((self.Dg_open_max/2) *1.5)
				cluster_subset1_down = []

				# below is the approach to select all fruits that needs to be pushed
				for sb1 in cluster_subset_down:
					if np.dot(np.array([1,0,0]),np.array([sb1[0]-cluster_init[0] , sb1[1]-cluster_init[1] , sb1[2]-cluster_init[2]])) <= D_plane:  # 1.5 safety factor to account for straw dimension uncertainty, projection of d on X-axis of camera
						cluster_subset1_down.append(sb1)


				## Needed to define the order of conditioning for the promp
				# ordering the cluster neighrest neighbors from the closer to arm gripper in its original path
				# below is the optimization to select the min nb of fruits that needs to be pushed
				min_wpts2 = []

				print('cluster_subset1_down=', cluster_subset1_down)
				if len(cluster_subset1_down)!= 0:
					ind2, wpts2 = zip(*sorted(enumerate(cluster_subset1_down), key=lambda x: x[1][2])) 
					min_wpts2.append(wpts2[0])
					for j, val in enumerate(wpts2[1:]):
						for j1, val1 in enumerate(min_wpts2):
							if val[2] == val1[2]: ## to adjust for .any()
								print('wpts with same level')
								print('remove an unnecessary wpt from the subset bottom')
								di = math.sqrt((cluster_init[0]-val1[0])**2 + (cluster_init[1]-val1[1])**2 + (cluster_init[2]-val1[2])**2)
								dii = math.sqrt((cluster_init[0]-val[0])**2 + (cluster_init[1]-val[1])**2 + (cluster_init[2]-val[2])**2)
								if di < dii:
									print('keep the appended wpt')
								else:
									min_wpts2 = np.delete(min_wpts2, j1, axis = 0) # to adjust
									min_wpts2 = list(min_wpts2)
									min_wpts2.append(val)
							else:
								min_wpts2.append(val)
				else:
					wpts2 = cluster_subset1_down
				
				pushed_wpts2 = []

				cond_pts = []		

				len_wpts2 = len(min_wpts2)

				if (len_wpts2 != 0):
					last_nb2 = min_wpts2[-1]
				else:
					last_nb2 = 0

				## Get the orientation of soft fruits 
				print('Getting direction of neighbored soft fruits')
				i2 = 0

				shift_downset = []
				subset_Ncond = []
				nextelem2 = []

				if (len(min_wpts2) > 1):
					cycle2 = cycle(min_wpts2[1:])

				if len_wpts2 != 0:
					for downset in min_wpts2: # push away from goal fruit (start_glob)
						# fruit orientation in camera1 frame
						if (len(min_wpts2) > 1):
							nextelem2 = next(cycle2)
						ind = np.where(np.sum(np.abs(np.asarray(collection['points'])-downset), axis = -1) == 0)
						ind = ind[0][0]
						v1 = collection['orient'][ind]
						# draw an arc path of the NN fruit, get the orthogonal directions to v2 (2 possible)
						v1 = form.normalize(v1)
						print('v1=', v1)
						tg1 = form.perpendicular(v1)
						print('tg1=', tg1)
						tg1 = form.normalize(tg1)
						tg2 = np.cross(v1, tg1)
						print('tg2=', tg2)

						## Get the right perpendicular direction to stem : solution 2
						if np.sum(np.abs(v1 - np.array([0, 0, 1])), axis = -1) == 0:
							tg3 = np.array([0, 1, 0])
							tg4 = np.array([1, 0, 0])
						else:
							tg3 = np.cross(v1, np.array([0, 0, 1])) # along gravity and passing thru root
							print('tg3=', tg3)
							tg4 = np.cross(v1, tg3)
							print('tg4=', tg4)
						# line l1 with direction tg3 and passing thru wpts2 COM and of magnitude 0.02 (2cm) 
						P3 = downset + (0.01*tg3)
						#l1 = Line(downset, P3)
						len_l1 = math.sqrt((P3[0]-downset[0])**2 + (P3[1]-downset[1])**2 + (P3[2]-downset[2])**2)
						vl1 = np.array([P3[0]-downset[0], P3[1]-downset[1], P3[2]-downset[2]])
						# Intersection of l1 with (P) : I1
						# distance from the I1 to root of stem (relative): d1
						# line l2 with direction tg4 and passing thru wpts2 COM and of magnitude 0.02 (2cm) 
						P4 = downset + 0.01*tg4
						#l2 = Line(downset, P4)
						len_l2 = math.sqrt((P4[0]-downset[0])**2 + (P4[1]-downset[1])**2 + (P4[2]-downset[2])**2)
						vl2 = np.array([P4[0]-downset[0], P4[1]-downset[1], P4[2]-downset[2]])
						# get the direction that is pointing toward the next wpts2
						print('downset=', downset)
						print('last_nb2=', last_nb2)
						if ((downset != last_nb2).any() and (nextelem2[2]<=downset[2])): 
						# if not the last one and if the next one is at same or smaller hight than it , then push normal to tabletop.. ifnot: if next one is at higher level .. then push it parallel to tabletop
							print('I am here')
							dij = math.sqrt((min_wpts2[i2+1][0]-downset[0])**2 + (min_wpts2[i2+1][1]-downset[1])**2 + (min_wpts2[i2+1][2]-downset[2])**2)
							vdij = 	np.array([min_wpts2[i2+1][0]-downset[0], min_wpts2[i2+1][1]-downset[1],  min_wpts2[i2+1][2]-downset[2]])

							proj_l1 = vdij.dot(vl1)
							proj_l2 = vdij.dot(vl2)
							if proj_l1 < proj_l2:
								vf = tg3 
							else:
								vf = tg4

							beta = self.pi/2 - ((pi - collection['dtheta'][ind]) / 2)
							F0Ft = np.sqrt((collection['Lstem'][ind])**2 + (collection['Lstem'][ind])**2 - 2*(collection['Lstem'][ind]*collection['Lstem'][ind]*np.cos(collection['dtheta'][ind])))
							proj_F0Ft_ontg = np.cos(beta) * F0Ft
							vect_wpt2P4 = vf * proj_F0Ft_ontg 
							vect_wpt2P3 = vect_wpt2P4 / np.cos(beta) 
							P3x = downset[0] + vect_wpt2P3[0]
							P3y = downset[1] + vect_wpt2P3[1]
							P3z = downset[2] + vect_wpt2P3[2]
							P3 = np.array([P3x, P3y, P3z])

							if (downset[0] - cluster_init[0] < 0):
								shift_downset.append(np.array([downset[0]-D_plane/2, downset[1], downset[2]]))
							elif (downset[0] - cluster_init[0] > 0):
								shift_downset.append(np.array([downset[0]+D_plane/2, downset[1], downset[2]]))

						elif ((downset != last_nb2).any() and (nextelem2[2] > downset[2])): # push parallel to tabletop but away from the next one 
							dij = math.sqrt((min_wpts2[i2+1][0]-downset[0])**2 + (min_wpts2[i2+1][1]-downset[1])**2 + (min_wpts2[i2+1][2]-downset[2])**2)
							vdij = 	np.array([min_wpts2[i2+1][0]-downset[0], min_wpts2[i2+1][1]-downset[1],  min_wpts2[i2+1][2]-downset[2]])

							proj_l1 = vdij.dot(vl1)
							proj_l2 = vdij.dot(vl2)
							if proj_l1 < proj_l2:
								vf = tg3 
							else:
								vf = tg4

							beta = self.pi/2 - ((self.pi - collection['dtheta'][ind]) / 2)
							F0Ft = np.sqrt((collection['Lstem'][ind])**2 + (collection['Lstem'][ind])**2 - 2*(collection['Lstem'][ind]*collection['Lstem'][ind]*np.cos(collection['dtheta'][ind])))
							proj_F0Ft_ontg = np.cos(beta) * F0Ft
							vect_wpt2P4 = vf * proj_F0Ft_ontg 
							vect_wpt2P3 = vect_wpt2P4 / np.cos(beta) 
							P3x = downset[0] + vect_wpt2P3[0]
							P3y = downset[1] + vect_wpt2P3[1]
							P3z = downset[2] + vect_wpt2P3[2]
							P3 = np.array([P3x, P3y, P3z])

							if (downset[0] - cluster_init[0] < 0):
								shift_downset.append(np.array([downset[0]-D_plane/2, downset[1], downset[2]]))
							elif (downset[0] - cluster_init[0] > 0):
								shift_downset.append(np.array([downset[0]+D_plane/2, downset[1], downset[2]]))

						else: # last one in the subset --> away from goal parallel to tabletop
							print('I am there')

							# Check the direction of the wpt2 
							i_wpt2 = np.where(np.sum(np.abs(np.asarray(collection_ord['points'])-downset), axis = -1) == 0)
							i_wpt2 = i_wpt2[0][0]
							wpt2_orient = collection_ord['orient'][i_wpt2]
							k = np.array([0, 0, 1])
							wpt2_update = np.array([])
							if np.sum(np.abs(wpt2_orient - k), axis = -1) != 0: # update later to only min theta needed to create opening fo rthe gripper
								theta_wpt2 = np.arccos(k[0]* wpt2_orient[0]+ k[1]* wpt2_orient[1] + k[2]* wpt2_orient[2])
								deltax2 = np.sin(theta_wpt2) * collection_ord['Lstem'][i_wpt2]
								Xwpt2_update = downset[0] + deltax2  # pay attention if + or -
								print('Xwpt2=', Xwpt2_update)
								root_wpt2 = collection_ord['root'][i_wpt2] - collection_ord['Lstem'][i_wpt2]
								Zwpt2_update = root_wpt2[2]
								print('Zwpt2=', Zwpt2_update)
								P3 = np.array([Xwpt2_update, downset[1], Zwpt2_update])

								if (downset[0] - cluster_init[0] < 0):
									shift_downset.append(np.array([downset[0]-D_plane/2, downset[1], downset[2]]))
								elif (downset[0] - cluster_init[0] > 0):
									shift_downset.append(np.array([downset[0]+D_plane/2, downset[1], downset[2]]))

							else:
								dij = math.sqrt((downset[0]-cluster_init[0])**2 + (downset[1]-cluster_init[1])**2 + (downset[2]-cluster_init[2])**2)
								vdij = 	np.array([downset[0]-cluster_init[0], downset[1]-cluster_init[1],  downset[2]-cluster_init[2]])
								proj_l1 = vdij.dot(vl1)
								proj_l2 = vdij.dot(vl2)
								if proj_l1 > proj_l2:
									vf = tg4 
								else:
									vf = tg3
								if form.angle_between(vdij, vf) == 0:
								 	vf = vf
								else:
								 	vf = -vf				

								print('dtheta=', collection['dtheta'][ind])
								print('Lstem=', collection['Lstem'][ind])
								beta = self.pi/2 - ((self.pi - collection['dtheta'][ind]) / 2)
								print('beta=', beta)
								F0Ft = np.sqrt((collection['Lstem'][ind])**2 + (collection['Lstem'][ind])**2 - 2*(collection['Lstem'][ind]*collection['Lstem'][ind]*np.cos(collection['dtheta'][ind])))
								print('F0Ft=', F0Ft)
								proj_F0Ft_ontg = np.cos(beta) * F0Ft
								print('proj=', proj_F0Ft_ontg)
								print('vf=', vf)
								vect_wpt2P4 = vf * proj_F0Ft_ontg 
								vect_wpt2P3 = vect_wpt2P4 / np.cos(beta) 
								print('vect_wpt2P3=', vect_wpt2P3)
								P3x = downset[0] + vect_wpt2P3[0]
								P3y = downset[1] + vect_wpt2P3[1]
								P3z = downset[2] + vect_wpt2P3[2]
								P3 = np.array([P3x, P3y, P3z])
								print('P3=', P3)

								if (downset[0] - cluster_init[0] < 0):
									shift_downset.append(np.array([downset[0]-D_plane/2, downset[1], downset[2]]))
								elif (downset[0] - cluster_init[0] > 0):
									shift_downset.append(np.array([downset[0]+D_plane/2, downset[1], downset[2]]))

						pushed_wpts2.append(downset) 
						pushed_wpts2.append(P3)  
						i2 = i2 + 1
					for dp in pushed_wpts2:
						cond_pts.append(dp)
					if (len(cluster_subset1_down) > 1):
						print('len cluster_subset1_down=', len(cluster_subset1_down))
						for a2 in range(len(pushed_wpts2)):
							print('len(pushed_wpts2)=', len(pushed_wpts2))
							i22 = np.where(np.sum(np.abs(np.asarray(cluster_subset1_down)-pushed_wpts2[a2]), axis = -1) == 0)
							print('i22=', i22)
							i22 = i22[0]
							print('cluster_subset1_down=', np.asarray(cluster_subset1_down))
							cluster_subset1_down = np.asarray(cluster_subset1_down)
							cluster_subset1_down =  np.delete(cluster_subset1_down, i22, axis = 0) # non conditioned pts from each subset
							print('cluster_subset1_down=', cluster_subset1_down )
					else:
						for a2 in range(len(pushed_wpts2)):
							if (np.sum(np.abs(np.asarray(cluster_subset1_down)-pushed_wpts2[a2]), axis = -1) == 0):
								cluster_subset1_down = np.asarray(cluster_subset1_down)
								cluster_subset1_down =  np.delete(cluster_subset1_down, 0, axis = 0) # non conditioned pts from each subset
					for nc in cluster_subset1_down:
						subset_Ncond.append(nc)



				# Check the direction of the goal to push it or not
				i_gl = np.where(np.sum(np.abs(np.asarray(collection_ord['points'])-cluster_init), axis = -1) == 0)
				i_gl = i_gl[0][0]
				goal_orient = collection_ord['orient'][i_gl]
				print('goal orient =', goal_orient)
				reorient_goal = False
				k = np.array([0, 0, 1])
				goal_update = np.array([])
				if np.sum(np.abs(goal_orient - np.array([0, 0, 1])), axis = -1) != 0:
					reorient_goal = True
					theta_goal = np.arccos(k[0]* goal_orient[0]+ k[1]* goal_orient[1] + k[2]* goal_orient[2])
					deltax = np.sin(theta_goal) * collection_ord['Lstem'][i_gl]
					xg_update = cluster_init[0] - deltax
					print('xg=', xg_update)
					root_gl = collection_ord['root'][i_gl] - collection_ord['Lstem'][i_gl]
					zg_update = root_gl[2]
					print('zg=', zg_update)
					print('cluster init 1=', cluster_init[1])
					goal_update = np.array([xg_update, cluster_init[1], zg_update])


				print('Start conditioning while pushing a goal reduced neighborhood')
				## Save data for testing
				saved_wpts = np.array([cond_pts, subset_Ncond, shift_downset])
				#print('saved_wpts=', saved_wpts)
				#print('saved_wpts shape=', saved_wpts.shape)
				#with open('GRASPberry_Config{}_PushingWpts.npz'.format(conf), 'w') as f:
				#	np.save(f, saved_wpts)
				return IC0, cluster_init, cond_pts, min_wpts2, subset_Ncond, goal_update


if __name__ == '__main__':
	bunch_xyz = []
	bunch_col = []
	bunch_orient = []

	# Collect point clouds
	X_goal7 = np.array([0, 0.5, 0.46-0.13])
	X_goal8 = np.array([0, 0.5, 0.46-0.13]) 
	X_goal9 = np.array([-0.08+0.2*np.sin((np.pi)/10), 0.5, 0.46-0.2*np.cos((np.pi)/10)])   

	# neighbors of X_goal1
	goal7_n1 = np.array([0.1-0.23*np.sin((np.pi)/6), X_goal7[1], 0.46-0.23*np.cos((np.pi)/6)])  	

	# neighbors of X_goal2
	goal8_n1 = np.array([0.12-0.28*np.sin((np.pi)/6), X_goal8[1], 0.46-0.28*np.cos((np.pi)/6)]) 
	goal8_n2 = np.array([-0.1+0.2*np.sin((np.pi)/8), X_goal8[1], 0.46-0.2*np.cos((np.pi)/8)])  

	# neighbors of X_goal3
	goal9_n1 = np.array([0.08-0.28*np.sin((np.pi)/8), 0.5, 0.46-0.28*np.cos((np.pi)/8)])  
	goal9_n2 = np.array([0, 0.5, 0.46-0.13])


	# colors
	X_goal7c = True #  is a mature fruit, red
	X_goal8c = True
	X_goal9c = True
	goal7_n1c = False
	goal8_n1c = False
	goal8_n2c = False
	goal9_n1c = False
	goal9_n2c = False


	# orientation: rotation about y-axis of camera2 (mobile wrt arm base) , or about z-axis of camera1 (fixed wrt arm base) 
	# the following are preset orientation , call form.get_orientation when importing pcls
	theta7 = 0 
	theta7n1 = (np.pi)/6 
	theta8 = 0 
	theta9 = (np.pi)/10
	#X_goal7o = np.array([0,0,1]) # important goal orientation definition
	X_goal7o = np.array([0,0,1]) 
	X_goal8o = np.array([0,0,1])  # important goal orientation definition
	X_goal9o = np.cos(theta9)* np.array([0,0,1])  # important goal orientation definition
	goal7_n1o = np.cos(theta7n1)* np.array([0,0,1])
	goal8_n1o = np.array([0,0,1]) 
	goal8_n2o = np.array([0,0,1])
	goal9_n1o = np.array([0,0,1]) 
	goal9_n2o = np.array([0,0,1])  


	print('Please enter which cluster configuration you want to simulate.')
	print('case 7 is configuration where NF is below goal.')
	print('case 8 is configuration where NF are below goal.')
	print('case 9 is configuration where NF are @ below and above goal.')
	conf = input()
	if conf == 7:
		bunch_xyz.append(X_goal7) 
		bunch_xyz.append(goal7_n1)
		bunch_col.append(X_goal7c)
		bunch_col.append(goal7_n1c)
		bunch_orient.append(X_goal7o)
		bunch_orient.append(goal7_n1o)
	elif conf == 8:
		bunch_xyz.append(X_goal8)
		bunch_xyz.append(goal8_n1)
		bunch_xyz.append(goal8_n2)
		bunch_col.append(X_goal8c)
		bunch_col.append(goal8_n1c)
		bunch_col.append(goal8_n2c)
		bunch_orient.append(X_goal8o)
		bunch_orient.append(goal8_n1o)
		bunch_orient.append(goal8_n2o)
	elif conf == 9:
		bunch_xyz.append(X_goal9) 
		bunch_xyz.append(goal9_n1)
		bunch_xyz.append(goal9_n2)
		bunch_col.append(X_goal9c)
		bunch_col.append(goal9_n1c)
		bunch_col.append(goal9_n2c)
		bunch_orient.append(X_goal9o)
		bunch_orient.append(goal9_n1o)
		bunch_orient.append(goal9_n2o)  
	else:
		print('configuration requested doesnt exist')

 	PushOpt_planner(bunch_xyz, bunch_col, bunch_orient, conf)
