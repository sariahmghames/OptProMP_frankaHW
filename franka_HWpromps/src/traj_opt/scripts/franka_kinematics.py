import numpy as np
import matplotlib.pyplot as plt
import tf.transformations as tf_tran
from mpl_toolkits import mplot3d
import scipy.optimize as opt


class FrankaKinematics():

    def __init__(self, ):
        self.a = np.array( [0, 0, 0, 0.0825, -0.0825, 0, 0.088, 0] )
        self.d = np.array( [0.333, 0, 0.316, 0, 0.384, 0, 0, 0.107] )
        self.alpha = np.array( [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, np.pi / 2, np.pi / 2, 0] )
        self.numJoints = len( self.a ) - 1
        self.T_desired = []

    def fwd_kin(self, joint_values):  # Expects a 7 x 1 values for joint_values
        joint_values = np.insert( joint_values, len( joint_values ), 0.0,
                                  axis=0 )  # added 0 at the end for the fixed joint
        #print('joint values =', joint_values)

        T = np.eye( 4 )
        T_joint = np.zeros( [self.numJoints + 1, 4, 4] )  # +1 for the fixed joint at the end
        # As per Craigs convention, T = Rotx * Trans x * Rotz * Trans z; Franka follows this
        for i in range( self.numJoints + 1 ):
            Tx = tf_tran.translation_matrix( (self.a[i], 0, 0) )  # x translation
            Rx = tf_tran.euler_matrix( self.alpha[i], 0, 0, axes='sxyz' )  # x rotation
            aa = tf_tran.concatenate_matrices( Rx, Tx )

            Tz = tf_tran.translation_matrix( (0, 0, self.d[i]) )  # z translation
            Rz = tf_tran.euler_matrix( 0, 0, joint_values[i], axes='sxyz' )  # z rotation
            ab = tf_tran.concatenate_matrices( Rz, Tz )
            ac = tf_tran.concatenate_matrices( aa, ab )
            T = tf_tran.concatenate_matrices( T, ac )
            T_joint[i, :, :] = T  # gives the transformation of each joint wrt base CS
        return T, T_joint # T will be the T_0_ee and T_joint will contain T01, T02, T03 ... T0_ee

    def fwd_kin_trajectory(self, joint_trajectory):
        endEffTrajectory = np.zeros( (joint_trajectory.shape[0], 7) )
        for i in range( joint_trajectory.shape[0] ):
            T, tmp = self.fwd_kin( joint_trajectory[i, :] )
            pos = T[0:3, 3]
            quat = tf_tran.quaternion_from_matrix( T )
            endEffTrajectory[i, :] = np.hstack( (pos, quat) )

        return endEffTrajectory

    def S_matrix(self, w):
        S = np.zeros( (3, 3) )
        S[0, 1] = -w[2]
        S[0, 2] = w[1]
        S[1, 0] = w[2]
        S[1, 2] = -w[0]
        S[2, 0] = -w[1]
        S[2, 1] = w[0]
        return S

    def jacobian(self, T_joint, T_current):  # Method 1: for finding Jacobian
        num_joints = len( self.a ) - 1
        M = np.zeros( [6, 6] )
        Jac = np.zeros( [6, num_joints] )  # initilizing jacobian
        for jp in range( num_joints ):
            j_T_ee = np.dot( tf_tran.inverse_matrix( T_joint[jp, :, :] ), T_current )
            j_t_ee = j_T_ee[0:3, 3]  # Translation part
            ee_R_j = j_T_ee[0:3, 0:3].T  # transposing
            S = self.S_matrix( j_t_ee )
            M[0:3, 0:3] = ee_R_j
            M[3:6, 3:6] = ee_R_j
            M[0:3, 3:6] = -np.dot( ee_R_j, S )
            Jac[:, jp] = M[:, 5]
        return Jac


    def geometric_jacobian(self, T_joint, T_current):  # Method 2: for finding Jacobian
        num_joints = len( self.a ) - 1
        Jac = np.zeros( [6, num_joints] )  # initilizing jacobian
        for i in range( len( T_joint ) - 1 ):
            pos_vec = T_current[0:3, 3] - T_joint[i, 0:3, 3]
            rot_axis = T_joint[i, 0:3, 2]
            Jac[0:3, i] = np.cross( rot_axis, pos_vec )
            Jac[3:6, i] = rot_axis
        return Jac


    # Method 1:
    def inv_kin(self, q_current, T_desired, ):
        T_current, T_joint = self.fwd_kin( q_current )
        num_joints = len( self.a ) - 1
        c_T_d = tf_tran.concatenate_matrices( tf_tran.inverse_matrix( T_current ),
                                              T_desired )  # transformation of desired frame with respect to current frame
        c_t_d = c_T_d[0:3, 3]  # extracting the translation part
        # c_R_d = c_T_d[0:3, 0:3]  # extracting the rotation part
        ROT = np.array( tf_tran.euler_from_matrix( c_T_d ) )
        delta_x = c_t_d
        P = 1.0
        dx = P * delta_x  # ( velocity change wrt ee_current frame)
        v_ee = np.zeros( [6, 1] )
        v_ee[0:3] = dx.reshape( [3, 1] )
        v_ee[3:6] = ROT.reshape( [3, 1] )
        # Jac = self.jacobian(T_joint, T_current)
        Jac = self.geometric_jacobian( T_joint, T_current )
        qq = np.ones( [num_joints, 1] ) * 1
        J_pinv = np.linalg.pinv( Jac )
        qn_dot = np.dot( (np.identity( num_joints ) - np.dot( J_pinv, Jac )), qq )  # null space jacobian
        final_theta = np.dot( J_pinv, v_ee )  # + qn_dot  # final_theta are the joint velocities
        final_theta = np.insert( final_theta, num_joints, 0 )  # Adding 0 at the end for the fixed joint
        return final_theta


    def inv_kin_optfun(self, q):
        T, T_joint = self.fwd_kin( q ) 
        k = T - self.T_desired
        k = np.reshape( k, (16, 1) )
        k = k.T.dot( k )
        return k


    def inv_kin2(self, q_current, T_desired):
        x0 = q_current
        self.T_desired = T_desired
        final_theta = opt.minimize( self.inv_kin_optfun, x0,
                                    method='BFGS', )  # jac=self.geometric_jacobian(T_joint, T_current))
        # print 'res \n', final_theta
        final_theta = np.insert( final_theta.x, self.numJoints, 0 )  # Adding 0 at the end for the fixed joint
        return final_theta


    def __laplace_cost_and_grad(self, theta, mu_theta, inv_sigma_theta, mu_x, inv_sigma_x):
        # f_th, jac_th, ori = self.position_and_jac(theta)
        f_th, T_joint = self.fwd_kin( theta ) 
        jac_th = self.geometric_jacobian( T_joint, f_th )
        f_th = f_th[0:3, 3]
        jac_th = jac_th[0:3, :]
        diff1 = theta - mu_theta
        tmp1 = np.dot( inv_sigma_theta, diff1 )
        diff2 = f_th - mu_x
        tmp2 = np.dot( inv_sigma_x, diff2 )
        nll = 0.5 * (np.dot( diff1, tmp1 ) + np.dot( diff2, tmp2 ))
        grad_nll = tmp1 + np.dot( jac_th.T, tmp2 )
        return nll, grad_nll


    def inv_kin_seb(self, mu_theta, sig_theta, mu_x, sig_x):
        inv_sig_theta = np.linalg.inv( sig_theta )
        inv_sig_x = np.linalg.inv( sig_x )
        cost_grad = lambda theta: self.__laplace_cost_and_grad( theta, mu_theta, inv_sig_theta, mu_x, inv_sig_x )
        cost = lambda theta: cost_grad( theta )[0]
        grad = lambda theta: cost_grad( theta )[1]
        res = opt.minimize( cost, mu_theta, method='BFGS', jac=grad )
        post_mean = res.x
        post_cov = res.hess_inv
        return post_mean, post_cov

    ##################################################
    def qrt_orient_error(self, theta, mu_ang_euler_des):
        f_th, T_joint = self.fwd_kin( theta )
        qrt_e = tf_tran.quaternion_from_matrix( f_th )  # quaternion of the end-effector at every instant
        eps_e = np.array( [qrt_e[0], qrt_e[1], qrt_e[2]] )  # vector part of quart end-eff
        eta_e = qrt_e[-1]  # scalar part of quart
        qrt_d = tf_tran.quaternion_from_euler( mu_ang_euler_des[0], mu_ang_euler_des[1], mu_ang_euler_des[2], 'szyz' )
        eps_d = np.array( [qrt_d[0], qrt_d[1], qrt_d[2]] )  # vector part of quart desired orientation of eef
        eta_d = qrt_d[-1]  # scalar part of quart
        S_ed = self.S_matrix( eps_d )
        orien_error = eta_e * eps_d - eta_d * eps_e - S_ed.dot(
            eps_e )  # eqn 3.91 Robotics Modeling Planning and Control
        # print 'orient_error', orien_error.T.dot(orien_error)
        return orien_error

    def laplace_cost_and_grad_pose(self, theta, mu_theta, inv_sigma_theta, mu_x, inv_sigma_x, mu_ang_euler_des,
                                   inv_sig_euler):
        #print('theta=', theta)
        f_th, T_joint = self.fwd_kin( theta ) # f_th is the overall robot transformation_matrix and T_joint encompasses transformation matrices up to each joint in the robot chain  
        #print('f_th=', f_th)

        jac_th = self.geometric_jacobian( T_joint, f_th )
        euler_angles = tf_tran.euler_from_matrix( f_th, 'szyz' )
        s_phi, s_nu, c_phi, c_nu = np.sin( euler_angles[0] ), np.sin(euler_angles[1]), np.cos(
            euler_angles[0] ), np.cos(euler_angles[1])
        T = np.array( [[0, -s_phi, c_phi * s_nu], [0, c_phi, s_phi * s_nu], [1, 0, c_nu]] )
        jac_analytic = np.dot( tf_tran.inverse_matrix( T ), jac_th[3:6, :] )

        pos_th = f_th[0:3, 3]
        jac_pos_th = jac_th[0:3, :]
        diff1 = theta - mu_theta
        tmp1 = np.dot( inv_sigma_theta, diff1 )
        #print('mu_x=', mu_x)
        #print('pos_th=', pos_th.shape)
        diff2 = pos_th - mu_x
        tmp2 = np.dot( inv_sigma_x, diff2 )

        ori_th = tf_tran.euler_from_matrix( f_th, 'szyz' )
        jac_ori_th = jac_analytic  # [3:6, :]
        diff3 = np.array( ori_th ) - np.array( mu_ang_euler_des )
        tmp3 = np.dot( inv_sig_euler, diff3 )

        nll = 0.5 * (np.dot( diff1, tmp1 ) + np.dot( diff2, tmp2 )) + 0.5 * np.dot( diff3, tmp3 )
        # print 'nll', nll
        # print '###########################'
        #print('sigx=', inv_sigma_x.shape)
        #print('diff2=', diff2.shape)
        #print('jac=', jac_pos_th.shape)
        grad_nll = tmp1 + np.dot( jac_pos_th.T, tmp2 ) + np.dot( jac_ori_th.T, tmp3 ) ## grad wrt theta
        #print('nll=', nll)
        #print('grad_nll=', grad_nll)
        return nll, grad_nll


    def inv_kin_ash_pose(self, mu_theta, sig_theta, mu_x, sig_x, mu_ang_euler_des, sig_euler):
        inv_sig_theta = np.linalg.inv( sig_theta )
        inv_sig_x = np.linalg.inv( sig_x )
        inv_sig_euler = np.linalg.inv( sig_euler )
        print('starting laplace cost and grad pose')
        i = 0

        cost_grad = lambda theta: self.laplace_cost_and_grad_pose( theta, mu_theta, inv_sig_theta, mu_x, inv_sig_x,
                                                                   mu_ang_euler_des, inv_sig_euler)
        i = i + 1
        #print('i=', i)
        cost = lambda theta: cost_grad(theta)[0]
        #print('cost=', cost)
        grad = lambda theta: cost_grad(theta)[1]
        #print('finishing grad retrieval and start optimization')

        # res = opt.minimize(cost, mu_theta, method='BFGS', jac=grad)
        res = opt.minimize( cost, mu_theta, method='BFGS' )
        # print 'result', res.x
        post_mean = res.x
        tmp = cost( post_mean )
        post_cov = res.hess_inv
        return post_mean, post_cov

    def plotter(self, ax, T_joint, lgnd, color='r'):
        x, y, z = 0, 0, 0
        plt.rcParams["keymap.quit"] = ["ctrl+w", "cmd+w"]
        for i in range( len( self.a ) ):
            ax.plot( [x, T_joint[i, 0, 3]], [y, T_joint[i, 1, 3]], [z, T_joint[i, 2, 3]], color, label=lgnd )
            #plt.hold( True )
            ax.scatter( T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3], 'gray' )
            x, y, z = T_joint[i, 0, 3], T_joint[i, 1, 3], T_joint[i, 2, 3]
            plt.xlabel( 'X' )
            plt.ylabel( 'Y' )
            scale = 0.4
            ax.set_xlim( -1 * scale, 1 * scale )
            ax.set_ylim( -1 * scale, 1 * scale )
            ax.set_zlim( 0, 1 )


if __name__ == '__main__':
    current_joint_values = np.zeros( 7 ) + 0.01 * np.random.random( 7 )
    ang_deg = 60
    # desired_joint_values = [np.pi*ang_deg/180, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    desired_joint_values = [np.pi * ang_deg / 180, np.pi / 3, 0.0, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6, ]
    fig = plt.figure()
    ax = fig.add_subplot( 111, projection="3d" )

    franka_kin = FrankaKinematics()
    T_current, T_joint_current = franka_kin.fwd_kin( current_joint_values )
    franka_kin.plotter( ax, T_joint_current, 'current', color='red' )
    T_desired, T_joint_desired = franka_kin.fwd_kin( desired_joint_values )
    franka_kin.plotter( ax, T_joint_desired, 'desired', color='blue' )

    # Inverse kinematics solution using optimization (without Jacobian)
    final_theta2 = franka_kin.inv_kin2( current_joint_values, T_desired )
    T_invkin2, T_joint_invkin2 = franka_kin.fwd_kin( final_theta2 )
    franka_kin.plotter( ax, T_joint_invkin2, 'Inverse_kinematics2', color='green', )

    # Sebastian's Inverse Kinematics
    mu_theta = current_joint_values
    sig_theta = np.eye( 7 ) * 0.1
    mu_x = T_desired[0:3, -1]
    sig_x = np.eye( 3 ) * 0.0002
    post_mean, post_cov = franka_kin.inv_kin_seb( mu_theta, sig_theta, mu_x, sig_x )
    T_invkinSeb, T_joint_invkinSebas = franka_kin.fwd_kin( post_mean )
    franka_kin.plotter( ax, T_joint_invkinSebas, 'Inverse_kinematicsSebas', color='cyan', )
    #

    # Ash's Inverse Kinematics with orientation
    mu_theta = current_joint_values
    sig_theta = np.eye( 7 ) * 0.1
    mu_x = T_desired[0:3, -1]
    sig_x = np.eye( 3 ) * 0.000002
    mu_ang_euler_des = tf_tran.euler_from_matrix( T_desired, 'szyz' )
    sig_euler = np.eye( 3 ) * 0.00002
    post_mean_Ash, post_cov = franka_kin.inv_kin_ash_pose( mu_theta, sig_theta, mu_x, sig_x, mu_ang_euler_des,
                                                           sig_euler )
    T_invkinAshPose, T_joint_invkinAshPose = franka_kin.fwd_kin( post_mean_Ash )
    franka_kin.plotter( ax, T_joint_invkinAshPose, 'Inverse_kinematicsAshPose', color='magenta', )

    plt.show()