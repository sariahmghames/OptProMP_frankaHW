import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from cost_function import CostFunction
import scipy.ndimage
import basis as basis
import phase as phase
from scipy import linalg
from franka_kinematics import FrankaKinematics
import scipy.optimize as opt

from pylab import * # support subplot


with open('100demos.npz', 'r') as f:
    data = np.load(f)
    Q = data['Q']
    time_q = data['time']


class takayuki_ral:

    def __init__(self, ee_trj, ee_time):
        self.ee_trj = ee_trj
        self.n_dem = len(ee_trj)
        self.time = ee_time
        self.ndof = 7
        self.franka_kin = FrankaKinematics()


    def get_par_promp(self, basisgenerator):
        # n_dof = self.ndof
        d_sample = 4
        ns = len(self.ee_trj[0])//d_sample
        t = np.linspace(0, 1, ns)
        indx = np.arange((ns)) * d_sample
        basisfunc = basisgenerator.basisMultiDoF(t, 1)
        w1_all = []
        w2_all = []
        w3_all = []
        n_dof = self.ndof #  No. of dof for ProMP
        x = np.zeros([1, ns, n_dof])
        q1 = np.zeros([1, ns, n_dof])
        eps = 10**(-10)
        for i in range(len(self.ee_trj)):
            q = self.ee_trj[i]
            traj = self.franka_kin.fwd_kin_trajectory(q)
            time = np.linspace(0, 1, len(traj))
            bf = basisgenerator.basisMultiDoF(time, 1)
            wx = np.array(np.inner(np.linalg.pinv(bf + eps), traj.transpose()))
            wq = np.array(np.inner(np.linalg.pinv(bf + eps), q.transpose()))
            w10 = np.array(np.inner(np.linalg.pinv(bf) + eps, q[:, 1]))
            w20 = np.array(np.inner(np.linalg.pinv(bf) + eps, q[:, 1]))
            w30 = np.array(np.inner(np.linalg.pinv(bf) + eps, q[:, 2]))
            q0 = np.array([np.inner(basisfunc, wq.transpose()[:n_dof, :])])
            x0 = np.array([np.inner(basisfunc, wx.transpose()[:n_dof, :])])
            q1 = np.concatenate((q1, q0))
            x = np.concatenate((x, x0))
            w1_all = np.append(w1_all, w10)
            w2_all = np.append(w2_all, w20)
            w3_all = np.append(w3_all, w30, axis=0)
        xtrj = x[1:, :, :]
        qtrj = q1[1:, :, :]

        qmean = np.mean(qtrj, axis=0)
        N = xtrj.shape[0]
        xMean = xtrj.sum(0, keepdims=True) / N
        m = x - xMean
        xCov = np.einsum('ijk,ijl->jkl', m, m) / (N - 1)
        x_Cov = np.array([xCov[:, 0, 0], xCov[:, 1, 1], xCov[:, 2, 2]])
        orientation_sam_dem = self.ee_trj[0][indx, 3:]
        return xMean[0, :, :3], x_Cov.transpose(), orientation_sam_dem, qmean

class trajectory_optimization():

    def __init__(self, xmean, xcov, orientatoin_sam_dem, q):
        self.franka_kin = FrankaKinematics()
        self.cost_fun = CostFunction()
        # self.object_list = np.array([[0.25, 0.21, 0.71, 0.1], [0.25, -0.21, 0.51, 0.1]])  # for two spheres
        self.object_list = np.array([[0.25, 0.0, 0.71, 0.1]])  # for one sphere (x,y,z, radius)
        self.initial_joint_values = np.zeros(7)  # + 0.01 * np.random.random(7)
        self.ang_deg = 60
        self.desired_joint_values = np.array(
            [np.pi * self.ang_deg / 180, np.pi / 3, 0.0, np.pi / 6, np.pi / 6, np.pi / 6, np.pi / 6])
        self.optCurve = []
        self.step_size = 0.02
        self.xcov = xcov
        self.xmean = xmean
        self.orien = orientatoin_sam_dem
        self.q = q
        self.w_dem = 0.005
        self.w_man = 1


    def discretize(self, start, goal, step_size=0.02):
        w = goal - start
        step = step_size * w / np.linalg.norm(w)
        n = int(np.linalg.norm(w) / np.linalg.norm(step)) + 1
        discretized = np.outer(np.arange(1, n), step) + start
        return discretized

    def sphere(self, ax, radius, centre):
        u = np.linspace(0, 2 * np.pi, 13)
        v = np.linspace(0, np.pi, 7)
        x = centre[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = centre[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = centre[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        xdata = scipy.ndimage.zoom(x, 3)
        ydata = scipy.ndimage.zoom(y, 3)
        zdata = scipy.ndimage.zoom(z, 3)
        ax.plot_surface(xdata, ydata, zdata, rstride=3, cstride=3, color='w', shade=0)

    def finite_diff_matrix(self, trajectory):
        rows, columns = trajectory.shape  # columns = nDoF
        A = 2 * np.eye(rows)
        A[0, 1] = -1
        A[rows-1, rows-2] = -1
        for ik in range(0, rows-2):
            A[ik + 1, ik] = -1
            A[ik + 1, ik + 2] = -1

        dim = rows*columns
        fd_matrix = np.zeros((dim, dim))
        b = np.zeros((dim, 1))
        i, j = 0, 0
        while i < dim:
            fd_matrix[i:i+len(A), i:i+len(A)] = A
            b[i] = -2 * self.initial_joint_values[j]
            b[i+len(A)-1] = -2 * self.desired_joint_values[j]
            i = i + len(A)
            j = j + 1
        return fd_matrix, b

    def smoothness_objective(self, trajectory):
        trajectory = np.squeeze(trajectory)
        trajectory = np.transpose(trajectory.reshape((7, len(trajectory) / 7)))
        rows, columns = trajectory.shape
        dim = rows * columns
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        F_smooth = 0.5*(np.dot(trajectory.T, np.dot(fd_matrix, trajectory)) + np.dot(trajectory.T, b) + 0.25*np.dot(b.T, b))
        return F_smooth

    def calculate_obstacle_cost(self, trajectory):
        obstacle_cost = 0
        trajectory = np.squeeze(trajectory)
        trajectory = np.transpose(trajectory.reshape((7, len(trajectory) / 7)))
        vel_normalised, vel_mag, vel = self.calculate_normalised_workspace_velocity(
            trajectory)  # vel_normalised = vel/vel_mag
        for jvs in range(len(trajectory)):
            robot_discretised_points = np.array(
                self.cost_fun.get_robot_discretised_points(trajectory[jvs], self.step_size))
            dist = self.cost_fun.compute_minimum_distance_to_objects(robot_discretised_points, self.object_list)
            obsacle_cost_potential = np.array(self.cost_fun.cost_potential(dist))
            obstacle_cost += np.sum(np.multiply(obsacle_cost_potential, vel_mag[jvs, :]))
            # obstacle_cost += np.sum(obsacle_cost_potential)
        return obstacle_cost

    def calculate_t_cost_ral(self, trajectory):
        F_smooth = self.smoothness_objective(trajectory)  # smoothness of trajectory is captured here
        obstacle_cost = self.calculate_obstacle_cost(trajectory)
        dem_cost = self.calculate_demosntration_cost(trajectory)
        manip = np.sum(self.calculate_mani(trajectory))**(-1)

        return 10 * F_smooth + 1.5 * obstacle_cost + self.w_dem * dem_cost + self.w_man * manip
    def calculate_total_cost(self, trajectory):
        F_smooth = self.smoothness_objective(trajectory)  # smoothness of trajectory is captured here
        obstacle_cost = self.calculate_obstacle_cost(trajectory)
        return 10 * F_smooth + 1.5 * obstacle_cost

    def calculate_smoothness_gradient(self, trajectory):
        trajectory = np.squeeze(trajectory)
        trajectory = np.transpose(trajectory.reshape((7, len(trajectory) / 7)))
        rows, columns = trajectory.shape
        dim = rows * columns
        fd_matrix, b = self.finite_diff_matrix(trajectory)
        trajectory = trajectory.T.reshape((dim, 1))
        smoothness_gradient = 0.5*b + fd_matrix.dot(trajectory)
        return smoothness_gradient #/np.linalg.norm(smoothness_gradient)

    def calculate_normalised_workspace_velocity(self, trajectory):
        # We have not divided by  time as this has been indexed and is thus not available
        trajectory = np.insert(trajectory, len(trajectory), self.desired_joint_values, axis=0)
        robot_body_points = np.array([self.cost_fun.get_robot_discretised_points(joint_values, self.step_size) for joint_values in trajectory])
        velocity = np.diff(robot_body_points, axis=0)
        vel_magnitude = np.linalg.norm(velocity, axis=2)
        vel_normalised = np.divide(velocity, vel_magnitude[:, :, None], out=np.zeros_like(velocity), where=vel_magnitude[:, :, None] != 0)
        return vel_normalised, vel_magnitude, velocity

    def calculate_curvature(self, vel_normalised, vel_magnitude, velocity):
        time_instants, body_points, n = velocity.shape[0], velocity.shape[1], velocity.shape[2]
        acceleration = np.gradient(velocity, axis=0)
        curvature, orthogonal_projector = np.zeros((time_instants, body_points, n)), np.zeros((time_instants, body_points, n, n))
        for tm in range(time_instants):
            for pts in range(body_points):
                ttm = np.dot(vel_normalised[tm, pts, :].reshape(3, 1), vel_normalised[tm, pts, :].reshape(1, 3))
                temp = np.eye(3) - ttm
                orthogonal_projector[tm, pts] = temp
                if vel_magnitude[tm, pts]:
                    curvature[tm, pts] = np.dot(temp, acceleration[tm, pts, :])/vel_magnitude[tm, pts]**2
                else:
                    # curv.append(np.array([0, 0, 0]))
                    curvature[tm, pts] = np.array([0, 0, 0])
        return curvature, orthogonal_projector

    def fun(self, points):
        dist = self.cost_fun.compute_minimum_distance_to_objects(points, self.object_list)
        return np.array(self.cost_fun.cost_potential(dist))

    def gradient_cost_potential(self, robot_discretised_points):
        gradient_cost_potential = list()
        for points in robot_discretised_points:
            grad = opt.approx_fprime(points, self.fun, [1e-06, 1e-06, 1e-06])
            gradient_cost_potential.append(grad)
        return np.array(gradient_cost_potential)

    def calculate_mt(self, trajectory):
        n_time, ndof = trajectory.shape  #
        M_t =np.zeros((n_time, ndof, n_time*ndof))
        for t in range(n_time):
            k = 0
            for d in range(ndof):
                M_t[t, d, k + t] = 1
                k = k + n_time
        return M_t

    def calculate_obstacle_cost_gradient(self, trajectory):
        trajectory = np.squeeze(trajectory)
        trajectory = np.transpose(trajectory.reshape((7, len(trajectory) / 7)))
        vel_normalised, vel_magnitude, velocity = self.calculate_normalised_workspace_velocity(trajectory)
        curvature, orthogonal_projector = self.calculate_curvature(vel_normalised, vel_magnitude, velocity)
        obstacle_gradient = list()
        for jvs in range(len(trajectory)):
            obst_grad = np.zeros(trajectory.shape[1])
            robot_discretised_points, joint_index = np.array(self.cost_fun.get_robot_discretised_points(trajectory[jvs], self.step_size, with_joint_index=True))
            dist = self.cost_fun.compute_minimum_distance_to_objects(robot_discretised_points, self.object_list,)
            obstacle_cost_potential = np.array(self.cost_fun.cost_potential(dist))
            gradient_cost_potential = self.gradient_cost_potential(robot_discretised_points)
            jacobian = self.cost_fun.calculate_jacobian(robot_discretised_points, joint_index, trajectory[jvs])
            for num_points in range(robot_discretised_points.shape[0]):
                temp1 = orthogonal_projector[jvs, num_points].dot(gradient_cost_potential[num_points, :])
                temp2 = obstacle_cost_potential[num_points] * curvature[jvs, num_points, :]
                temp3 = vel_magnitude[jvs, num_points] * (temp1 - temp2)
                temp4 = jacobian[num_points, :].T.dot(temp3)
                obst_grad += temp4   #/np.linalg.norm(temp4)
                # obst_grad += jacobian[num_points, :].T.dot(gradient_cost_potential[num_points, :])
            obstacle_gradient.append(obst_grad)
        temp5 = np.transpose(np.array(obstacle_gradient))
        return temp5

    def calculate_demosntration_cost(self, joint_trajectory):
        joint_index = 7
        cost = 0
        npoints = len(self.xmean)
        for i in range(npoints):
            joint_val = np.squeeze(joint_trajectory)[7*i:7 * (i+1)]
            if len(joint_val)!= 7:
                jj = 1
            T_current, T_joint = self.franka_kin.fwd_kin(joint_val)
            eePos = T_current[0:3, 3]
            deltaX = self.xmean[i]-eePos
            tmp = np.inner(deltaX,  np.multiply(np.reciprocal(self.xcov[i]+0.00001), deltaX) )  # delta_x inv_Sigma delta_x
            cost += tmp
        return cost

    def calculate_manipulability(self, joint_val):
        T_current, T_joint = self.franka_kin.fwd_kin(joint_val)
        jac = self.franka_kin.jacobian(T_joint, T_current)
        manipul = np.linalg.det(np.inner(jac, jac))**0.5
        return manipul

    def calculate_mani(self, joint_trajectory):
        npoints = len(self.xmean)
        manip = []
        for i in range(npoints):
            joint_val = np.squeeze(joint_trajectory)[7*i:7 * (i+1)]
            manip = np.append(manip, self.calculate_manipulability(joint_val))
        return manip


    def calculate_mani_grad(self, joint_trajectory):
        npoints = len(self.xmean)
        eps = 1.001
        delta_q = np.array([[eps, 1, 1, 1, 1, 1, 1],
                            [1, eps, 1, 1, 1, 1, 1],
                            [1, 1, eps, 1, 1, 1, 1],
                            [1, 1, 1, eps, 1, 1, 1],
                            [1, 1, 1, 1, eps, 1, 1],
                            [1, 1, 1, 1, 1, eps, 1],
                            [1, 1, 1, 1, 1, 1, eps]])
        TotmanGrad = []
        manipTot = []
        for i in range(npoints):
            joint_val = np.squeeze(joint_trajectory)[7*i:7 * (i+1)]
            manip = (self.calculate_manipulability(joint_val))**(-1)
            gradMan = []
            for j in range(0, 7):
                dq = np.multiply(delta_q[j, :], joint_val)
                djac = self.calculate_manipulability(dq)
                gradMan = np.append(gradMan, (djac**(-1) - manip) / eps)
            manipTot = np.append(manipTot, manip)
            TotmanGrad = np.append(TotmanGrad, gradMan)
        return TotmanGrad

    def calculate_demosntration_gradient(self, joint_trajectory):
        joint_index = 7
        deltaDem = []
        npoints = len(self.xmean)
        for i in range(npoints):
            joint_val = np.squeeze(joint_trajectory)[7*i:7 * (i+1)]
            T_current, T_joint = self.franka_kin.fwd_kin(joint_val)
            eePos = T_current[0:3, 3]
            jac = self.franka_kin.jacobian(T_joint, T_current)
            deltaX = self.xmean[i]-eePos
            tmp = np.inner(jac.transpose()[:,:3],  np.multiply(np.reciprocal(self.xcov[i]), deltaX) )  # JT inv_Sigma delta_x
            deltaDem = np.append(deltaDem, tmp, axis=0)
        delDem = deltaDem.reshape([len(deltaDem)/7, 7])
        return deltaDem

    def cost_gradient_analytic(self, trajectory):  # calculate grad(cost) = grad(smoothness_cost) + grad(obstacle_cost)
        smoothness_gradient = self.calculate_smoothness_gradient(trajectory)
        obstacle_gradient = self.calculate_obstacle_cost_gradient(trajectory)
        demonstration_gradient = self.calculate_demosntration_gradient(trajectory)
        trajectory = np.squeeze(trajectory)
        cost_gradient = obstacle_gradient.reshape((len(trajectory), 1)) + smoothness_gradient #+ self.w_dem * demonstration_gradient.reshape((len(trajectory), 1))
        return np.squeeze(cost_gradient)

    def cost_grad_analytic_ral(self, trajectory):  # calculate grad(cost) = grad(smoothness_cost) + grad(obstacle_cost)
        smoothness_gradient = self.calculate_smoothness_gradient( trajectory )
        obstacle_gradient = self.calculate_obstacle_cost_gradient( trajectory )
        demonstration_gradient = self.calculate_demosntration_gradient( trajectory )
        manip_cost = self.calculate_mani_grad(trajectory)
        trajectory = np.squeeze( trajectory )
        cost_gradient = obstacle_gradient.reshape(
            (len( trajectory ), 1) ) + smoothness_gradient + self.w_dem * demonstration_gradient.reshape(
            (len( trajectory ), 1) ) + self.w_man * manip_cost.reshape(len( trajectory ), 1)
        return np.squeeze( cost_gradient )

    def cost_gradient_numeric(self, trajectory):
        trajectory = np.squeeze(trajectory)
        obst_cost_grad_numeric = opt.approx_fprime(trajectory, traj_opt.calculate_obstacle_cost,
                                              1e-08 * np.ones(len(trajectory)))
        smoothness_gradient = np.squeeze(self.calculate_smoothness_gradient(trajectory))
        return np.squeeze(obst_cost_grad_numeric + smoothness_gradient)

    def animation(self, optimized_trajectory, initial_trajectory):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        # ax.axis('off')
        plt.show(block=False)
        while True:
            for i in range(len(optimized_trajectory)):
                _, T_joint_optim = self.franka_kin.fwd_kin(optimized_trajectory[i])
                _, T_joint_init = self.franka_kin.fwd_kin(initial_trajectory[i])
                ax.clear()
                self.sphere(ax, 0.05, [0.25, 0.21, 0.71])
                self.franka_kin.plotter(ax, T_joint_optim, 'optim', color='blue')
                self.franka_kin.plotter(ax, T_joint_init, 'init', color='red')
                # for x, y, z in self.cost_fun.get_robot_discretised_points(trajectory[i],step_size=0.2):
                #     plt.grid()
                #     ax.scatter(x, y, z, 'gray')
                fig.canvas.draw()
                fig.canvas.flush_events()
                plt.pause(0.01)
            plt.pause(1)
        # plt.show(block=True)

    def optim_callback(self, xk):
        # xk = xk.reshape(len(xk) / 7, 7)
        costs = self.calculate_total_cost(xk)
        self.optCurve.append(xk)
        print('Iteration {}: {}\n'.format(len(self.optCurve), costs))

class ProMP:

    def __init__(self, basis, phase, numDoF):
        self.basis = basis
        self.phase = phase
        self.numDoF = numDoF
        self.numWeights = basis.numBasis * self.numDoF
        self.mu = np.zeros(self.numWeights)
        self.covMat = np.eye(self.numWeights)
        self.observationSigma = np.ones(self.numDoF)

    def getTrajectorySamples(self, time, n_samples=1):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        weights = np.random.multivariate_normal(self.mu, self.covMat, n_samples)
        weights = weights.transpose()
        trajectoryFlat = basisMultiDoF.dot(weights)
        # a = trajectoryFlat
        trajectoryFlat = trajectoryFlat.reshape((self.numDoF, trajectoryFlat.shape[0] / self.numDoF, n_samples))
        trajectoryFlat = np.transpose(trajectoryFlat, (1, 0, 2))
        # trajectoryFlat = trajectoryFlat.reshape((a.shape[0] / self.numDoF, self.numDoF, n_samples))

        return trajectoryFlat

    def getMeanAndCovarianceTrajectory(self, time):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        trajectoryFlat = basisMultiDoF.dot(self.mu.transpose())
        trajectoryMean = trajectoryFlat.reshape((self.numDoF, trajectoryFlat.shape[0] / self.numDoF))
        trajectoryMean = np.transpose(trajectoryMean, (1, 0))
        covarianceTrajectory = np.zeros((self.numDoF, self.numDoF, len(time)))

        for i in range(len(time)):
            basisSingleT = basisMultiDoF[slice(i, (self.numDoF - 1) * len(time) + i + 1, len(time)), :]
            covarianceTimeStep = basisSingleT.dot(self.covMat).dot(basisSingleT.transpose())
            covarianceTrajectory[:, :, i] = covarianceTimeStep

        return trajectoryMean, covarianceTrajectory

    def getMeanAndStdTrajectory(self, time):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)
        trajectoryFlat = basisMultiDoF.dot(self.mu.transpose())
        trajectoryMean = trajectoryFlat.reshape((self.numDoF, trajectoryFlat.shape[0] / self.numDoF))
        trajectoryMean = np.transpose(trajectoryMean, (1, 0))
        stdTrajectory = np.zeros((len(time), self.numDoF))

        for i in range(len(time)):
            basisSingleT = basisMultiDoF[slice(i, (self.numDoF - 1) * len(time) + i + 1, len(time)), :]
            covarianceTimeStep = basisSingleT.dot(self.covMat).dot(basisSingleT.transpose())
            stdTrajectory[i, :] = np.sqrt(np.diag(covarianceTimeStep))

        return trajectoryMean, stdTrajectory

    def getMeanAndCovarianceTrajectoryFull(self, time):
        basisMultiDoF = self.basis.basisMultiDoF(time, self.numDoF)

        meanFlat = basisMultiDoF.dot(self.mu.transpose())
        covarianceTrajectory = basisMultiDoF.dot(self.covMat).dot(basisMultiDoF.transpose())

        return meanFlat, covarianceTrajectory

    def jointSpaceConditioning(self, time, desiredTheta, desiredVar):
        newProMP = ProMP(self.basis, self.phase, self.numDoF)
        basisMatrix = self.basis.basisMultiDoF(time, self.numDoF)
        temp = self.covMat.dot(basisMatrix.transpose())
        L = np.linalg.solve(desiredVar + basisMatrix.dot(temp), temp.transpose())
        L = L.transpose()
        newProMP.mu = self.mu + L.dot(desiredTheta - basisMatrix.dot(self.mu))
        newProMP.covMat = self.covMat - L.dot(basisMatrix).dot(self.covMat)
        return newProMP

    def getTrajectoryLogLikelihood(self, time, trajectory):

        trajectoryFlat = trajectory.transpose().reshape(trajectory.shape[0] * trajectory.shape[1])
        meanFlat, covarianceTrajectory = self.getMeanAndCovarianceTrajectoryFull(self, time)

        return stats.multivariate_normal.logpdf(trajectoryFlat, mean=meanFlat, cov=covarianceTrajectory)

    def getWeightsLogLikelihood(self, weights):

        return stats.multivariate_normal.logpdf(weights, mean=self.mu, cov=self.covMat)

    def plotProMP(self, time, indices=None):
        import plotter as plotter

        trajectoryMean, stdTrajectory = self.getMeanAndStdTrajectory(time)

        plotter.plotMeanAndStd(time, trajectoryMean, stdTrajectory, indices)

    def dem_mean_cov(self, traj_dem):
        traj_mean = traj_dem
        traj_cov = traj_dem
        return traj_mean, traj_cov

class MAPWeightLearner():

    def __init__(self, proMP, regularizationCoeff=10 ** -9, priorCovariance=10 ** -4, priorWeight=1):
        self.proMP = proMP
        self.priorCovariance = priorCovariance
        self.priorWeight = priorWeight
        self.regularizationCoeff = regularizationCoeff

    def learnFromData(self, trajectoryList, timeList):
        numTraj = len(trajectoryList)
        weightMatrix = np.zeros((numTraj, self.proMP.numWeights))
        for i in range(numTraj):
            trajectory = trajectoryList[i]
            time = timeList[i]
            trajectoryFlat = trajectory.transpose().reshape(trajectory.shape[0] * trajectory.shape[1])
            basisMatrix = self.proMP.basis.basisMultiDoF(time, self.proMP.numDoF)
            temp = basisMatrix.transpose().dot(basisMatrix) + np.eye(self.proMP.numWeights) * self.regularizationCoeff
            weightVector = np.linalg.solve(temp, basisMatrix.transpose().dot(trajectoryFlat))
            weightMatrix[i, :] = weightVector

        self.proMP.mu = np.mean(weightMatrix, axis=0)

        sampleCov = np.cov(weightMatrix.transpose())
        self.proMP.covMat = (numTraj * sampleCov + self.priorCovariance * np.eye(self.proMP.numWeights)) / (
                    numTraj + self.priorCovariance)

if __name__ == '__main__':
    phaseGenerator = phase.LinearPhaseGenerator()
    basisGenerator = basis.NormalizedRBFBasisGenerator(phaseGenerator, numBasis=5, duration=1, basisBandWidthFactor=5,
                                                       numBasisOutside=1)
    time = np.linspace(0, 1, 50)
    nDof = 7
    proMP = ProMP( basisGenerator, phaseGenerator, nDof )  # 3 argument = nDOF
    trajectories = proMP.getTrajectorySamples(time, 4)  # 2nd argument is numSamples/Demonstrations/trajectories
    meanTraj, covTraj = proMP.getMeanAndCovarianceTrajectory(time)

    time_stacked = np.matlib.repmat( time, 7, 1 )
    meanTraj = meanTraj + 0.1 * time_stacked.transpose()

    franka_kin = FrankaKinematics()

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection="3d")
    #ax.clear()
    endEffTrajectory = list()
    eet =np.empty([trajectories.shape[2]], dtype=object)
    #eet = np.array([franka_kin.fwd_kin_trajectory(trajectories[:, :, 0]), []], dtype=object)

    for i in range(trajectories.shape[2]):
        endEffTrajectory_tmp = franka_kin.fwd_kin_trajectory(trajectories[:, :, i])
        endEffTrajectory.append(np.array(endEffTrajectory_tmp))
        eet[i] = np.array(franka_kin.fwd_kin_trajectory(trajectories[:, :, 0]), dtype=object)

    learnedProMP = ProMP(basisGenerator, phaseGenerator, nDof)
    learner = MAPWeightLearner(learnedProMP)
    learner.learnFromData(Q, time_q)
    mu_theta, sig_theta = learnedProMP.getMeanAndCovarianceTrajectory(np.array([1.0]))

    trj_takayuki = takayuki_ral(Q, time_q)
    [xMEAN, x_Cov, orientatoin_sam_dem, q] = trj_takayuki.get_par_promp(basisGenerator)

    subplot( 3, 1, 1 )
    xticks( [] ), yticks( [] )
    title( 'X_{mean}' )
    plt.plot( xMEAN[:, 0], 'r' )


    subplot( 3, 1, 2 )
    #xticks( [] ), yticks( [] )
    title( 'Y_{mean}' )
    plt.plot( xMEAN[:, 1], 'b' )

    subplot( 3, 1, 3 )
    xticks( [] ), yticks( [] )
    title( 'Z_{mean}' )
    plt.plot( xMEAN[:, 2], 'g' )

    show()


    plt.figure(5)
    plt.plot(x_Cov[:, 0], 'r', x_Cov[:, 1], 'b', x_Cov[:, 2], 'g')
    plt.title('Covariance of the demonstrations')

    traj_MEAN = np.append(xMEAN, orientatoin_sam_dem, axis=1)

    trajectoryFlat = traj_MEAN.T.reshape((-1))




    trajOpt = trajectory_optimization(xMEAN, x_Cov, orientatoin_sam_dem, q)


    optimized_trajectory = opt.minimize(trajOpt.calculate_total_cost, trajectoryFlat, method='BFGS', jac=trajOpt.cost_gradient_analytic, options={'maxiter': 15, 'disp': True}, callback=trajOpt.optim_callback)
    optimized_trajec_ral = opt.minimize(trajOpt.calculate_t_cost_ral, trajectoryFlat, method='BFGS', jac=trajOpt.cost_grad_analytic_ral, options={'maxiter': 15, 'disp': True}, callback=trajOpt.optim_callback)

    optimized_trajectory = np.transpose(optimized_trajectory.x.reshape((7, len(optimized_trajectory.x) / 7)))
    optimized_trajec_ral = np.transpose(optimized_trajec_ral.x.reshape((7, len(optimized_trajec_ral.x) / 7)))
    initial_trajectory  = q
    plt.figure()
    trajOpt.animation(initial_trajectory, initial_trajectory)
    plt.figure()
    trajOpt.animation(optimized_trajec_ral, initial_trajectory)

    end = 1
    # #plt.close('all')
