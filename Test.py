import sys
sys.path.append('../ilqr/ilqr')

from ilqr import iLQR

import numpy as np
import matplotlib.pyplot as plt

from Simulate import simulate_system
from EKF import EKF
from KalmanFilter import KalmanFilter
from iLQRNonlinearDynamics import iLQR_nonlinear_dynamics
from iLQRLinearDynamics import iLQR_linear_dynamics
from iLQRCost import iLQR_cost
from NonlinearModel import create_nonlinear_model_parameters
from LinearModel import create_linear_model_parameters
from MotionModel import MotionModel
from MeasurementModel import MeasurementModel
from PIDPosAttiController import PIDPosAttiController
from PIDPositionController import PIDPosController
from GeometricPosController import GeoPosController
from iLQRController import iLQR_Controller
from NonlinearGuidanceLaw import NLGL
from Utils import actuator_limits, state_limits, hover_traj, step_response, circle_traj, line_traj

if __name__ == '__main__':

	np.random.seed(21)

	T = 0.05 # seconds
	time = 10.0 # seconds
	K = int(time/T) # num time steps
	mass = 0.734 #0.61
	J = 0.01 * np.eye(3) # np.diag(np.array([0.004, 0.004, 0.009]))
	g = 9.81
	d = 0.335/2.0
	# c = 0.2
	init_P = 0.0

	linear = False
	noise = True
	filt = True
	if filt and not noise:
		raise TypeError('noise must be True to use Kalman Filter')

	# controller = 'PIDPos'
	controller = 'GeoPos'
	# controller = 'PIDPosAtti'
	# controller = 'iLQR'

	# Immediate TODO
	# 1. reoptimize traj for any controller
	# 2. get more complicated trajectories to work (start with circle)
	# 3. debug nlgl
	# 4. disturbance estimation and rejection
	# 5. iterative learning control

	use_iLQR_traj = True
	reoptimize_traj = True
	reoptimize_freq = 2.0 # Hz
	reoptimize_k = int(1.0/T/reoptimize_freq)
	optimize_gains = False
	nlgl_on = True
	trajectory = hover_traj(0.0, 0.0, 5.0, K, linear)
	trajectory = step_response(trajectory, 0.0, 5.0, int(2.5/T), int(time/T))
	# trajectory = circle_traj(1.0, 1.0, K, linear)
	# trajectory = line_traj('x', 5.0, 0.0, 60.0, K, linear)

	x0 = trajectory[0, :]

	if linear:
		if noise:
			model_cov = [1e-3, 1e-3]
			meas_cov = [1e-2, 1e-2]
		else:
			model_cov = [0, 0]
			meas_cov = [0, 0]
		A, B, H, Q, R = create_linear_model_parameters(T, model_cov, meas_cov)
		motion_model = MotionModel([A, B, Q])
		meas_model = MeasurementModel(H, R, linear)
		if filt:
			P0 = init_P*np.eye(6)
			kalman_filter = KalmanFilter(A, B, H, Q, R, x0, P0)
			est_state = np.zeros((K, 6))
			est_cov = np.zeros((K, 6, 6))
			error = np.zeros((K, 6))
		
		
	else:
		if noise:
			model_cov = [1e-3, 1e-3, 1e-5, 1e-5]
			meas_cov = [1e-1, 1e-1, 1e-3, 1e-3]
		else:
			model_cov = [0, 0, 0, 0]
			meas_cov = [0, 0, 0, 0]
		f, F, h, H, Q, R = create_nonlinear_model_parameters(T, mass, g, J, d, model_cov, meas_cov)
		motion_model = MotionModel([f, Q])
		meas_model = MeasurementModel(h, R, linear)
		if filt:
			P0 = init_P*np.eye(18)
			kalman_filter = EKF(f, F, h, H, Q, R, x0, P0)
			est_state = np.zeros((K, 18))
			est_cov = np.zeros((K, 18, 18))
			error = np.zeros((K, 18))


	if controller == 'PIDPos':
		gains = [10, 10, 0.001] # (best for z step response)
		kp, kd, ki = gains
		controller = PIDPosController(kp, kd, ki, linear)

	elif controller == 'GeoPos':
		if linear:
			raise TypeError('This controller does not work on a linear model')
		gains = [10, 10, 0.001, 0.001] # (best for z step response)
		kx, kv, kR, kw = gains
		controller = GeoPosController(kx, kv, kR, kw, mass, g, J, T)

	elif controller == 'PIDPosAtti':
		if linear:
			raise TypeError('This controller does not work on a linear model')
		gains = [10, 10, 0.001, 0.001] # (best for z step response)
		kx, kv, kR, kw = gains
		controller = PIDPosAttiController(kx, kv, kR, kw, mass, g)

	if controller == 'iLQR' or use_iLQR_traj:
		if linear:
			dynamics = iLQR_linear_dynamics(T)
			us_init = np.zeros((K, 3))
			xs = np.zeros((K, 6))
			us = np.zeros((K, 3))
		else:
			dynamics = iLQR_nonlinear_dynamics(T, mass, g, J)
			us_init = np.zeros((K, 4))
			us_init[:, 0] = us_init[:, 0] + g
			xs = np.zeros((K, 18))
			us = np.zeros((K, 4))

		# chunks = 10
		# num_steps = int(K/chunks)
		# for i in range(chunks):
		# 	print('chunk ', i)
		# 	x0_temp = trajectory[i*num_steps, :]
		# 	if i == chunks-1:
		# 		goal = trajectory[-1, 0:3]
		# 	else:
		# 		goal = trajectory[(i+1)*num_steps, 0:3]
		# 	cost = iLQR_cost(goal, linear)
		# 	print('run iLQR')
		# 	ilqr = iLQR(dynamics, cost, num_steps-1)
		# 	xs_chunk, us_chunk = ilqr.fit(x0_temp, us_init[i*num_steps:(i+1)*num_steps-1, :])
		# 	xs[i*num_steps:(i+1)*num_steps, :] = xs_chunk
		# 	us[i*num_steps:(i+1)*num_steps-1, :] = us_chunk

		goal = trajectory[-1, 0:3]
		cost = iLQR_cost(goal, linear)
		print('run iLQR')
		ilqr = iLQR(dynamics, cost, K)
		xs, us = ilqr.fit(x0, us_init)

		trajectory = xs
		if controller == 'iLQR':
			# controller = iLQR_Controller(us)
			controller = iLQR_Controller(dynamics, us, cost, us_init, K)
			# controller = iLQR_Controller(dynamics, trajectory, us, linear, 20)

	if optimize_gains:
		best_error = np.inf
		for a in [0.1, 0.5, 1, 2, 5, 10]:
			for b in [1, 2, 5, 10, 20]:
				for c in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2]:
					for d in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:
						# print(a, b, c, d)
						gains = [a, b, c, d]
						controller.set_gains(gains)
						state, meas, setpoint, control = simulate_system(K, x0, trajectory, controller, motion_model, meas_model)
						error = np.sum(np.abs(setpoint-state))/K
						if error < best_error:
							best_error = error
							best_gains = [a, b, c, d]
							print(best_gains, best_error)
		print(best_gains, best_error)
		controller.set_gains(best_gains)

	if nlgl_on:
		nlgl = NLGL(trajectory, 1, 0.5)
	else:
		nlgl = None

	state, meas, setpoint, control = simulate_system(K, x0, trajectory, controller, motion_model, meas_model, nlgl_on, nlgl)
	tracking_error = np.sum(np.abs(setpoint-state))/K
	measurement_error = np.sum(np.abs(setpoint-meas))/K
	print(tracking_error)
	print(measurement_error)
	if filt:
		x = x0
		P = P0
		for k in range(K):
			kp1 = k+1 if k < K-1 else k
			u = controller(x, trajectory[k], trajectory[kp1], k)

			kalman_filter.predict(u)
			kalman_filter.update(meas[k, :])
			(x, P) = kalman_filter.get_state()
			est_state[k, :] = x
			est_cov[k, ...] = P
			error[k, :] = x - state[k, :]
		filtered_tracking_error = np.sum(np.abs(setpoint-est_state))/K
		print(filtered_tracking_error)


	t = np.linspace(0, K-1, K)

	plt.figure(1)
	if filt: # and linear:
		plt.plot(meas[:, 0], meas[:, 1], 'rx')
	plt.plot(setpoint[:, 0], setpoint[:, 1], '-go')
	plt.plot(state[:, 0], state[:, 1], '-bo')
	if filt:
		plt.plot(est_state[:, 0], est_state[:, 1], '-ko')
	plt.xlabel('x [m]')
	plt.ylabel('y [m]')
	if filt: # and linear:
		plt.legend(['observed measurement', 'setpoint', 'ground truth', 'state estimate'])
	# elif filt and not linear:
	# 	plt.legend(['setpoint', 'ground truth', 'state estimate'])
	else:
		plt.legend(['setpoint', 'ground truth'])

	plt.figure(2)
	if filt: # and linear:
		plt.plot(t, meas[:, 2], 'rx')
	# elif filt and not linear:
	# 	plt.plot(t, meas[:, 0], 'rx')
	plt.plot(t, setpoint[:, 2], '-go')
	plt.plot(t, state[:, 2], '-bo')
	if filt:
		plt.plot(t, est_state[:, 2], '-ko')
	plt.xlabel('t [s]')
	plt.ylabel('z [m]')
	if filt:
		plt.legend(['observed measurement', 'setpoint', 'ground truth', 'state estimate'])
	else:
		plt.legend(['setpoint', 'ground truth'])

	plt.figure(3)
	if filt: # and linear:
		plt.plot(t, meas[:, 0], 'rx')
	plt.plot(t, setpoint[:, 0], '-go')
	plt.plot(t, state[:, 0], '-bo')
	if filt:
		plt.plot(t, est_state[:, 0], '-ko')
	plt.xlabel('t [s]')
	plt.ylabel('x [m]')
	if filt: # and linear:
		plt.legend(['observed measurement', 'setpoint', 'ground truth', 'state estimate'])
	# elif filt and not linear:
	# 	plt.legend(['setpoint', 'ground truth', 'state estimate'])
	else:
		plt.legend(['setpoint', 'ground truth'])
	

	# plt.figure(4)
	# if filt and linear:
	# 	plt.plot(t, meas[:, 1], 'rx')
	# plt.plot(t, setpoint[:, 1], '-go')
	# plt.plot(t, state[:, 1], '-bo')
	# if filt:
	# 	plt.plot(t, est_state[:, 1], '-ko')
	# plt.xlabel('t [s]')
	# plt.ylabel('y [m]')
	# if filt and linear:
	# 	plt.legend(['observed measurement', 'setpoint', 'ground truth', 'state estimate'])
	# elif filt and not linear:
	# 	plt.legend(['setpoint', 'ground truth', 'state estimate'])
	# else:
	# 	plt.legend(['setpoint', 'ground truth'])

	# plt.figure(5)
	# plt.plot(t, np.log(state[:, 6]))
	# plt.plot(t, np.log(state[:, 7]))
	# plt.plot(t, np.log(state[:, 8]))
	# plt.plot(t, np.log(state[:, 9]))
	# plt.plot(t, np.log(state[:, 10]))
	# plt.plot(t, np.log(state[:, 11]))
	# plt.plot(t, np.log(state[:, 12]))
	# plt.plot(t, np.log(state[:, 13]))
	# plt.plot(t, np.log(state[:, 14]))
	# plt.legend(['Rx1', 'Rx2', 'Rx3', 'Ry1', 'Ry2', 'Ry3', 'Rz1', 'Rz2', 'Rz3'])

	# plt.figure(6)
	# plt.plot(t, state[:, 3])
	# plt.plot(t, state[:, 4])
	# plt.plot(t, state[:, 5])
	# plt.legend(['vx', 'vy', 'vz'])

	# plt.figure(7)
	# plt.plot(t, state[:, 15])
	# plt.plot(t, state[:, 16])
	# plt.plot(t, state[:, 17])
	# plt.legend(['wx', 'wy', 'wz'])

	# plt.figure(8)
	# plt.plot(t, control[:, 0])
	# plt.plot(t, control[:, 1])
	# plt.plot(t, control[:, 2])
	# plt.plot(t, control[:, 3])
	# plt.legend(['xdd', 'ydd', 'zdd'])
	# plt.legend(['f', 'M1', 'M2', 'M3'])

	plt.show()