import numpy as np
from Utils import hat

def create_nonlinear_model_parameters(T, m, g, J, d, model_cov, meas_cov):

	Jinv = np.linalg.inv(J)
	Ixx, Iyy, Izz = np.diag(J)

	# state x = [x, v, R, omega] 
	# state x = [x, y, z, xdot, ydot, zdot, Rx, Ry, Rz, omegax, omegay, omegaz]
	# state x = [x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, omegax, omegay, omegaz]

	# input u = [f, M]
	# input u = [f, M1, M2, M3]

	Q = np.eye(18)
	Q[0:3, 0:3] = model_cov[0] * Q[0:3, 0:3]
	Q[3:6, 3:6] = model_cov[1] * Q[3:6, 3:6]
	Q[6:15, 6:15] = model_cov[2] * Q[6:15, 6:15]
	Q[15:, 15:] = model_cov[3] * Q[15:, 15:]

	# R = np.eye(16)
	# R[0, 0] = meas_cov[0] * R[0, 0]
	# R[1:4, 1:4] = meas_cov[1] * R[1:4, 1:4]
	# R[4:13, 4:13] = meas_cov[2] * R[4:13, 4:13]
	# R[13:, 13:] = meas_cov[3] * R[13:, 13:]

	R = np.eye(18)
	R[0:3, 0:3] = meas_cov[0] * R[0:3, 0:3]
	R[3:6, 3:6] = meas_cov[1] * R[3:6, 3:6]
	R[6:15, 6:15] = meas_cov[2] * R[6:15, 6:15]
	R[15:, 15:] = meas_cov[3] * R[15:, 15:]

	# def f(state, u):
	# 	x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state
	# 	if type(u) == type(None):
	# 		f, M1, M2, M3 = [0, 0, 0, 0]
	# 	elif len(u) == 4:
	# 		f, M1, M2, M3 = u
	# 	state = [x + T * xdot,
	# 			 y + T * ydot,
	# 			 z + T * zdot,
	# 			 xdot + T * (-f/m * Rz1),
	# 			 ydot + T * (-f/m * Rz2),
	# 			 zdot + T * (g - f/m * Rz3),
	# 			 Rx1 + T * (Ry1 * wz - Rz1 * wy),
	# 			 Rx2 + T * (Ry2 * wz - Rz2 * wy),
	# 			 Rx3 + T * (Ry3 * wz - Rz3 * wy),
	# 			 Ry1 + T * (Rz1 * wx - Rx1 * wz),
	# 			 Ry2 + T * (Rz2 * wx - Rx2 * wz),
	# 			 Ry3 + T * (Rz3 * wx - Rx3 * wz),
	# 			 Rz1 + T * (Rx1 * wy - Ry1 * wx),
	# 			 Rz2 + T * (Rx2 * wy - Ry2 * wx),
	# 			 Rz3 + T * (Rx3 * wy - Ry3 * wx),
	# 			 wx + T * (M1/Ixx - wy * wz * (Izz/Ixx - 1)),
	# 			 wy + T * (M2/Iyy - wx * wz * (1 - Izz/Iyy)),
	# 			 wz + T * (M3/Izz)]
	# 	# print('x: ', state[0:3])
	# 	# print('v: ', state[3:6])
	# 	# print('R: ', np.array([state[6:9], state[9:12], state[12:15]]))
	# 	# print('w: ', state[15:])

	# 	return np.array([x + T * xdot,
	# 					 y + T * ydot,
	# 					 z + T * zdot,
	# 					 xdot + T * (-f/m * Rz1),
	# 					 ydot + T * (-f/m * Rz2),
	# 					 zdot + T * (g - f/m * Rz3),
	# 					 Rx1 + T * (Ry1 * wz - Rz1 * wy),
	# 					 Rx2 + T * (Ry2 * wz - Rz2 * wy),
	# 					 Rx3 + T * (Ry3 * wz - Rz3 * wy),
	# 					 Ry1 + T * (Rz1 * wx - Rx1 * wz),
	# 					 Ry2 + T * (Rz2 * wx - Rx2 * wz),
	# 					 Ry3 + T * (Rz3 * wx - Rx3 * wz),
	# 					 Rz1 + T * (Rx1 * wy - Ry1 * wx),
	# 					 Rz2 + T * (Rx2 * wy - Ry2 * wx),
	# 					 Rz3 + T * (Rx3 * wy - Ry3 * wx),
	# 					 wx + T * (M1/Ixx - wy * wz * (Izz/Ixx - 1)),
	# 					 wy + T * (M2/Iyy - wx * wz * (1 - Izz/Iyy)),
	# 					 wz + T * (M3/Izz)])

	def f(state, u):
		x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state
		# f, M1, M2, M3 = u
		if len(u) == 4:
			f, M1, M2, M3 = u
		elif len(u) == 3:
			# TODO WHAT THE FUCK
			# TODO WHERE ELSE COULD FUNDAMENTAL ERRORS BE?
			# TODO looks like it's likely a problem with the PID controller rather than the dynamics model because the geo controller might work
			xdd, ydd, zdd = u
			f = m * (zdd + g)
			M1 = Ixx * xdd / d
			M2 = Iyy * ydd / d
			M3 = 0 # Izz * (zdd + g)
		x = np.array([x, y, z])
		v = np.array([xdot, ydot, zdot])
		R = np.array([[Rx1, Ry1, Rz1], [Rx2, Ry2, Rz2], [Rx3, Ry3, Rz3]])
		# print(Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3)
		# print(R[:, 0], R[:, 1], R[:, 2])
		# print(R.flatten())
		# print('')
		w = np.array([wx, wy, wz])
		M = np.array([M1, M2, M3])
		e3 = np.array([0, 0, 1])
		xdot = v
		vdot = g*e3 - f/m * np.matmul(R, e3)
		Rdot = np.matmul(R, hat(w))
		wdot = np.matmul(Jinv, (M - np.cross(w, np.matmul(J, w))))
		state = state +  T * np.concatenate([xdot, vdot, Rdot[:, 0], Rdot[:, 1], Rdot[:, 2], wdot])
		# if not np.isnan(state[0]):
			# print('x: ', state[0:3])
			# print('v: ', state[3:6])
			# print('R: ', np.array([state[6:9], state[9:12], state[12:15]]))
			# print('w: ', state[15:])
		return state #+  T * np.concatenate([xdot, vdot, Rdot.flatten(), wdot])

	def F(state, u):
		x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state
		if len(u) == 4:
			f, M1, M2, M3 = u
		elif len(u) == 3:
			xdd, ydd, zdd = u
			f = m * (zdd + g)
			M1 = Ixx * xdd / d
			M2 = Iyy * ydd / d
			M3 = 0 # Izz * (zdd + g)
		return np.array([[1, 0, 0, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 1, 0, 0, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 1, 0, 0, T, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -f/m * T, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -f/m * T, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -f/m * T, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 1, 0, 0, T*wz, 0, 0, -T*wy, 0, 0, 0, -T*Rz1, T*Ry1],
						 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, T*wz, 0, 0, -T*wy, 0, 0, -T*Rz2, T*Ry2],
						 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, T*wz, 0, 0, -T*wy, 0, -T*Rz3, T*Ry3],
						 [0, 0, 0, 0, 0, 0, -T*wz, 0, 0, 1, 0, 0, T*wx, 0, 0, T*Rz1, 0, -T*Rx1],
						 [0, 0, 0, 0, 0, 0, 0, -T*wz, 0, 0, 1, 0, 0, T*wx, 0, T*Rz2, 0, -T*Rx2],
						 [0, 0, 0, 0, 0, 0, 0, 0, -T*wz, 0, 0, 1, 0, 0, T*wx, T*Rz3, 0, -T*Rx3],
						 [0, 0, 0, 0, 0, 0, T*wy, 0, 0, -T*wx, 0, 0, 1, 0, 0, -T*Ry1, T*Rx1, 0],
						 [0, 0, 0, 0, 0, 0, 0, T*wy, 0, 0, -T*wx, 0, 0, 1, 0, -T*Ry2, T*Rx2, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, T*wy, 0, 0, -T*wx, 0, 0, 1, -T*Ry3, T*Rx3, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, -wz*(Izz/Ixx - 1), -wy*(Izz/Ixx-1)],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -wz*(1-Izz/Iyy), 1, -wx*(1-Izz/Iyy)],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

	def h(state):
		return state
		# x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state
		# return np.array([z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz])

	def H(x):
		return np.eye(18)
		# return np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
		# 				 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

	return f, F, h, H, Q, R