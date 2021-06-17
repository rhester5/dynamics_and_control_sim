import numpy as np
import matplotlib.pyplot as plt
from ekf_control import EKF

def eulerToR(roll, pitch, yaw):
	c = np.cos
	s = np.sin
	R11 = c(yaw) * c(pitch)
	R12 = c(yaw) * s(pitch) * s(roll) - s(yaw) * c(roll)
	R13 = c(yaw) * s(pitch) * c(roll) + s(yaw) * s(roll)
	R21 = s(yaw) * c(pitch)
	R22 = s(yaw) * s(pitch) * s(roll) + c(yaw) * c(roll)
	R23 = s(yaw) * s(pitch) * c(roll) - c(yaw) * s(roll)
	R31 = -s(pitch)
	R32 = c(pitch) * s(roll)
	R33 = c(pitch) * c(roll)
	return np.array([[R11, R12, R13], [R21, R22, R23], [R31, R32, R33]])


def RToEuler(R):
	roll = np.arctan2(R[2, 1], R[2, 2])
	pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))
	yaw = np.arctan2(R[1, 0], R[0, 0])
	return roll, pitch, yaw


def hat(omega):
	wx, wy, wz = omega
	return np.array([[0, -wz, wy],
					 [wz, 0, -wx],
					 [-wy, wx, 0]])


def vee(Omega):
	return np.array([-Omega[1, 2],
					 Omega[0, 2],
					 -Omega[0, 1]])

def actuator_limits(u, g, m, d, c, J):
	Ixx, Iyy, Izz = np.diag(J)
	if len(u) == 4:
		f, M1, M2, M3 = u
		# TODO not sure if there should be an abs here
		if np.abs(f) > 2*m*g:
			f = np.sign(f) * 2*m*g
		if np.abs(M1) > 0.5*m*g*d:
			M1 = np.sign(M1) * 0.5*m*g*d
		if np.abs(M2) > 0.5*m*g*d:
			M2 = np.sign(M2) * 0.5*m*g*d
		if np.abs(M3) > 0.5*m*g*c:
			M3 = np.sign(M3) * 0.5*m*g*c
		u = np.array([f, M1, M2, M3])
	elif len(u) == 3:
		xdd, ydd, zdd = u
		# TODO same as above
		if np.abs(zdd) > g:
			zdd = np.sign(zdd) * g
		# if np.abs(xdd) > 0.5*m*g*d**2/Ixx:
		# 	xdd = np.sign(xdd) * 0.5*m*g*d**2/Ixx
		# if np.abs(ydd) > 0.5*m*g*d**2/Iyy:
		# 	ydd = np.sign(ydd) * 0.5*m*g*d**2/Iyy
		if np.abs(xdd) > 25: # 0.5*m*g*d**2/Ixx:
			xdd = np.sign(xdd) * 25 # 0.5*m*g*d**2/Ixx
		if np.abs(ydd) > 25: # 0.5*m*g*d**2/Iyy:
			ydd = np.sign(ydd) * 25 # 0.5*m*g*d**2/Iyy
		# if np.abs(zdd) > 0.5*m*g*c**2/Izz:
		# 	zdd = np.sign(zdd) * 0.5*m*g*c**2/Izz
		u = np.array([xdd, ydd, zdd])

	return u

def state_limits(x):
	_, _, yaw = RToEuler(np.array([[x[6], x[9], x[12]], [x[7], x[10], x[13]], [x[8], x[11], x[14]]]))
	xdot, ydot, zdot = x[3:6]
	if x[2] < 0:
		x[2] = 0
	if np.sqrt(xdot**2 + ydot**2) > 18:
		x[3] = 18*np.cos(yaw)
		x[4] = 18*np.sin(yaw)
	if np.abs(zdot) > 2:
		x[5] = np.sign(zdot) * 2
	return x


class MotionModel():
	def __init__(self, f, Q):
		self.f = f
		self.Q = Q

		(m, _) = Q.shape
		self.zero_mean = np.zeros(m)

	def __call__(self, x, u):
		new_state = self.f(x, u) + np.random.multivariate_normal(self.zero_mean, self.Q)
		return new_state


class MeasurementModel():
	def __init__(self, h, R):
		self.h = h
		self.R = R

		(n, _) = R.shape
		self.zero_mean = np.zeros(n)

	def __call__(self, x):
		measurement = self.h(x) + np.random.multivariate_normal(self.zero_mean, self.R)
		return measurement

class PIDController():
	def __init__(self, kp, kd, ki, m, g):
		self.kp = kp
		self.kd = kd
		self.ki = ki

		# self.total_error = np.zeros(3)

		# self.kp = kp
		# self.kd = kd
		# self.kR = kR
		# self.kw = kw
		self.m = m
		self.g = g

	def __call__(self, x, sp, k):
		error = sp - x
		p_error = error[0:3]
		d_error = error[3:6]
		# TODO this is supposed to prevent integrator windup
		# if np.linalg.norm(self.total_error) < 5:
		# 	i_error = p_error+self.total_error
		# else:
		# 	i_error = np.zeros(3) # self.total_error
		# self.total_error = i_error
		i_error = np.zeros(3)
		# if k>137 and k<141:
		# 	print('p_error ', p_error)
		# 	print('d_error ', d_error)
		# 	print('i_error ', i_error)
		control = self.kp * p_error + self.kd * d_error + self.ki * i_error
		# TODO WHAT THE FUCK
		control[0], control[1] = -control[1], control[0]
		control[2] = -control[2]

		# ep = x[0:3] - sp[0:3]
		# ev = x[3:6] - sp[3:6]

		# # TODO a_d?
		# # TODO how is this different from geo control?
		# # ^ maybe because the idea is that you use u1 to figure out what Rd and omegad are? (which you're not doing)
		# # TODO do I need to put accel, velocity, omega, and atti in the trajectory?
		# # ^ is that why traj opt works? shouldn't this be able to respond to a step input without a full traj though?
		# u1 = np.matmul(np.array([0, 0, self.m]), self.g + self.kd * ev + self.kp * ep)

		# R = np.array([[x[6], x[9], x[12]], [x[7], x[10], x[13]], [x[8], x[11], x[14]]])
		# Rd = np.array([[sp[6], sp[9], sp[12]], [sp[7], sp[10], sp[13]], [sp[8], sp[11], sp[14]]])

		# roll, pitch, yaw = RToEuler(R)
		# roll_d, pitch_d, yaw_d = RToEuler(Rd)

		# eR = np.array([roll-roll_d, pitch-pitch_d, yaw-yaw_d])
		# ew = x[15:] - sp[15:]

		# u2 = -self.kR * eR - self.kw * ew

		# control = np.array([u1, u2[0], u2[1], u2[2]])

		return control

# class GeoAttiController():

class GeoPosController():
	def __init__(self, kx, kv, kR, kw, m, g, J):
		self.m = m
		self.g = g
		self.J = J
		self.kx = kx
		self.kv = kv
		self.kR = kR
		self.kw = kw

		self.prev_Rd = np.zeros((3, 3))
		self.prev_wc = np.zeros(3)

	def __call__(self, state, sp, k):
		x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state

		x = np.array([x, y, z])
		v = np.array([xdot, ydot, zdot])
		R = np.array([[Rx1, Ry1, Rz1], [Rx2, Ry2, Rz2], [Rx3, Ry3, Rz3]])
		w = np.array([wx, wy, wz])

		e3 = np.array([0, 0, 1])
 		xd = np.array([sp[0], sp[1], sp[2]])
		xdotd = np.array([sp[3], sp[4], sp[5]])
		xddotd = np.zeros(3) # np.array([self.sp[18], self.sp[19], self.sp[20]])

		ex = x - xd
		ev = v - xdotd

		thrust = self.kx*ex + self.kv*ev + self.m*self.g*e3 - self.m*xddotd
		f = np.dot(thrust, np.matmul(R, e3))

		roll, pitch, yaw = RToEuler(R)
		b1c = np.array([np.cos(yaw), np.sin(yaw), 0])
		b3c = -thrust/np.linalg.norm(thrust)
		b2c = np.cross(b3c, b1c)

		Rd = np.block([[b1c], [b2c], [b3c]])
		Rdotd = Rd - self.prev_Rd
		self.prev_Rd = Rd

		whatc = np.matmul(np.transpose(Rd), Rdotd)
		wc = vee(whatc)
		wdotc = wc - self.prev_wc
		self.prev_wc = wc

		# eR = 0.5 * vee((np.matmul(np.transpose(Rc), R) - np.matmul(np.transpose(R), Rc)))
		# ew = w - np.matmul(np.matmul(np.transpose(R), Rc), wc)

		# Rd = np.array([self.sp[6:9], self.sp[9:12], self.sp[12:15]])
		eRhat = 0.5 * (np.matmul(np.transpose(Rd), R) - np.matmul(np.transpose(R), Rd))
		eR = vee(eRhat)
		# ew = w - np.array(self.sp[15:18])
		ew = w - np.matmul(np.matmul(np.transpose(R), Rd), wc)

		term1 = np.matmul(np.matmul(np.matmul(hat(w), np.transpose(R)), Rd), wc)
		term2 = np.matmul(np.matmul(np.transpose(R), Rd), wdotc)
		big_term = np.matmul(self.J, term1 - term2)
		M = -self.kR * eR - self.kw * ew + np.cross(w, np.matmul(self.J, w)) - big_term

		return np.array([f, M[0], M[1], M[2]])


def create_model_parameters(T, m, g, J, d, model_cov, meas_cov):

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
	R = np.eye(16)
	R[0, 0] = meas_cov[0] * R[0, 0]
	R[1:4, 1:4] = meas_cov[1] * R[1:4, 1:4]
	R[4:13, 4:13] = meas_cov[2] * R[4:13, 4:13]
	R[13:, 13:] = meas_cov[3] * R[13:, 13:]

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
			M3 = Izz * zdd / d
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
			M3 = Izz * zdd / d
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
		x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state
		return np.array([z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz])

	def H(x):
		return np.array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
						 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])

	return f, F, h, H, Q, R


def simulate_system(T, K, x0, trajectory, gains, mass, g, J, d, model_cov, meas_cov):

	(f, F, h, H, Q, R) = create_model_parameters(T, mass, g, J, d, model_cov, meas_cov)

	# Create models
	motion_model = MotionModel(f, Q)
	meas_model = MeasurementModel(h, R)
	# kp, kv, kR, kw = gains
	# controller = GeoPosController(kp, kv, kR, kw, mass, g, J)
	# controller = PIDController(kp, kv, kR, kw, mass, g)
	if len(gains) == 3:
		kp, kd, ki = gains
		controller = PIDController(kp, kd, ki, mass, g)
	elif len(gains) == 4:
		kx, kv, kR, kw = gains
		controller = GeoPosController(kx, kv, kR, kw, mass, g, J)

	(m, _) = Q.shape
	(n, _) = R.shape

	state = np.zeros((K, m))
	meas = np.zeros((K, n))
	setpoint = np.zeros((K, m))
	control = np.zeros((K, len(gains)))

	# initial state
	x = x0
	for k in range(K):
		# if k > 137 and k < 141:
		# 	print(k)
		u = controller(x, trajectory[k], k)
		u = actuator_limits(u, g, mass, d, c, J)
		# if not np.isnan(u[0]):
		# 	print(u)
		# print('u: ', u)
		x = motion_model(x, u)
		x = state_limits(x)
		# print('x: ', x)
		# if k > 137 and k < 141:
		# 	print('x: ', x[0:3])
		# 	print('v: ', x[3:6])
		# 	R = np.transpose(np.array([x[6:9]/np.linalg.norm(x[6:9]), x[9:12]/np.linalg.norm(x[9:12]), x[12:15]/np.linalg.norm(x[12:15])]))
		# 	f = mass * (u[2] + g)
		# 	print('vdot: ', g*np.array([0, 0, 1]) - f/mass * np.matmul(R, np.array([0, 0, 1])))
		# 	print('R: ', np.transpose(np.array([x[6:9]/np.linalg.norm(x[6:9]), x[9:12]/np.linalg.norm(x[9:12]), x[12:15]/np.linalg.norm(x[12:15])])))
		# 	print('w: ', x[15:])
		# 	print('u: ', u)
		# 	print('')
		z = meas_model(x)

		state[k, :] = x
		meas[k, :] = z
		setpoint[k, :] = trajectory[k]
		control[k, :] = u

	return state, meas, setpoint, control

def hover_traj(x, y, z, K):
	trajectory = np.zeros((K, 18))
	trajectory[:, :] = [x, y, z, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
	return trajectory

def step_response(trajectory, i, step, begin, end):
	trajectory[begin:end, i] = trajectory[begin:end, i] + step
	return trajectory

def circle_traj(r, z, K):
	trajectory = np.zeros((K, 18))
	theta = np.linspace(0, 2*np.pi, K/100)
	x = r * np.cos(theta)
	y = r * np.sin(theta)
	for i in range(K/100):
		trajectory[i:i+100, 0] = x[i]
		trajectory[i:i+100, 1] = y[i]
	trajectory[:, 2] = z
	trajectory[:, 6] = 1
	trajectory[:, 10] = 1
	trajectory[:, 14] = 1
	return trajectory

def line_traj(x_or_y, z, begin, end, K):
	trajectory = np.zeros((K, 18))
	line = np.linspace(begin, end, K/100)
	if x_or_y == 'x':
		for i in range(K/100-1):
			trajectory[i:i+100, 0] = line[i+1]
	elif x_or_y == 'y':
		for i in range(K/100-1):
			trajectory[i:i+100, 1] = line[i+1]
	trajectory[:, 2] = z
	trajectory[:, 6] = 1
	trajectory[:, 10] = 1
	trajectory[:, 14] = 1
	return trajectory


if __name__ == '__main__':
	np.random.seed(21)

	T = 0.01
	time = 20 # seoncds
	K = int(time/T)
	mass = 0.734 #0.61
	J = 0.01 * np.eye(3) # np.diag(np.array([0.004, 0.004, 0.009]))
	g = 9.81
	d = 0.335/2
	c = 0.2
	model_cov = [1e-3, 1e-3, 1e-6, 1e-6]
	meas_cov = [1e-1, 1e-1, 1e-1, 1e-1]

	trajectory = hover_traj(0, 0, 5.0, K)
	trajectory = step_response(trajectory, 2, 5.0, int(2.5/T), int(time/T)) # 200, 1200) #700)
	# trajectory = circle_traj(5.0, 5.0, K)
	# trajectory = line_traj('x', 5.0, 0.0, 60.0, K)
	x0 = trajectory[0, :]
	P0 = 0*np.eye(18)

	# Kp = 1
	# Kd = 0.5
	# Ki = 0.1

	# gains = [Kp, Kd, Ki]

	# kx = .020
	# kv = .012
	# kR = .040
	# kw = .060

	# v optimization bullshit

	# gains = [0.1*kx, 20*kv, kR, 0.1*kw]
	# gains = [20*kx, 20*kv, 0.03*kR, 0.12*kw]

	# (state, meas) = simulate_system(K, x0, sp, gains)

	best_error = np.inf
	for a in [0.001, 0.01, 0.1, 1, 10]:
		for b in [0.001, 0.01, 0.1, 1, 10]:
			for c in [0.001, 0.01, 0.1, 1, 10]:
				for d in [0.001, 0.01, 0.1, 1, 10]:
					print(a, b, c, d)
					gains = [a, b, c, d]
					(state, _, _, _) = simulate_system(T, K, x0, trajectory, gains, mass, g, J, d, model_cov, meas_cov)
					error = np.sum(np.abs(trajectory[1500:, 2] - state[1500:, 2]))
					# error = np.sum(np.abs(trajectory[:, 2] - state[:, 2]))/K
					# error = np.abs(trajectory[:, 2] - state[:, 2])
					# error[250:750] = 10*error[250:750]
					# error = np.sum(error)/K
					if error < best_error:
						best_error = error
						best_gains = [a, b, c, d]
						print(best_gains, best_error)
	print(best_gains, best_error)

	# best_error = np.inf
	# for a in [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]: # [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 20, 40, 100]: #, 2, 5, 10, 20, 40, 100]:
	# 	for b in [0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40]:  # [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 1, 2, 5, 10, 20, 40, 100]: #, 2, 5, 10, 20, 40, 100]:
	# 		for c in [0.001]: # [0.0001, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 1, 10]:
	# 			print(a, b, c)
	# 			gains = [a, b, c]
	# 			(state, _, _, _) = simulate_system(T, K, x0, trajectory, gains, mass, g, J, d, model_cov, meas_cov)
	# 			error = np.sum(np.abs(trajectory[:, 0] - state[:, 0]))**2/K
	# 			if error < best_error:
	# 				best_error = error
	# 				best_gains = [a, b, c]
	# 				print('new best: ', best_gains, best_error)
	# print(best_gains, best_error)

	# kx = 0.012
	# kv = 0.288
	# kR = 0.000004
	# kw = 0.0018

	gains = best_gains
	# gains = [10, 10, 0.001] # (best for z step response)
	# gains = [0.001, 0.01, 0.001]
	# gains = [10, 10, 0.001, 0.001] # (best for z step response)
	# gains = [10, 0.001, 0.001, 10]

	# gains = [0.01, 0.01, 0.01, 0.001]
	# gains = [0.01, 0.1, 10, 0.1]
	# gains = [1, 10, 0.001, 0.01]
	# gains = [10, 10, 10, 1]

	# ^ optimization bullshit

	(state, meas, setpoint, control) = simulate_system(T, K, x0, trajectory, gains, mass, g, J, d, model_cov, meas_cov)
	f, F, h, H, Q, R = create_model_parameters(T, mass, g, J, d, model_cov, meas_cov)
	kalman_filter = EKF(f, F, h, H, Q, R, x0, P0)
	if len(gains) == 3:
		kp, kd, ki = gains
		controller = PIDController(kp, kd, ki, mass, g)
	elif len(gains) == 4:
		kx, kv, kR, kw = gains
		controller = GeoPosController(kx, kv, kR, kw, mass, g, J)

	est_state = np.zeros((K, 18))
	est_cov = np.zeros((K, 18, 18))
	error = np.zeros((K, 18))

	x = x0
	P = P0
	for k in range(K):
	    u = controller(x, trajectory[k], k)

	    kalman_filter.predict(u)
	    kalman_filter.update(meas[k, :])
	    (x, P) = kalman_filter.get_state()
	    # print(x[2], state[k, 2], meas[k, 1])
	    # print()
	    # print(x[1], state[k, 1])
	    est_state[k, :] = x
	    est_cov[k, ...] = P
	    error[k, :] = x - state[k, :]

	# print(state[1:100, 0])

	t = np.linspace(0, K-1, K)

	tmin = 0 # 125 # 175 # 385
	tmax = K # 150 # 250 # 391
	xmin = -25
	xmax = 75
	ymin = -1
	ymax = 5

	# plt.figure(1)
	# plt.plot(setpoint[:, 0], setpoint[:, 1], '-go')
	# plt.plot(state[:, 0], state[:, 1], '-bo')
	# plt.xlabel('x [m]')
	# plt.ylabel('y [m]')
	# plt.xlim([xmin, xmax])
	# plt.ylim([ymin, ymax])
	# plt.legend(['true state'])

	plt.figure(2)
	plt.plot(t, meas[:, 0], 'rx')
	plt.plot(t, setpoint[:, 2], '-go')
	plt.plot(t, state[:, 2], '-bo')
	plt.plot(t, est_state[:, 2], '-ko')
	plt.xlabel('t [s]')
	plt.ylabel('z [m]')
	plt.xlim([tmin, tmax])
	# plt.ylim([ymin, ymax])
	# plt.legend(['setpoint', 'ground truth'])
	plt.legend(['observed measurement', 'setpoint', 'ground truth', 'state estimate'])

	plt.figure(3)
	plt.plot(t, setpoint[:, 0], '-go')
	plt.plot(t, state[:, 0], '-bo')
	plt.plot(t, est_state[:, 0], '-ko')
	plt.xlabel('t [s]')
	plt.ylabel('x [m]')
	plt.xlim([tmin, tmax])
	# plt.ylim([ymin, ymax])
	plt.legend(['ground truth', 'observed measurement'])
	plt.legend(['ground truth', 'observed measurement', 'state estimate'])

	plt.figure(8)
	# plt.plot(t, meas[:, 0], 'rx')
	plt.plot(t, setpoint[:, 1], '-go')
	plt.plot(t, state[:, 1], '-bo')
	plt.xlabel('t [s]')
	plt.ylabel('y [m]')
	plt.xlim([tmin, tmax])
	# plt.ylim([ymin, ymax])
	# plt.legend(['true state', 'observed measurement'])

	plt.figure(4)
	plt.plot(t, np.log(state[:, 6]))
	plt.plot(t, np.log(state[:, 7]))
	plt.plot(t, np.log(state[:, 8]))
	plt.plot(t, np.log(state[:, 9]))
	plt.plot(t, np.log(state[:, 10]))
	plt.plot(t, np.log(state[:, 11]))
	plt.plot(t, np.log(state[:, 12]))
	plt.plot(t, np.log(state[:, 13]))
	plt.plot(t, np.log(state[:, 14]))
	plt.xlim([tmin, tmax])
	plt.legend(['Rx1', 'Rx2', 'Rx3', 'Ry1', 'Ry2', 'Ry3', 'Rz1', 'Rz2', 'Rz3'])

	plt.figure(5)
	plt.plot(t, state[:, 3])
	plt.plot(t, state[:, 4])
	plt.plot(t, state[:, 5])
	plt.legend(['vx', 'vy', 'vz'])
	plt.xlim([tmin, tmax])
	# plt.ylim([ymin, ymax])

	plt.figure(6)
	plt.plot(t, state[:, 15])
	plt.plot(t, state[:, 16])
	plt.plot(t, state[:, 17])
	plt.legend(['wx', 'wy', 'wz'])
	plt.xlim([tmin, tmax])

	plt.figure(7)
	plt.plot(t, control[:, 0])
	plt.plot(t, control[:, 1])
	plt.plot(t, control[:, 2])
	plt.legend(['xdd', 'ydd', 'zdd'])
	plt.xlim([tmin, tmax])

	plt.show()