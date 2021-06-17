import numpy as np
from Utils import RToEuler

class PIDPosAttiController():
	def __init__(self, kp, kd, kR, kw, m, g):
		self.kp = kp
		self.kd = kd
		self.kR = kR
		self.kw = kw
		self.m = m
		self.g = g

	def __call__(self, x, sp, next_sp, k):
		ep = x[0:3] - sp[0:3]
		ev = x[3:6] - sp[3:6]

		# TODO a_d?
		# TODO how is this different from geo control?
		# ^ maybe because the idea is that you use u1 to figure out what Rd and omegad are? (which you're not doing)
		# TODO do I need to put accel, velocity, omega, and atti in the trajectory?
		# ^ is that why traj opt works? shouldn't this be able to respond to a step input without a full traj though?
		u1 = np.matmul(np.array([0, 0, self.m]), self.g + self.kd * ev + self.kp * ep)

		R = np.array([[x[6], x[9], x[12]], [x[7], x[10], x[13]], [x[8], x[11], x[14]]])
		Rd = np.array([[sp[6], sp[9], sp[12]], [sp[7], sp[10], sp[13]], [sp[8], sp[11], sp[14]]])

		roll, pitch, yaw = RToEuler(R)
		roll_d, pitch_d, yaw_d = RToEuler(Rd)

		eR = np.array([roll-roll_d, pitch-pitch_d, yaw-yaw_d])
		ew = x[15:] - sp[15:]

		u2 = -self.kR * eR - self.kw * ew

		control = np.array([u1, u2[0], u2[1], u2[2]])

		return control

	def set_gains(self, gains):
		self.kp = gains[0]
		self.kd = gains[1]
		self.kR = gains[2]
		self.kw = gains[3]