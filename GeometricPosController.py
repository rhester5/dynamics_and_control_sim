import numpy as np
from Utils import RToEuler, vee, hat

class GeoPosController():
	def __init__(self, kx, kv, kR, kw, m, g, J, dt):
		self.m = m
		self.g = g
		self.J = J
		self.dt = dt
		self.kx = kx
		self.kv = kv
		self.kR = kR
		self.kw = kw

		# self.prev_Rd = np.zeros((3, 3))
		# self.prev_wc = np.zeros(3)

	def __call__(self, state, sp, next_sp, k):
		x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz = state

		x = np.array([x, y, z])
		v = np.array([xdot, ydot, zdot])
		R = np.array([[Rx1, Ry1, Rz1], [Rx2, Ry2, Rz2], [Rx3, Ry3, Rz3]])
		w = np.array([wx, wy, wz])

		e3 = np.array([0, 0, 1])
		xd = np.array([sp[0], sp[1], sp[2]])
		xdotd = np.array([sp[3], sp[4], sp[5]])
		xddotd = (next_sp[3:6] - sp[3:6])/self.dt # np.zeros(3) # np.array([self.sp[18], self.sp[19], self.sp[20]])

		ex = x - xd
		ev = v - xdotd

		thrust = self.kx*ex + self.kv*ev + self.m*self.g*e3 - self.m*xddotd
		f = np.dot(thrust, np.matmul(R, e3))

		# roll, pitch, yaw = RToEuler(R)
		# b1c = np.array([np.cos(yaw), np.sin(yaw), 0])
		# b3c = -thrust/np.linalg.norm(thrust)
		# b2c = np.cross(b3c, b1c)

		# Rd = np.block([[b1c], [b2c], [b3c]])
		# Rdotd = Rd - self.prev_Rd
		# self.prev_Rd = Rd

		# whatc = np.matmul(np.transpose(Rd), Rdotd)
		# wc = vee(whatc)
		# wdotc = wc - self.prev_wc
		# self.prev_wc = wc

		# eR = 0.5 * vee((np.matmul(np.transpose(Rc), R) - np.matmul(np.transpose(R), Rc)))
		# ew = w - np.matmul(np.matmul(np.transpose(R), Rc), wc)

		# Rd = np.array([self.sp[6:9], self.sp[9:12], self.sp[12:15]])
		# eRhat = 0.5 * (np.matmul(np.transpose(Rd), R) - np.matmul(np.transpose(R), Rd))
		# eR = vee(eRhat)
		# ew = w - np.array(self.sp[15:18])
		# ew = w - np.matmul(np.matmul(np.transpose(R), Rd), wc)

		# term1 = np.matmul(np.matmul(np.matmul(hat(w), np.transpose(R)), Rd), wc)
		# term2 = np.matmul(np.matmul(np.transpose(R), Rd), wdotc)
		# big_term = np.matmul(self.J, term1 - term2)
		# M = -self.kR * eR - self.kw * ew + np.cross(w, np.matmul(self.J, w)) - big_term

		Rd = np.array([[sp[6], sp[9], sp[12]], [sp[7], sp[10], sp[13]], [sp[8], sp[11], sp[14]]])
		wd = sp[15:]
		wdotd = (next_sp[15:] - sp[15:])/self.dt
		eR = 0.5*vee((np.matmul(np.transpose(Rd), R) - np.matmul(np.transpose(R), Rd)))
		ew = w - np.matmul(np.matmul(np.transpose(R), Rd), wd)
		M = -self.kR * eR - self.kw * ew + np.cross(w, np.matmul(self.J, w)) - np.matmul(self.J, (np.matmul(np.matmul(np.matmul(hat(w), np.transpose(R)), Rd), wd) - np.matmul(np.matmul(np.transpose(R), Rd), wdotd)))

		self.prev_sp = sp

		return np.array([f, M[0], M[1], M[2]])

	def set_gains(self, gains):
		self.kx = gains[0]
		self.kv = gains[1]
		self.kR = gains[2]
		self.kw = gains[3]