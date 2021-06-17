import numpy as np
from Utils import eulerToR, RToEuler

class NLGL():
	def __init__(self, trajectory, v_path, L):
		self.trajectory = trajectory
		self.gamma_prev = 0
		self.v_path = v_path # this could either be constant or changed based on the current position
		self.L = L

	def __call__(self, x, k):
		# current position p
		# current yaw psi
		# desired velocity of the vehicle on the body x-axis V_path
		# scalar parameter of guidance law L
		# gamma_dmin is the point on the path that is at a minimum distance to the locaiton of the vehicle
		# gamma_prev is previous gamma_dmin, initialized to 0 (or x0 would probably be better)
		# gamma_f is the last point on the trajectory
		p = x[0:3]
		R = np.array([[x[6], x[9], x[12]], [x[7], x[10], x[13]], [x[8], x[11], x[14]]])
		_, _, yaw = RToEuler(R)
		distances = np.linalg.norm(p-self.trajectory[self.gamma_prev:, 0:3], axis=1)
		dmin = np.amin(distances)
		gamma_dmin = np.argmin(distances) + self.gamma_prev
		if dmin > self.L:
			gamma_vtp = gamma_dmin
		else:
			gamma_vtp = gamma_dmin + np.argmin(np.abs(np.linalg.norm(p-self.trajectory[gamma_dmin:, 0:3], axis=1) - self.L))
		yaw_ref = np.arctan2(self.trajectory[gamma_vtp, 1] - p[1], self.trajectory[gamma_vtp, 0] - p[0])
		if np.abs(yaw_ref - yaw) > np.pi:
			yaw_ref = yaw + np.sign(yaw_ref-yaw) * np.pi
		# print(yaw_ref)
		z_ref = self.trajectory[gamma_dmin, 2]
		dvtp = np.linalg.norm(p[0:2] - self.trajectory[gamma_vtp, 0:2])
		v_ref = self.v_path * dvtp/self.L

		vx_ref = v_ref * np.cos(yaw_ref)
		vy_ref = v_ref * np.sin(yaw_ref)
		R_ref = eulerToR(0, 0, yaw_ref)

		setpoint = np.array([self.trajectory[gamma_vtp, 0], self.trajectory[gamma_vtp, 1], z_ref, vx_ref, vy_ref, 0, R_ref[0, 0], R_ref[1, 0], R_ref[2, 0], R_ref[0, 1], R_ref[1, 1], R_ref[2, 1], R_ref[0, 2], R_ref[1, 2], R_ref[2, 2], 0, 0, 0])
		# print(self.gamma_prev, gamma_vtp, p[0:2], setpoint[0:2], v_ref)
		self.gamma_prev = gamma_vtp
		# print(setpoint[0:3], self.trajectory[k, 0:3])

		return setpoint