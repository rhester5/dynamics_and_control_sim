import numpy as np

class PIDPosController():
	def __init__(self, kp, kd, ki, linear):
		self.kp = kp
		self.kd = kd
		self.ki = ki
		self.linear = linear

		self.total_error = np.zeros(3)

	def __call__(self, x, sp, next_sp, k):
		error = sp - x
		p_error = error[0:3]
		d_error = error[3:6]
		i_error = p_error+self.total_error
		# i_error = np.zeros(3)

		# TODO this is supposed to prevent integrator windup
		# if np.linalg.norm(self.total_error) < 5:
		# 	i_error = p_error+self.total_error
		# else:
		# 	i_error = np.zeros(3) # self.total_error
		
		self.total_error = i_error
		control = self.kp * p_error + self.kd * d_error + self.ki * i_error
		# TODO WHAT THE FUCK
		if not self.linear:
			control[0], control[1] = -control[1], control[0]
			control[2] = -control[2]

		return control

	def set_gains(self, gains):
		self.kp = gains[0]
		self.kd = gains[1]
		self.ki = gains[2]