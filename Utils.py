import numpy as np

def wrapToPi(angle):
        # https://stackoverflow.com/questions/27093704/converge-values-to-range-pi-pi-in-matlab-not-using-wraptopi
        
        angle_wrapped = angle - 2*np.pi * np.floor((angle + np.pi) / (2*np.pi))
        
        return angle_wrapped

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

def hover_traj(x, y, z, K, linear):
	if linear:
		trajectory = np.zeros((K, 6))
		trajectory[:, :] = [x, y, z, 0, 0, 0]
	else:
		trajectory = np.zeros((K, 18))
		trajectory[:, :] = [x, y, z, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
	return trajectory

def step_response(trajectory, i, step, begin, end):
	trajectory[begin:end, i] = trajectory[begin:end, i] + step
	return trajectory

def circle_traj(r, z, K, linear):
	theta = np.linspace(0, 2*np.pi, K)
	x = r * np.cos(theta)
	y = r * np.sin(theta)

	if linear:
		trajectory = np.zeros((K, 6))
	else:
		trajectory = np.zeros((K, 18))
		trajectory[:, 6] = 1
		trajectory[:, 10] = 1
		trajectory[:, 14] = 1

	for i in range(K):
		trajectory[i, 0] = x[i]
		trajectory[i, 1] = y[i]
	trajectory[:, 2] = z
	return trajectory

def line_traj(x_or_y, z, begin, end, K, linear):
	if linear:
		trajectory = np.zeros((K, 6))
	else:
		trajectory = np.zeros((K, 18))
		trajectory[:, 6] = 1
		trajectory[:, 10] = 1
		trajectory[:, 14] = 1

	line = np.linspace(begin, end, K)
	if x_or_y == 'x':
		for i in range(K):
			trajectory[i:i, 0] = line[i+1]
	elif x_or_y == 'y':
		for i in range(K):
			trajectory[i:i, 1] = line[i+1]
	trajectory[:, 2] = z

	return trajectory

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