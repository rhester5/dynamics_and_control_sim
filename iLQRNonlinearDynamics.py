import sys
sys.path.append('../ilqr/ilqr')

from ilqr import iLQR 
import theano.tensor as T
from ilqr.dynamics import AutoDiffDynamics
import numpy as np

def iLQR_nonlinear_dynamics(dt, m, g, J):
	# state
	x = T.dscalar("x")  # Position
	y = T.dscalar("y")  # Position
	z = T.dscalar("z")  # Position
	xdot = T.dscalar("x_dot")  # Velocity
	ydot = T.dscalar("y_dot")  # Velocity
	zdot = T.dscalar("z_dot")  # Velocity
	Rx1 = T.dscalar("Rx1")  # Rotation matrix
	Rx2 = T.dscalar("Rx2")  # Rotation matrix
	Rx3 = T.dscalar("Rx3")  # Rotation matrix
	Ry1 = T.dscalar("Ry1")  # Rotation matrix
	Ry2 = T.dscalar("Ry2")  # Rotation matrix
	Ry3 = T.dscalar("Ry3")  # Rotation matrix
	Rz1 = T.dscalar("Rz1")  # Rotation matrix
	Rz2 = T.dscalar("Rz2")  # Rotation matrix
	Rz3 = T.dscalar("Rz3")  # Rotation matrix
	wx = T.dscalar("omega_x")  # Angular velocity
	wy = T.dscalar("omega_y")  # Angular velocity
	wz = T.dscalar("omega_z")  # Angular velocity

	# input
	F = T.dscalar("F")  # Thrust
	M1 = T.dscalar("M1")  # Moment
	M2 = T.dscalar("M2")  # Moment
	M3 = T.dscalar("M3")  # Moment

	Ixx, Iyy, Izz = np.diag(J)

	# Discrete dynamics model definition
	f = T.stack([
	    x + dt * xdot,
		y + dt * ydot,
		z + dt * zdot,
		xdot + dt * (-F/m * Rz1),
		ydot + dt * (-F/m * Rz2),
		zdot + dt * (g - F/m * Rz3),
		Rx1 + dt * (Ry1 * wz - Rz1 * wy),
		Rx2 + dt * (Ry2 * wz - Rz2 * wy),
		Rx3 + dt * (Ry3 * wz - Rz3 * wy),
		Ry1 + dt * (Rz1 * wx - Rx1 * wz),
		Ry2 + dt * (Rz2 * wx - Rx2 * wz),
		Ry3 + dt * (Rz3 * wx - Rx3 * wz),
		Rz1 + dt * (Rx1 * wy - Ry1 * wx),
		Rz2 + dt * (Rx2 * wy - Ry2 * wx),
		Rz3 + dt * (Rx3 * wy - Ry3 * wx),
		wx + dt * (M1/Ixx - wy * wz * (Izz/Ixx - 1.0)),
		wy + dt * (M2/Iyy - wx * wz * (1.0 - Izz/Iyy)),
		wz + dt * (M3/Izz),
	])

	x_inputs = [x, y, z, xdot, ydot, zdot, Rx1, Rx2, Rx3, Ry1, Ry2, Ry3, Rz1, Rz2, Rz3, wx, wy, wz]  # State vector
	u_inputs = [F, M1, M2, M3]  # Control vector

	# Compile the dynamics
	# NOTE: This can be slow as it's computing and compiling the derivatives
	# But that's okay since it's only a one-time cost on startup
	print('compiling dynamics')
	dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)
	return dynamics