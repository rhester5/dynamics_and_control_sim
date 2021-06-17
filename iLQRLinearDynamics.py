import sys
sys.path.append('../ilqr/ilqr')

from ilqr import iLQR 
import theano.tensor as T
from ilqr.dynamics import AutoDiffDynamics
import numpy as np

def iLQR_linear_dynamics(dt):
	# state
	x = T.dscalar("x")  # Position
	y = T.dscalar("y")  # Position
	z = T.dscalar("z")  # Position
	xdot = T.dscalar("x_dot")  # Velocity
	ydot = T.dscalar("y_dot")  # Velocity
	zdot = T.dscalar("z_dot")  # Velocity

	# input
	xdd = T.dscalar("xdd") # Acceleration
	ydd = T.dscalar("ydd") # Acceleration
	zdd = T.dscalar("zdd") # Acceleration

	# Discrete dynamics model definition
	f = T.stack([
	    x + dt * xdot + dt**2/2 * xdd,
		y + dt * ydot + dt**2/2 * ydd,
		z + dt * zdot + dt**2/2 * zdd,
		xdot + dt * xdd,
		ydot + dt * ydd,
		zdot + dt * zdd,
	])

	x_inputs = [x, y, z, xdot, ydot, zdot]  # State vector
	u_inputs = [xdd, ydd, zdd]  # Control vector

	# Compile the dynamics
	# NOTE: This can be slow as it's computing and compiling the derivatives
	# But that's okay since it's only a one-time cost on startup
	print('compiling dynamics')
	dynamics = AutoDiffDynamics(f, x_inputs, u_inputs)
	return dynamics