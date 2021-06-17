import sys
sys.path.append('../ilqr/ilqr')

import numpy as np
from ilqr.cost import QRCost

def iLQR_cost(goal, linear):
	if linear:
		state_size = 6 # [position, velocity]
		action_size = 3 # [acceleration]
	else:
		state_size = 18  # [position, velocity, orientation, angular velocity]
		action_size = 4  # [force, moment]

	# The coefficients weigh how much your state error is worth to you vs
	# the size of your controls. You can favor a solution that uses smaller
	# controls by increasing R's coefficient.
	Q = 10 * np.eye(state_size)
	R = 0.1 * np.eye(action_size)

	# This is optional if you want your cost to be computed differently at a
	# terminal state.
	Q_terminal = 0.1 * np.eye(state_size)
	Q_terminal[0:3, 0:3] = 1000*Q_terminal[0:3, 0:3]

	# State goal is set to a position of 0m with no velocity, 6m altitude, and neutral orientation.
	x, y, z = goal
	if linear:
		x_goal = np.array([x, y, z, 0.0, 0.0, 0.0])
	else:
		x_goal = np.array([x, y, z, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

	# NOTE: This is instantaneous and completely accurate.
	# print('set cost')
	cost = QRCost(Q, R, Q_terminal=Q_terminal, x_goal=x_goal)
	return cost