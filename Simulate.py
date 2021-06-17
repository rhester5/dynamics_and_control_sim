import numpy as np

def simulate_system(K, x0, trajectory, controller, motion_model, meas_model, nlgl_on, nlgl):

	m = motion_model.get_state_size()
	n = meas_model.get_meas_size()

	state = np.zeros((K, m))
	meas = np.zeros((K, n))
	setpoint = np.zeros((K, m))

	# initial state
	x = x0
	for k in range(K):
		kp1 = k+1 if k < K-1 else k
		if nlgl_on:
			spp = nlgl(x, k)
			sp = trajectory[k]
			sp[0:3] = spp[0:3]
		else:
			sp = trajectory[k]
		u = controller(x, sp, trajectory[kp1], k)

		if k == 0:
			control = np.zeros((K, len(u)))
		
		# u = actuator_limits(u, g, mass, d, c, J)
		x = motion_model(x, u)
		# x = state_limits(x)
		z = meas_model(x)

		state[k, :] = x
		meas[k, :] = z
		setpoint[k, :] = trajectory[k] # sp
		control[k, :] = u

	return state, meas, setpoint, control