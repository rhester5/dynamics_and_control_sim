import numpy as np
import matplotlib.pyplot as plt

from extended_kalman_filter import EKF
from simulate_satellite_model import simulate_system, create_model_parameters

np.random.seed(21)

(f, F, h, H, Q, R) = create_model_parameters(s_r=0.0001, s_phi=0.00005, lambda_r=0.01, lambda_phi=0.001)
K = 10
# initial state
R = 6371
x = np.array([R + 35786, 0, 0, 2*np.pi/1440])#86400])
P = 0 * np.eye(4)

(state, meas) = simulate_system(K, x)
kalman_filter = EKF(f, F, h, H, Q, R, x, P)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))
error = np.zeros((K, 4))

for k in range(K):
    kalman_filter.predict()
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()
    # print(x[2], state[k, 2], meas[k, 1])
    # print()
    # print(x[1], state[k, 1])
    print('r: ', x[0], state[k, 0])
    print('rdot: ', x[1], state[k, 1])
    print('phi: ', x[2]*180/np.pi, state[k, 2]*180/np.pi)
    print('phidot: ', x[3]*180/np.pi, state[k, 3]*180/np.pi)
    print
    est_state[k, :] = x
    est_cov[k, ...] = P
    error[k, :] = x - state[k, :]

def polar_to_x(r, phi):
    return r * np.cos(phi)

def polar_to_y(r, phi):
    return r * np.sin(phi)

avg_error = np.sum(error)/K

# # kalman_filter.plot()
# # plt.figure(9)
# plt.figure(1)
# plt.plot(range(K), state[:, 2], 'o')
# # plt.figure(10)
# plt.plot(range(K), est_state[:, 2], 'o')
# # plt.figure(11)
# plt.plot(range(K), meas[:, 1], 'o')
# # plt.legend(['F', 'infer_pre', 'H', 'S', 'V', 'K', 'P_pre', 'P_up', 'state', 'infer_up']) #, 'meas'])
# # plt.legend(['infer_pre', 'K', 'P_pre', 'P_up', 'state', 'infer_up'])
# # plt.legend(['K', 'state', 'infer_up'])
# plt.legend(['state', 'infer', 'meas'])
# plt.show()

plt.figure(figsize=(7, 5))
plt.plot(polar_to_x(state[:, 0], state[:, 2]), polar_to_y(state[:, 0], state[:, 2]), '-bo')
plt.plot(polar_to_x(est_state[:, 0], est_state[:, 2]), polar_to_y(est_state[:, 0], est_state[:, 2]), '-ko')
plt.plot(polar_to_x(meas[:, 0], meas[:, 1]), polar_to_y(meas[:, 0], meas[:, 1]), 'rx')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state', 'inferred state', 'observed measurement'])
plt.axis('square')
plt.tight_layout(pad=0)
plt.plot()
plt.show()