import numpy as np
import matplotlib.pyplot as plt

# from kalman_filter import KalmanFilter
from kalman_filter_control import KalmanFilter
from simulate_model_control import Controller, simulate_system, create_model_parameters

np.random.seed(21)
(A, B, H, Q, R) = create_model_parameters()
K = 40
# initial state
x = np.array([0, 0.1, 0, 0.1])
setpoint = x
Kp = 1
Kd = 0.1
P = 0 * np.eye(4)

(state, meas) = simulate_system(K, x, setpoint, Kp, Kd)
# kalman_filter = KalmanFilter(A, H, Q, R, x, P)
kalman_filter = KalmanFilter(A, B, H, Q, R, x, P)
controller = Controller(setpoint, Kp, Kd)

est_state = np.zeros((K, 4))
est_cov = np.zeros((K, 4, 4))
dead_reckoning = np.zeros((K, 4))

for k in range(K):
    u = controller(x)

    kalman_filter.predict(u)
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P
    if k > 0:
        dead_reckoning[k, :] = dead_reckoning[k-1, :] + np.array([0.1, 0, 0.1, 0])
    else:
        dead_reckoning[k, :] = np.array([0.1, 0.1, 0.1, 0.1])

plt.figure(figsize=(7, 5))
plt.plot(state[:, 0], state[:, 2], '-bo')
plt.plot(est_state[:, 0], est_state[:, 2], '-ko')
plt.plot(meas[:, 0], meas[:, 1], ':rx')
plt.plot(dead_reckoning[:, 0], dead_reckoning[:, 2], '-go')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state', 'inferred state', 'observed measurement', 'dead reckoning'])
plt.axis('square')
plt.tight_layout(pad=0)
plt.plot()
plt.show()