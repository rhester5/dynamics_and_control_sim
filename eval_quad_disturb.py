import numpy as np
import matplotlib.pyplot as plt

# from kalman_filter import KalmanFilter
from kalman_filter_quad_control import KalmanFilter
from sim_quad_disturb_model import Controller, simulate_system, create_model_parameters

np.random.seed(21)
(A, B, H, Q, R) = create_model_parameters()
K = 3000
# initial state
x = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0])
sp = x
Kp = 4
Kd = 4
Ki = 0.01
P = 1e-3 * np.eye(9)

(state, meas, _) = simulate_system(K, x, sp, Kp, Kd, Ki)

x = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0])
sp = x

kalman_filter = KalmanFilter(A, B, H, Q, R, x, P)
controller = Controller(sp, Kp, Kd, Ki, kf=True)

est_state = np.zeros((K, 9))
est_cov = np.zeros((K, 9, 9))
setpoint = np.zeros((K, 9))

for k in range(K):
    if k in [500, 1000, 1500, 2000, 2500]:
        sp += np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]) 
        controller = Controller(sp, Kp, Kd, Ki, kf=True)
    u = controller(x, 0, x[6], x[7], x[8])

    kalman_filter.predict(u, 0)
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P
    setpoint[k, :] = sp

t = np.linspace(0, K, K)

plt.figure(figsize=(7, 5))
plt.plot(meas[:, 0], meas[:, 1], ':rx')
plt.plot(state[:, 0], state[:, 1], '-b')
plt.plot(setpoint[:, 0], setpoint[:, 1], '-g')
plt.plot(est_state[:, 0], est_state[:, 1], '-k')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['observed measurement', 'true state', 'setpoint', 'inferred state'])
plt.axis('square')
plt.tight_layout(pad=0)

plt.figure()
plt.plot(t, meas[:, 0], ':rx')
plt.plot(t, state[:, 0], '-b')
plt.plot(t, setpoint[:, 0], '-g')
plt.plot(t, est_state[:, 0], '-k')
plt.xlabel('t [s]')
plt.ylabel('x [m]')
plt.legend(['x meas', 'x position', 'x setpoint', 'inferred x'])
plt.tight_layout(pad=0)

plt.figure()
plt.plot(t, meas[:, 1], ':rx')
plt.plot(t, state[:, 1], '-b')
plt.plot(t, setpoint[:, 1], '-g')
plt.plot(t, est_state[:, 1], '-k')
plt.xlabel('t [s]')
plt.ylabel('y [m]')
plt.legend(['y meas', 'y position', 'y setpoint', 'inferred y'])
plt.tight_layout(pad=0)

plt.figure()
plt.plot(t, meas[:, 2], ':rx')
plt.plot(t, state[:, 2], '-b')
plt.plot(t, setpoint[:, 2], '-g')
plt.plot(t, est_state[:, 2], '-k')
plt.xlabel('t [s]')
plt.ylabel('z [m]')
plt.legend(['z meas', 'z position', 'z setpoint', 'inferred z'])
plt.tight_layout(pad=0)

plt.figure()
plt.plot(t, state[:, 6], t, state[:, 7], t, state[:, 8], t, est_state[:, 6], t, est_state[:, 7], t, est_state[:, 8])
plt.xlabel('t [s]')
plt.ylabel('disturbance compensation')
plt.legend(['roll_comp', 'pitch_comp', 'thrust_comp', 'inferred roll', 'inferred pitch', 'inferred thrust'])
plt.tight_layout(pad=0)

plt.show()