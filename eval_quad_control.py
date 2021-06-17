import numpy as np
import matplotlib.pyplot as plt

from ekf_control import EKF
from sim_quad_model import GeoPosController, simulate_system, create_model_parameters

np.random.seed(21)

(f, F, h, H, Q, R) = create_model_parameters(T=1, sx=[0.1, 0.1, 0.05], sv=[0.05, 0.05, 0.01], sR=[0.1, 0.1, 0.05], sw=[0.05, 0.05, 0.01], lamx=[0.1], lamv=[0.1, 0.1, 0.05], lamR=[0.3, 0.3, 0.1], lamw=[0.1, 0.1, 0.05])
K = 120
x = np.array([0, 0, 3, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0])
sp = x

kx = .020
kv = .012
kR = .040
kw = .060

# a, b, c, d = [2, 20, 0.001, 0.05]
# a, b, c, d = [0.1, 12, 0.0005, 0.05]
# a, b, c, d = [0.1, 16, 0.0001, 0.03]
# a, b, c, d = [0.1, 16, 0.0005, 0.1]
# a, b, c, d = [0.6, 24, 0.0001, 0.03] # works
a, b, c, d = [1.5, 20, 0.0001, 0.03] # works
# a, b, c, d = [2, 16, 0.0001, 0.04]
# a, b, c, d = [4, 24, 0.0005, 0.03] 
gains = [a*kx, b*kv, c*kR, d*kw]

P = 0 * np.eye(18)

(state, meas) = simulate_system(K, x, sp, gains)
kalman_filter = EKF(f, F, h, H, Q, R, x, P)
sp = list(sp)
sp = sp + [0, 0, 0]
sp = np.array(sp)
controller = GeoPosController(sp, gains[0], gains[1], gains[2], gains[3])

est_state = np.zeros((K, 18))
est_cov = np.zeros((K, 18, 18))

for k in range(K):
    u = controller(x)

    kalman_filter.predict(u)
    kalman_filter.update(meas[k, :])
    (x, P) = kalman_filter.get_state()

    est_state[k, :] = x
    est_cov[k, ...] = P

t = np.linspace(0, K, K)

plt.figure(1)
plt.plot(state[:, 0], state[:, 1], '-bo')
plt.plot(est_state[:, 0], est_state[:, 1], '-ko')
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.legend(['true state', 'est state'])

plt.figure(2)
plt.plot(t, state[:, 2], '-bo')
plt.plot(t, est_state[:, 2], '-ko')
plt.plot(t, meas[:, 0], 'rx')
plt.xlabel('t [s]')
plt.ylabel('z [m]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.figure(3)
plt.plot(t, state[:, 3], '-bo')
plt.plot(t, est_state[:, 3], '-ko')
plt.plot(t, meas[:, 1], 'rx')
plt.xlabel('t [s]')
plt.ylabel('xdot [m/s]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.figure(4)
plt.plot(t, state[:, 4], '-bo')
plt.plot(t, est_state[:, 4], '-ko')
plt.plot(t, meas[:, 2], 'rx')
plt.xlabel('t [s]')
plt.ylabel('ydot [m/s]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.figure(5)
plt.plot(t, state[:, 5], '-bo')
plt.plot(t, est_state[:, 5], '-ko')
plt.plot(t, meas[:, 3], 'rx')
plt.xlabel('t [s]')
plt.ylabel('zdot [m/s]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.figure(6)
plt.plot(t, state[:, 15], '-bo')
plt.plot(t, est_state[:, 15], '-ko')
plt.plot(t, meas[:, 13], 'rx')
plt.xlabel('t [s]')
plt.ylabel('omega_x [rad/s]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.figure(7)
plt.plot(t, state[:, 16], '-bo')
plt.plot(t, est_state[:, 16], '-ko')
plt.plot(t, meas[:, 14], 'rx')
plt.xlabel('t [s]')
plt.ylabel('omega_y [rad/s]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.figure(8)
plt.plot(t, state[:, 17], '-bo')
plt.plot(t, est_state[:, 17], '-ko')
plt.plot(t, meas[:, 15], 'rx')
plt.xlabel('t [s]')
plt.ylabel('omega_z [rad/s]')
plt.legend(['true state', 'est state', 'observed measurement'])

plt.show()