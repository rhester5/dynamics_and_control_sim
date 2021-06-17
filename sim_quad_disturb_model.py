import numpy as np
import matplotlib.pyplot as plt


class MotionModel():
    def __init__(self, A, B, Q):
        self.A = A
        self.B = B
        self.Q = Q

        (m, _) = Q.shape
        self.zero_mean = np.zeros(m)

    def __call__(self, x, u, yaw):
        new_state = np.matmul(self.A(yaw), x) + np.matmul(self.B, u) + np.random.multivariate_normal(self.zero_mean, self.Q)
        if new_state[2] < 0:
            new_state[2] = 0
        return new_state


class MeasurementModel():
    def __init__(self, H, R):
        self.H = H
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = np.matmul(self.H, x) + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement

class Controller():
    def __init__(self, sp, kp, kd, ki, kf=False):
        self.sp = sp
        self.kp = kp
        self.kd = kd
        self.ki = ki
        self.kf = kf
        self.g = 9.81

        # self.last_error = np.array([0, 0, 0])
        self.total_error = np.array([0, 0, 0])

    def __call__(self, x, yaw, roll_comp, pitch_comp, thrust_comp):
        error = self.sp - x
        p_error = error[0:3]
        d_error = error[3:6] 
        # d_error = p_error - self.last_error
        # self.last_error = p_error
        self.total_error = self.total_error + p_error
        control = self.kp * p_error + self.kd * d_error + self.ki * self.total_error
        if self.kf:
            control = control + np.array([self.g*(roll_comp * np.sin(yaw) + pitch_comp * np.cos(yaw)), self.g*(-roll_comp * np.cos(yaw) + pitch_comp * np.sin(yaw)), thrust_comp])
        return control

def create_model_parameters(T=0.01):

    g = 9.81

    # state = [x, y, z, xdot, ydot, zdot, roll_comp, pitch_comp, thrust_comp]
    def A(yaw):
        M = np.zeros((3, 3))
        # M = np.array([[np.sin(yaw), np.cos(yaw), 0], [-np.cos(yaw), np.sin(yaw), 0], [0, 0, 1]])
        A = np.block([[np.eye(3), T*np.eye(3), -g * T**2/2 * M],
                      [np.zeros((3, 3)), np.eye(3), -g * T * M],
                      [np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3)]])
        return A
    B = np.block([[T**2/2 * np.eye(3)], [T * np.eye(3)], [np.zeros((3, 3))]])

    Qpos = 1e-4
    Qvel = 1e-5
    Qdis = 1e-5
    Q = np.diag([Qpos, Qpos, Qpos, Qvel, Qvel, Qvel, Qdis, Qdis, Qdis])
    Q = 0*np.eye(9)
    # Measurement model parameters

    H = np.block([[np.eye(6), np.zeros((6, 3))]])

    Rpos = 1e-1
    Rvel = 1e-1
    R = np.diag([Rpos, Rpos, Rpos, Rvel, Rvel, Rvel])
    R = 0*np.eye(6)
    
    return A, B, H, Q, R


def simulate_system(K, x0, sp, kp, kd, ki):
    A, B, H, Q, R = create_model_parameters()

    # Create models
    motion_model = MotionModel(A, B, Q)
    meas_model = MeasurementModel(H, R)
    controller = Controller(sp, kp, kd, ki)

    (m, _) = Q.shape
    (n, _) = R.shape

    state = np.zeros((K, m))
    meas = np.zeros((K, n))
    setpoint = np.zeros((K, m))

    # initial state
    x = x0
    for k in range(K):
        if k in [500, 1000, 1500, 2000, 2500]:
            sp += np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]) 
            controller = Controller(sp, kp, kd, ki)
        u = controller(x, 0, x[6], x[7], x[8])
        x = motion_model(x, u, 0)
        z = meas_model(x)

        state[k, :] = x
        meas[k, :] = z
        setpoint[k, :] = sp

    return state, meas, setpoint


if __name__ == '__main__':
    np.random.seed(21)

    setpt = np.array([0, 0, 5, 0, 0, 0, 0, 0, 0])
    Kp = 4
    Kd = 4
    Ki = 0.01
    k = 3000
    (state, meas, setpoint) = simulate_system(K=k, x0=setpt, sp=setpt, kp=Kp, kd=Kd, ki=Ki)
    t = np.linspace(0, k, k)

    plt.figure(figsize=(7, 5))
    plt.plot(state[:, 0], state[:, 1], meas[:, 0], meas[:, 1], setpoint[:, 0], setpoint[:, 1])
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['true state', 'observed measurement', 'setpoint'])
    plt.axis('square')
    plt.tight_layout(pad=0)

    plt.figure()
    plt.plot(t, state[:, 0], t, setpoint[:, 0])
    plt.xlabel('t [s]')
    plt.ylabel('x [m]')
    plt.legend(['x position', 'x setpoint'])
    plt.tight_layout(pad=0)

    plt.figure()
    plt.plot(t, state[:, 1], t, setpoint[:, 1])
    plt.xlabel('t [s]')
    plt.ylabel('y [m]')
    plt.legend(['y position', 'y setpoint'])
    plt.tight_layout(pad=0)

    plt.figure()
    plt.plot(t, state[:, 2], t, setpoint[:, 2])
    plt.xlabel('t [s]')
    plt.ylabel('z [m]')
    plt.legend(['altitude', 'z setpoint'])
    plt.tight_layout(pad=0)

    plt.figure()
    plt.plot(t, state[:, 6], t, state[:, 7], t, state[:, 8])
    plt.xlabel('t [s]')
    plt.ylabel('disturbance compensation')
    plt.legend(['roll_comp', 'pitch_comp', 'thrust_comp'])
    plt.tight_layout(pad=0)

    plt.show()