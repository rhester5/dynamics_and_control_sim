import numpy as np
import matplotlib.pyplot as plt


class MotionModel():
    def __init__(self, f, Q):
        self.f = f
        self.Q = Q

        (m, _) = Q.shape
        self.zero_mean = np.zeros(m)

    def __call__(self, x, u):
        new_state = self.f(x, u) + np.random.multivariate_normal(self.zero_mean, self.Q)
        return new_state


class MeasurementModel():
    def __init__(self, h, R):
        self.h = h
        self.R = R

        (n, _) = R.shape
        self.zero_mean = np.zeros(n)

    def __call__(self, x):
        measurement = self.h(x) + np.random.multivariate_normal(self.zero_mean, self.R)
        return measurement

class Controller():
    def __init__(self, sp, kp, kd):
        self.sp = sp
        self.kp = kp
        self.kd = kd

        self.last_error = np.array([0, 0])

    def __call__(self, x):
        error = self.sp - x
        p_error = np.array([error[1], error[3]])
        d_error = p_error-self.last_error
        self.last_error = p_error
        control = self.kp * p_error + self.kd * d_error
        return control

def create_model_parameters(T=1, s_r=0.0001, s_phi=0.00005, lambda_r=0.01, lambda_phi=0.001)#s_r=0.1, s_phi=0.01, lambda_r=0.3, lambda_phi=0.03):

    # F = np.array([[1, T],
    #               [0, 1]])
    # base_sigma = np.array([[T ** 3 / 3, T ** 2 / 2],
    #                        [T ** 2 / 2, T]])

    # sigma_x = s2_x * base_sigma
    # sigma_y = s2_y * base_sigma

    # zeros_2 = np.zeros((2, 2))
    # A = np.block([[F, zeros_2],
    #               [zeros_2, F]])
    # Q = np.block([[sigma_x, zeros_2],
    #               [zeros_2, sigma_y]])

    M = 5.972e24
    G = 6.674e-20#18

    s2_r = s_r**2
    s2_phi = s_phi**2
    lambda2_r = lambda_r**2
    lambda2_phi = lambda_phi**2

    def f(x, u):
        return np.array([x[0] + T * x[1],
                         x[1] + T * (-x[1] * x[3]**2 - G*M/x[0]**2 + u[0]), # should be -x[0] not x[1]... or so I thought? because -x[1] works but -x[0] is a mess
                         x[2] + T * x[3],
                         x[3] + T * (-2 * x[1] * x[3] / x[0] + u[1] / x[0])])

    def F(x, u):
        return np.array([[1, T, 0, 0],
                         [T*(-x[3]**2 + 2*G*M/x[0]**3), 1, 0, 2*x[0]*T*x[3]], # should be -x[3]**2 but positive also works
                         [0, 0, 1, T],
                         [T*(2*x[1]*x[3]/x[0]**2  - u[1] / x[0]**2), -2*T*x[3]/x[0], 0, 1 - 2*x[1]*T/x[0]]])

    base_sigma = np.array([[T ** 3 / 3, T ** 2 / 2],
                           [T ** 2 / 2, T]])

    sigma_r = s2_r * base_sigma
    sigma_phi = s2_phi * base_sigma

    zeros_2 = np.zeros((2, 2))
    Q = np.block([[sigma_r, zeros_2],
                  [zeros_2, sigma_phi]])

    # Measurement model parameters

    # H = np.array([[1, 0, 0, 0],
    #               [0, 0, 1, 0]])
    # R = lambda2 * np.eye(2)

    def h(x):
        # return x
        return np.array([x[0], x[2]])

    def H(x):
        # return np.eye(4)
        return np.array([[1, 0, 0, 0],
                         [0, 0, 1, 0]])

    # R = lambda2 * np.eye(2)
    # R = lambda2 * np.eye(4)
    R = np.array([[lambda2_r, 0], [0, lambda2_phi]])

    return f, F, h, H, Q, R


def simulate_system(K, x0, sp, kp, kd):
    (f, F, h, H, Q, R) = create_model_parameters()

    # Create models
    motion_model = MotionModel(f, Q)
    meas_model = MeasurementModel(h, R)
    controller = Controller(sp, kp, kd)

    (m, _) = Q.shape
    (n, _) = R.shape

    state = np.zeros((K, m))
    meas = np.zeros((K, n))

    # initial state
    x = x0
    for k in range(K):
        u = controller(x)
        x = motion_model(x, u)
        z = meas_model(x)

        state[k, :] = x
        meas[k, :] = z

    return state, meas

def polar_to_x(r, phi):
    return r * np.cos(phi)

def polar_to_y(r, phi):
    return r * np.sin(phi)


if __name__ == '__main__':
    np.random.seed(21)

    R = 6371
    setpoint = np.array([R + 35786, 0, 0, 2*np.pi/1440])#86400])
    Kp = 1
    Kd = 0.5
    (state, meas) = simulate_system(K=300, x0=setpoint, sp=setpoint, kp=Kp, kd=Kd)
    # see K=48 with Kp=1, Kd=0.1
    twopi = np.linspace(0, 2*np.pi, 20)
    r = np.ones((1, 20)) * R

    plt.figure(figsize=(7, 5))
    # plt.plot(polar_to_x(r, twopi), polar_to_y(r, twopi), '-go')
    plt.plot(polar_to_x(state[:, 0], state[:, 2]), polar_to_y(state[:, 0], state[:, 2]), '-bo')
    plt.plot(polar_to_x(meas[:, 0], meas[:, 1]), polar_to_y(meas[:, 0], meas[:, 1]), 'rx')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['true state', 'observed measurement'])
    plt.axis('square')
    plt.tight_layout(pad=0)
    plt.show()