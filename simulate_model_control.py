import numpy as np
import matplotlib.pyplot as plt


class MotionModel():
    def __init__(self, A, B, Q):
        self.A = A
        self.B = B
        self.Q = Q

        (m, _) = Q.shape
        self.zero_mean = np.zeros(m)

    def __call__(self, x, u):
        new_state = np.matmul(self.A, x) + np.matmul(self.B, u) + np.random.multivariate_normal(self.zero_mean, self.Q)
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
    def __init__(self, sp, kp, kd):
        self.sp = sp
        self.kp = kp
        self.kd = kd

        self.last_error = np.array([0, 0])

    def __call__(self, x):
        error = self.sp - x
        v_error = np.array([error[1], error[3]])
        d_error = v_error-self.last_error
        self.last_error = v_error
        control = self.kp * v_error + self.kd * d_error
        return control

def create_model_parameters(T=1, s2_x=0.1 ** 2, s2_y=0.1 ** 2, lambda2=0.3 ** 2):
    # Motion model parameters
    F = np.array([[1, T],
                  [0, 1]])
    base_sigma = np.array([[T ** 3 / 3, T ** 2 / 2],
                           [T ** 2 / 2, T]])

    sigma_x = s2_x * base_sigma
    sigma_y = s2_y * base_sigma

    zeros_2 = np.zeros((2, 2))
    A = np.block([[F, zeros_2],
                  [zeros_2, F]])
    B = np.array([[0, 0], [1, 0], [0, 0], [0, 1]])
    Q = np.block([[sigma_x, zeros_2],
                  [zeros_2, sigma_y]])

    # Measurement model parameters
    H = np.array([[1, 0, 0, 0],
                  [0, 0, 1, 0]])
    R = lambda2 * np.eye(2)

    return A, B, H, Q, R


def simulate_system(K, x0, sp, kp, kd):
    (A, B, H, Q, R) = create_model_parameters()

    # Create models
    motion_model = MotionModel(A, B, Q)
    meas_model = MeasurementModel(H, R)
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


if __name__ == '__main__':
    np.random.seed(21)
    setpoint = np.array([0, 0.1, 0, 0.1])
    Kp = 1
    Kd = 0.1
    (state, meas) = simulate_system(K=20, x0=np.array([0, 0.1, 0, 0.1]), s0=setpoint, kp=Kp, kd=Kd)

    plt.figure(figsize=(7, 5))
    plt.plot(state[:, 0], state[:, 2], '-bo')
    plt.plot(meas[:, 0], meas[:, 1], 'rx')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.legend(['true state', 'observed measurement'])
    plt.axis('square')
    plt.tight_layout(pad=0)
    plt.show()