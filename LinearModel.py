import numpy as np

def create_linear_model_parameters(T, model_cov, meas_cov):

    # state = [x, y, z, xdot, ydot, zdot, roll_comp, pitch_comp, thrust_comp]
    A = np.block([[np.eye(3), T*np.eye(3)],
                  [np.zeros((3, 3)), np.eye(3)]])
    B = np.block([[T**2/2 * np.eye(3)], [T * np.eye(3)]])

    Qpos = model_cov[0]
    Qvel = model_cov[1]
    Q = np.diag([Qpos, Qpos, Qpos, Qvel, Qvel, Qvel])

    # Measurement model parameters

    H = np.eye(6)

    Rpos = meas_cov[0]
    Rvel = meas_cov[1]
    R = np.diag([Rpos, Rpos, Rpos, Rvel, Rvel, Rvel])

    return A, B, H, Q, R