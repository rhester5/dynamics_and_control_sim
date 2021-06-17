import numpy as np
import matplotlib.pyplot as plt

# from extended_kalman_filter import EKF
from ekf_control import EKF
# from simulate_satellite_model import simulate_system, create_model_parameters
from sim_sat_model_control import Controller, simulate_system, create_model_parameters

np.random.seed(21)

# sr = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
# sphi = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
# lamr = [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]
# lamphi = [0.0001, 0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

# sr = [0.001, 0.005, 0.01, 0.05, 0.1]
# sphi = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
# lamr = [0.001, 0.005, 0.01, 0.05, 0.1]
# lamphi = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]

sr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
sphi = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
lamr = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
lamphi = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

r_error_mat = np.zeros((len(sr), len(sphi), len(lamr), len(lamphi)))
phi_error_mat = np.zeros((len(sr), len(sphi), len(lamr), len(lamphi)))

for a in range(len(sr)):
    for b in range(len(sphi)):
        for c in range(len(lamr)):
            for d in range(len(lamphi)):
                print(sr[a], sphi[b], lamr[c], lamphi[d])
                (f, F, h, H, Q, R) = create_model_parameters(s_r=sr[a], s_phi=sphi[b], lambda_r=lamr[c], lambda_phi=lamphi[d])
                K = 50
                # initial state
                R = 6371
                x = np.array([R + 35786, 0, 0, 2*np.pi/86400])
                kp = 1
                kd = 0.5
                P = 0 * np.eye(4)

                # (state, meas) = simulate_system(K, x)
                (state, meas) = simulate_system(K, x, x, kp, kd)
                kalman_filter = EKF(f, F, h, H, Q, R, x, P)
                controller = Controller(x, kp, kd)

                est_state = np.zeros((K, 4))
                est_cov = np.zeros((K, 4, 4))
                r_error = np.zeros((K, 1))
                phi_error = np.zeros((K, 1))
                try:
                    for k in range(K):
                        u = controller(x)
                        kalman_filter.predict(u)
                        kalman_filter.update(meas[k, :])
                        (x, P) = kalman_filter.get_state()
                        est_state[k, :] = x
                        est_cov[k, ...] = P
                        r_err = np.abs(x[0] - state[k, 0])
                        if r_err > 500:
                            # print('r trash')
                            r_error[k, 0] = np.inf
                        else:
                            r_error[k, 0] = r_err

                        phi_err = np.abs(x[2] - state[k, 2])
                        if phi_err > 2*np.pi*5/360:
                            # print('phi trash')
                            phi_error[k, 0] = np.inf
                        else:
                            phi_error[k, 0] = phi_err

                    avg_r_error = np.sum(r_error)/K
                    avg_phi_error = np.sum(phi_error)/K
                except:
                    avg_r_error = np.inf
                    avg_phi_error = np.inf

                r_error_mat[a, b, c, d] = avg_r_error
                phi_error_mat[a, b, c, d] = avg_phi_error
i = 0
while i < 50:
    min_r_error = np.nanmin(r_error_mat)
    min_r_flat_index = np.argmin(r_error_mat)
    min_r_index = np.unravel_index(min_r_flat_index, (len(sr), len(sphi), len(lamr), len(lamphi)))
    a, b, c, d = min_r_index
    print(min_r_error)
    print(sr[a], sphi[b], lamr[c], lamphi[d])
    r_error_mat[a, b, c, d] = np.inf

    min_phi_error = np.nanmin(phi_error_mat)
    min_phi_flat_index = np.argmin(phi_error_mat)
    min_phi_index = np.unravel_index(min_phi_flat_index, (len(sr), len(sphi), len(lamr), len(lamphi)))
    a, b, c, d = min_phi_index
    print(min_phi_error)
    print(sr[a], sphi[b], lamr[c], lamphi[d])
    phi_error_mat[a, b, c, d] = np.inf

    i+=1