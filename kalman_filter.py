# https://towardsdatascience.com/wtf-is-sensor-fusion-part-2-the-good-old-kalman-filter-3642f321440

import numpy as np


class KalmanFilter():
    def __init__(self, A, H, Q, R, x_0, P_0):
        # Model parameters
        self.A = A
        self.H = H
        self.Q = Q
        self.R = R

        # Initial state
        self._x = x_0
        self._P = P_0

    def predict(self):
        self._x = np.matmul(self.A, self._x)
        self._P = np.matmul(self.A, np.matmul(self._P, self.A.transpose())) + self.Q

    def update(self, z):
        self.S = np.matmul(self.H, np.matmul(self._P, self.H.transpose())) + self.R
        self.V = z - np.matmul(self.H, self._x)
        self.K = np.matmul(self._P, np.matmul(self.H.transpose(), np.linalg.inv(self.S)))

        self._x = self._x + np.matmul(self.K, self.V)
        #self._P = self._P - np.matmul(self.K, np.matmul(self.S, self.K.transpose()))
        self._P = self._P - np.matmul(self.K, np.matmul(self.H, self._P))

    def get_state(self):
        return self._x, self._P