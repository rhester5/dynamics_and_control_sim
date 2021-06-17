import numpy as np
import matplotlib.pyplot as plt

class EKF():
    def __init__(self, f, F, h, H, Q, R, x_0, P_0):
        # Model parameters
        self.f = f
        self.F = F
        self.h = h
        self.H = H
        self.Q = Q
        self.R = R

        # Initial state
        self._x = x_0
        self._P = P_0

        # Monitoring
        self.F_list = []
        self.x_pre_list = []
        self.H_list = []
        self.S_list = []
        self.V_list = []
        self.K_list = []
        self.P_pre_list = []
        self.P_up_list = []

    def predict(self):
        Fk = self.F(self._x)
        self._x = self.f(self._x)
        self._P = np.matmul(Fk, np.matmul(self._P, Fk.transpose())) + self.Q
        self.x_pre_list.append(self._x[2])
        self.F_list.append(np.linalg.norm(Fk))
        self.P_pre_list.append(np.linalg.norm(self._P))

    def update(self, z):
        Hk = self.H(self._x)
        self.S = np.matmul(Hk, np.matmul(self._P, Hk.transpose())) + self.R
        self.V = z - self.h(self._x)

        # print(self._P, self.h(self._x), self.S, self.V, self.K)
        # print(self._P) 
        # print(self.S)
        # print(np.linalg.inv(self.S))
        self.K = np.matmul(self._P, np.matmul(Hk.transpose(), np.linalg.inv(self.S)))

        self._x = self._x + np.matmul(self.K, self.V)
        # self._P = self._P - np.matmul(self.K, np.matmul(self.S, self.K.transpose()))
        self._P = self._P - np.matmul(self.K, np.matmul(Hk, self._P))
        
        self.H_list.append(np.linalg.norm(Hk))
        self.S_list.append(np.linalg.norm(self.S))
        self.V_list.append(np.linalg.norm(self.V))
        self.K_list.append(np.linalg.norm(self.K))
        self.P_up_list.append(np.linalg.norm(self._P))

    def get_state(self):
        return self._x, self._P

    def plot(self):
        n = len(self.K_list)
        plt.figure(1)
        # plt.plot(range(n), self.F_list, 'o')
        # plt.figure(2)
        # plt.plot(range(n), self.x_pre_list, 'o')
        # plt.figure(3)
        # plt.plot(range(n), self.H_list, 'o')
        # plt.figure(4)
        # plt.plot(range(n), self.S_list, 'o')
        # plt.figure(5)
        # plt.plot(range(n), self.V_list, 'o')
        # plt.figure(6)
        plt.plot(range(n), self.K_list, 'o')
        # plt.figure(7)
        # plt.plot(range(n), self.P_pre_list, 'o')
        # plt.figure(8)
        # plt.plot(range(n), self.P_up_list, 'o')
        # plt.show()