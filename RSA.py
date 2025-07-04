import time
import numpy as np
from mpmath import eps


def RSA(X, F_obj, LB, UB, T):
    N, Dim = X.shape[0], X.shape[1]
    Xnew = np.zeros((N, Dim))
    Conv = np.zeros((1, T))
    t = 1
    Alpha = 0.1
    Beta = 0.005
    Ffun = np.zeros((X.shape[1 - 1], 1))
    Ffun_new = np.zeros((X.shape[1 - 1], 1))
    Best_F = np.zeros((10, 1))
    Best_P = np.zeros((10, Dim))
    for i in range(N):
        Ffun[i, :] = F_obj(X[i, :])
        Best_F[i, :] = Ffun[i, :]
        Best_P[i, :] = X[i, :]
    ct = time.time()
    # while t < T + 1:
    for t in range(T):
        ES = 2 * np.random.rand() * (1 - (t / T))
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                R = Best_P[i, :] - X[np.random.randint(np.array(X.shape[0])), j] / ((Best_P[i, :]) + eps)
                P = Alpha + (X[i, :] - np.mean(X[i, :])) / (Best_P[i, :] * (UB[i, :] - LB[i, :]) + eps)
                Eta = Best_P[i, :] * P
                if (t < T / 4):
                    Xnew[i, :] = Best_P[i, :] - Eta * Beta - R * np.random.rand()
                else:
                    if (t < 2 * T / 4 and t >= T / 4):

                        Xnew[i, j] = Best_P[i, j] * X[np.random.randint(np.array(X.shape[1 - 1])),
                        j] * ES * np.random.rand()
                    else:
                        if (t < 3 * T / 4 and t >= 2 * T / 4):
                            Xnew[i, j] = Best_P[i, j] * P[j] * np.random.rand()
                        else:
                            Xnew[i, j] = Best_P[i, j] - Eta[j] * eps - R[j] * np.random.rand()
            Flag_UB = Xnew[i, :] > UB[i, j]
            Flag_LB = Xnew[i, :] < LB[i, j]
            Xnew[i, :] = (np.multiply(Xnew[i, :], ((Flag_UB[j] + Flag_LB[j])))) + np.multiply(UB[i, :], Flag_UB[j]) + np.multiply(
                LB[i, :], Flag_LB[j])
            Ffun_new[i, :] = F_obj(Xnew[i, :])
            if Ffun_new[i, :] < Ffun[i, :]:
                X[i, :] = Xnew[i, :]
                Ffun[i, :] = Ffun_new[i, :]
            if Ffun[i, :] < Best_F[i, :]:
                Best_F[i, :] = Ffun[i]
                Best_P[i, :] = X[i, :]
                Conv[0,t] = np.min(Best_F)
    ct = time.time() - ct
    best_fit = np.min(Best_F)
    best_pos = Best_P[0]
    return best_fit,Conv, best_pos, ct
