import time
import numpy as np

# Archimedes Optimization Algorithm (AOA)
def fun_checkpositions(dim, vec_pos, var_no_group, lb, ub):
    Lb = lb * np.ones(dim)
    Ub = ub * np.ones(dim)

    for i in range(var_no_group):
        isBelow1 = vec_pos[i, :] < Lb[i]
        isAboveMax = (vec_pos[i, :] > Ub[i])
        if isBelow1.any():
            vec_pos[i, :] = Lb[i]
        elif isAboveMax.any():
            vec_pos[i, :] = Ub[i]

    return vec_pos



def AOA(X, fobj, lb, ub, Max_iter):
    # Initialization
    Materials_no, dim = X.shape
    C1 = 2
    C2 = 6
    C3 = 0.5
    C4 = 0.2
    u = 0.9
    l = 0.1
    den = np.random.rand(Materials_no, dim)  # Eq. (5)
    vol = np.random.rand(Materials_no, dim)
    acc = lb + np.random.rand(Materials_no, dim) * (ub - lb)  # Eq. (6)

    ct = time.time()
    Y = np.zeros(Materials_no)
    for i in range(Materials_no):
        Y[i] = fobj(X[i, :])

    Scorebest, Score_index = min(Y), np.argmin(Y)
    Xbest = X[Score_index, :]
    den_best, vol_best, acc_best = den[Score_index, :], vol[Score_index, :], acc[Score_index, :]
    acc_norm = acc.copy()
    Xnew = X.copy()
    Convergence_curve = np.zeros(Max_iter)

    for t in range(Max_iter):
        TF = np.exp(((t - Max_iter) / Max_iter))  # Eq. (8)
        if TF > 1:
            TF = 1
        d = np.exp((Max_iter - t) / Max_iter) - (t / Max_iter)  # Eq. (9)
        acc = acc_norm
        r = np.random.rand()

        for i in range(Materials_no):
            den[i, :] += r * (den_best - den[i, :])  # Eq. (7)
            vol[i, :] += r * (vol_best - vol[i, :])

            if TF < 0.45:
                mr = np.random.randint(Materials_no)
                acc_temp = (den[mr, :] + (vol[mr, :] * acc[mr, :])) / (
                            np.random.rand() * den[i, :] * vol[i, :])  # Eq. (10)
            else:
                acc_temp = (den_best + (vol_best * acc_best)) / (np.random.rand() * den[i, :] * vol[i, :])  # Eq. (11)

            acc_norm[i, :] = ((u * (acc_temp - np.min(acc_temp))) / (
                        np.max(acc_temp) - np.min(acc_temp))) + l  # Eq. (12)

            for j in range(dim):
                if TF < 0.4:
                    mrand = np.random.randint(Materials_no)
                    Xnew[i, j] = X[i, j] + C1 * np.random.rand() * acc_norm[i, j] * (
                                X[mrand, j] - X[i, j]) * d  # Eq. (13)
                else:
                    p = 2 * np.random.rand() - C4  # Eq. (15)
                    T = C3 * TF
                    if T > 1:
                        T = 1
                    if p < 0.5:
                        Xnew[i, j] = Xbest[j] + C2 * np.random.rand() * acc_norm[i, j] * (
                                    T * Xbest[j] - X[i, j]) * d  # Eq. (14)
                    else:
                        Xnew[i, j] = Xbest[j] - C2 * np.random.rand() * acc_norm[i, j] * (T * Xbest[j] - X[i, j]) * d

        Xnew = fun_checkpositions(dim, Xnew, Materials_no, lb, ub)

        for i in range(Materials_no):
            v = fobj(Xnew[i, :])
            if v < Y[i]:
                X[i, :] = Xnew[i, :]
                Y[i] = v

        var_Ybest, var_index = min(Y), np.argmin(Y)
        Convergence_curve[t] = var_Ybest

        if var_Ybest < Scorebest:
            Scorebest = var_Ybest
            Score_index = var_index
            Xbest = X[var_index, :]
            den_best, vol_best, acc_best = den[var_index, :], vol[var_index, :], acc_norm[var_index, :]
    ct = time.time() - ct
    return Scorebest, Convergence_curve, Xbest, ct


