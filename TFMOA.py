import time
import numpy as np


# Tomtit Flock Metaheuristic Optimization Algorithm (TFMOA)
def TFMOA(positions, fobj, VRmin, VRmax, max_iter):
    N, dim = positions.shape[0], positions.shape[1]
    lb = VRmin[0, :]
    ub = VRmax[0, :]

    Convergence_curve = np.zeros((max_iter, 1))

    best_fitness = np.zeros((N, 1))
    best_solution = float('inf')

    best_positions = positions.copy()

    t = 0
    ct = time.time()
    for t in range(max_iter):
        fitness = np.array([fobj(position) for position in positions])

        for j in range(N):
            if fitness[j] < best_fitness[j]:
                best_fitness[j] = fitness[j]
                best_positions[j] = positions[j].copy()

        # Find the global best position
        global_best_idx = np.argmin(best_fitness)
        global_best_position = best_positions[global_best_idx]

        # Update particle positions and velocities
        for j in range(N):
            r1, r2 = np.random.random(2)
            positions[j] = positions[j] + r1 * (best_positions[j] - positions[j]) + r2 * (
                        global_best_position - positions[j])
            positions[j] = np.clip(positions[j], lb, ub)

        Convergence_curve[t] = np.min(best_positions)


    best_solution = best_positions[0]
    best_fit = np.min(best_positions)
    ct = time.time() - ct

    return best_fit, Convergence_curve, best_solution, ct
