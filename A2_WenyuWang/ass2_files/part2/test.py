from gplearn.genetic import SymbolicRegressor
import numpy as np

x_in = np.array([-2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2, 2.25, 2.5])
y_out = np.array([37, 24.16016, 15.0625, 8.91016, 5, 2.72266, 1.5625, 1.09766, 1, 1.03516, 1.0625, 1.03516, 1, 1.09766, 1.5625,
         2.72266, 5, 8.91016, 15.0625])


if __name__ == "__main__":
    est_gp = SymbolicRegressor(population_size=100,
                               generations=40,
                               stopping_criteria=0.008,
                               p_crossover=0.5,
                               p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05,
                               p_point_mutation=0.1,
                               max_samples=0.9,
                               verbose=1,
                               parsimony_coefficient=0.01,
                               random_state=0)
    x_data = x_in.reshape(-1, 1)
    y_data = y_out.reshape(-1, 1).ravel()
    est_gp.fit(x_data, y_data)
    print(est_gp._program)