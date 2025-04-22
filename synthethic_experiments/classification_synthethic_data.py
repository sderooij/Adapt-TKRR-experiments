import numpy as np
from generate_data import generate_negative_data, generate_positive_data
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorlibrary.learning.features import kernel_mat_features
from sklearn.metrics.pairwise import rbf_kernel
from tensorlibrary.learning import CPKRR, CPKRR_Adapt
from plot_functions import plot_decision_boundary
from copy import deepcopy
from functools import partial
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# Parameters
SEED = 0 # seed for reproducibility
FILE_TYPE = 'pdf'
positive_samples = 100
negative_samples = 500
covariance = [[0.01, 0], [0, 0.01]]
bounds = [(-1, -1), (1, 1)]
exclusion_radius = 0.25

import matplotlib

# matplotlib.use("pgf")
matplotlib.use('TkAgg')

matplotlib.rcParams.update(
    {
        'font.size': 20,
        # 'text.usetex': True,
        # 'pgf.rcfonts': False,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
    }
)

# Centers for the Gaussian components for target and source
centers_s = [(-0.4, 0.5), (0.5, 0.7), (-0.1, -0.6)]
centers_t = [(-0.4, 0.3), (0.5, 0.3), (0, -0.65)]

# Generate data
np.random.seed(SEED)
positive_s = generate_positive_data(centers_s, covariance, positive_samples)
positive_s = np.vstack(positive_s)
negative_s = generate_negative_data(bounds, centers_s, exclusion_radius, negative_samples)
X_s = np.vstack((positive_s, negative_s))
y_s = np.array([1] * positive_samples + [-1] * negative_samples)

positive_t = generate_positive_data(centers_t, covariance, positive_samples)

positive_t = np.vstack(positive_t)
# pos_selected_samples = np.vstack(pos_selected_samples)
negative_t = generate_negative_data(bounds, centers_t, exclusion_radius, negative_samples)
X_t = np.vstack((positive_t, negative_t))
y_t = np.array([1] * positive_samples + [-1] * negative_samples)

# Sample target data
pos_indices = np.random.choice(positive_samples, 3)
pos_selected_samples = positive_t[pos_indices,:]
neg_indices = np.random.choice(negative_samples, 17, replace=False)
neg_selected_samples = negative_t[neg_indices,:]
X_t_selected = np.vstack((pos_selected_samples, neg_selected_samples))
y_t_selected = np.array([1] * 3 + [-1] * 17)

# train CPKRR model
gam = 5
sig = np.sqrt(1/(2*gam))
# --------- UNCOMMENT TO DETERMINE M AND L PARAMS ------------
# M_grid = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# L_grid = [1, 1.25, 1.5, 1.75, 2., 2.25, 2.5]
# full_grid = [(m, L) for m in M_grid for L in L_grid]
# K_rbf = rbf_kernel(X_s, X_s, gamma=gam)
# K_rbf = K_rbf[np.triu_indices_from(K_rbf, k=0)]
# # create dataframe to store results, where M is index and L's are columns
# ML_results = pd.DataFrame(columns=L_grid, index=M_grid)
#
# for (m, L) in full_grid:
#     K_phi = kernel_mat_features(X_s, X_s, m=m, feature_map="rbf", map_param=sig, Ld=L)
#     K_phi = K_phi[np.triu_indices_from(K_phi, k=0)]
#     err = np.linalg.norm(K_rbf - K_phi, 2)/np.linalg.norm(K_rbf, 2)
#     ML_results.loc[m, L] = err
#
# # %% print params in full
# print(ML_results.to_markdown())
# # save to file with index and column names
# ML_results.to_csv('hyperparams_CPKRR_synthetic_source.csv', index=True)
M = 14
Ld = 1.75
R = 4
reg_par = 0.001
# -----------------------------------------------------------
cp_model = CPKRR(
    feature_map='rbf',
    map_param=sig,
    M=M,
    num_sweeps=10,
    Ld=Ld,
    class_weight="balanced",
    mu=0,
    max_rank=R,
    reg_par=reg_par,
    train_loss_flag=True,
)

# Set up grid search for CPKRR
# reg_params = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]
# Rs = [4]
# param_grid = {'reg_par': reg_params, 'max_rank': Rs}
# from sklearn.model_selection import GridSearchCV
# cv_model = GridSearchCV(cp_model, param_grid, cv=5, n_jobs=-1, refit=False, return_train_score=True)
# cv_model.fit(X_s, y_s)
#
# print(f"Best parameters: {cv_model.best_params_}")
#
# # print test scores per parameter combination with the parameters
#
# test_scores = cv_model.cv_results_['mean_test_score']
# parameters = cv_model.cv_results_['params']
# for i, param in enumerate(parameters):
#     print(f"Parameters: {param}, Test score: {test_scores[i]}")
# # extract best parameters
# best_params = cv_model.best_params_
# R = best_params['max_rank']
# reg_par = best_params['reg_par']

# train source model with best parameters
source_model = deepcopy(cp_model)
# source_model.set_params(**best_params)
source_model.fit(X_s, y_s)

# train target model with best parameters
target_model = deepcopy(cp_model)
# target_model.set_params(**best_params)
target_model.fit(X_t_selected, y_t_selected)

# train adapted model
mu_grid = [1e-6,  1e-4,  1e-2, 1]

# train adapted model for each mu
adapted_models = []
tl_model = CPKRR(
    w_init=deepcopy(source_model.weights_),
    feature_map='rbf',
    map_param=sig,
    M=M,
    max_rank=R,
    num_sweeps=10,
    reg_par=0,
    Ld=Ld,
    class_weight="balanced",
    random_init=True,
    train_loss_flag=True,
)
f1_scores = []
for mu in mu_grid:
    adapt_model = deepcopy(tl_model)
    adapt_model.set_params(mu=mu)
    adapt_model.fit(X_t_selected, y_t_selected)
    adapted_models.append(adapt_model)
    y_pred = adapt_model.predict(X_t)
    f1 = f1_score(y_t, y_pred)
    f1_scores.append(f1)

# determine best adapted model
idx = np.argmax(f1_scores)
mu_best = mu_grid[idx]
best_adapted_model = adapted_models[idx]

# save f1 scores to file
f1_scores = pd.DataFrame({'mu': mu_grid, 'f1': f1_scores})
f1_scores.to_csv('synthetic_f1_scores.csv', index=False)

# train best adapted model with source as initial weights
adapted_model = deepcopy(tl_model)
adapted_model.set_params(mu=mu_best, random_init=False)
adapted_model.fit(X_t_selected, y_t_selected)

# compare models
y_pred_source = source_model.predict(X_t)
y_pred_target = target_model.predict(X_t)
y_pred_adapted = adapted_model.predict(X_t)
y_pred_best_adapted = best_adapted_model.predict(X_t)

# print results, f1 score
f1_source = f1_score(y_t, y_pred_source)
f1_target = f1_score(y_t, y_pred_target)
f1_adapted = f1_score(y_t, y_pred_adapted)
f1_best_adapted = f1_score(y_t, y_pred_best_adapted)

# save to file
results = pd.DataFrame({'source': f1_source, 'target': f1_target, 'adapted': f1_adapted, 'best_adapted': f1_best_adapted}, index=[0])
results.to_csv('synthetic_results_f1.csv', index=False)

ALPHA = 0.4
# plot decision boundaries (in separate plots and export to .pgf file)
plot_decision_boundary(source_model, X_s, y_s, save_path=f'source_decision_boundary_on_source.{FILE_TYPE}', x_hilight=X_t_selected)
disp = plot_decision_boundary(source_model, X_t, y_t, save_path=f'source_decision_boundary_on_target.{FILE_TYPE}', x_hilight=X_t_selected)
disp = plot_decision_boundary(target_model, X_t, y_t, save_path=f'target_decision_boundary.{FILE_TYPE}', x_hilight=X_t_selected)
# disp.ax_.savefig('target_decision_boundary.png')
disp = plot_decision_boundary(adapted_model, X_t, y_t, save_path=f'adapted_decision_boundary.{FILE_TYPE}', x_hilight=X_t_selected)
# disp.ax_.savefig('adapted_decision_boundary.png')
plot_decision_boundary(adapted_models[0], X_t, y_t, save_path=f'adapted_decision_boundary_mu_{mu_grid[0]}.{FILE_TYPE}', x_hilight=X_t_selected)
plot_decision_boundary(adapted_models[1], X_t, y_t, save_path=f'adapted_decision_boundary_mu_{mu_grid[1]}'
                                                              f'.{FILE_TYPE}', x_hilight=X_t_selected)
plot_decision_boundary(adapted_models[2], X_t, y_t, save_path=f'adapted_decision_boundary_mu_{mu_grid[2]}'
                                                              f'.{FILE_TYPE}', x_hilight=X_t_selected)
plot_decision_boundary(adapted_models[3], X_t, y_t, save_path=f'adapted_decision_boundary_mu_{mu_grid[3]}'
                                                              f'.{FILE_TYPE}', x_hilight=X_t_selected)








