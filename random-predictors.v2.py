#!/usr/bin/env ipython
# coding=utf-8

import numpy as np
import pandas as pd
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt

import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
rng = np.random.default_rng()

actual_aps = [
    0.0272,
    0.597,
    0.1848,
    0.1093,
    0.0038,
    0.0983,
    0.2344,
    0.5421,
    0.0015,
    0.1411,
    0.3028,
    0.7615,
    0.7836,
    0.0035,
    0.0093,
    0.61,
    0.5149,
    0.018,
    0.0283,
    0.034,
    0.2526,
    0.001,
    0.2796,
    0.3987,
    0.066,
    0.6082,
    0.0921,
    0.6635,
    0.1125,
    0.1163,
    0.1983,
    0.108,
    0.3407,
    0.339,
    0.3192,
    0.0994,
    0.254,
    0.4146,
    0.4115,
    0.053,
    0.2219,
    0.1737,
    0.014,
    0.0099,
    0.0899,
    0.0294,
    0.0436,
    0.81,
    0.2296,
    0.0523,
    0.4982,
    0.0435,
    0.2402,
    0.0581,
    0.1059,
    0.0108,
    0.1426,
    0.1411,
    0.0131,
    0.2383,
    0.3784,
    0.0887,
    0.0881,
    0.5169,
    0.7326,
    0.484,
    0.0725,
    0.5272,
    0.3127,
    0.1343,
    0.0023,
    0.0772,
    0.3409,
    0.3359,
    0.2394,
    0.004,
    0.3276,
    0.0078,
    0.0189,
    0.165,
    0.0513,
    0.4065,
    0.0224,
    0.2168,
    0.2394,
    0.0291,
    0.1444,
    0.026,
    0.0108,
    0.0897,
    0.0767,
    0.4824,
    0.0543,
    0.005,
    0.1353,
    0.3108,
    0.2744,
    0.0153,
    0.1457,
    0.3903,
    0.0264,
    0.1614,
    0.7454,
    0.1603,
    0.0612,
    0.407,
    0.3353,
    0.1365,
    0.1014,
    0.8371,
    0.2541,
    0.1384,
    0.1007,
    0.227,
    0.2143,
    0.3314,
    0.3108,
    0.4367,
    0.0991,
    0.4036,
    0.0335,
    0.1882,
    0.6764,
    0.1298,
    0.4536,
    0.0363,
    0.1396,
    0.2122,
    0.3696,
    0.5951,
    0.4869,
    0.0018,
    0.1377,
    0.3844,
    0.0678,
    0.057,
    0.0289,
    0.2177,
    0.0266,
    0.0681,
    0.6839,
    0.0166,
    0.1356,
    0.8341,
    0.1573,
    0.1667,
    0.2562,
    0.0119,
    0.0771,
    0.2743,
]
num_queries = len(actual_aps)

max_predictors = 60 # 100
num_trials = 50


def compute_rhos_rmses_via_ols(predictions): # Retrospective
    print(f'{predictions.shape} {len(actual_aps)}')
    model = LinearRegression().fit(predictions, actual_aps)
    ap_predicted = model.predict(predictions)
    print(f'{predictions.shape} {len(actual_aps)} {ap_predicted.shape}')
    rho, _ = stats.pearsonr(actual_aps, ap_predicted)
    rmse = root_mean_squared_error(actual_aps, ap_predicted)
    print(f'{predictions.shape[1]} {rho:10.6f} {rmse:>10.6f}')
    return (rho, rmse)

def compute_rhos_rmses_via_ols_loo(predictions):
    # Leave-one-out mode (should use sklearn function for this)
    ap_predicted = np.empty((num_queries,))
    for i in range(num_queries) :
        train_x = np.concatenate([predictions[0:i,:], predictions[i+1:,:]])
        train_ap = actual_aps[0:i] + actual_aps[i+1:]
        model = LinearRegression().fit(train_x, train_ap)
        test_x = predictions[i,:].reshape(1,-1)
        ap_predicted[i] = model.predict(test_x)[0]

    rho, _ = stats.pearsonr(actual_aps, ap_predicted)
    rmse = root_mean_squared_error(actual_aps, ap_predicted)
    print(f'{predictions.shape[1]} {rho:10.6f} {rmse:>10.6f}')
    return (rho, rmse)

if __name__ == '__main__':
    mean_rhos = np.empty((max_predictors,))
    std_rhos = np.empty((max_predictors,))
    mean_rmses = np.empty((max_predictors,))
    std_rmses = np.empty((max_predictors,))

    rhos = np.empty((num_trials,))
    rmses = np.empty((num_trials,))

    # For max_predictors = 100, LOO fitting:
    # rng = np.random.default_rng(seed=445136130656)

    # For max_predictors = 60, retrospective (train == test) fitting:
    # rng = np.random.default_rng(seed=34658102814551)
    

    for num_predictors in range(1,max_predictors+1) :
        for j in range(1,num_trials+1) :
            # Generate random predictions
            predictions = rng.random((num_queries, num_predictors))
            #rho, rmse = compute_rhos_rmses_via_ols(predictions)
            rho, rmse = compute_rhos_rmses_via_ols_loo(predictions)
            rhos[j-1] = rho
            rmses[j-1] = rmse
        mean_rhos[num_predictors-1] = np.mean(rhos)
        std_rhos[num_predictors-1] = np.std(rhos)
        mean_rmses[num_predictors-1] = np.mean(rmses)
        std_rmses[num_predictors-1] = np.std(rmses)

    fig, axs = plt.subplots(1, 2)
    axs[0].plot(mean_rhos)
    axs[1].plot(mean_rmses)
