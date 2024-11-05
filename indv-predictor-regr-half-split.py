"""
Compute multiple regression 
"""
import pandas as pd
import numpy as np
import sys
#from ranx import Run, fuse

import argparse, itertools
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error
from pandas import ExcelWriter
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lars, LarsCV, Lasso, LassoCV
from sklearn.linear_model import Ridge, RidgeCV, LassoLars, LassoLarsCV, GammaRegressor, PoissonRegressor
from sklearn.model_selection import KFold, RepeatedKFold
from scipy.stats import ttest_ind
from sklearn.svm import SVR, NuSVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge

def min_max_normalize(lst):
    """Min-max normalization of a list of numbers."""

    min_val = min(lst)
    max_val = max(lst)

    if min_val == max_val:
        return [0] * len(lst) 

    normalized_lst = [(x - min_val) / (max_val - min_val) for x in lst]
    return normalized_lst

def squared_error(predictions, targets):
    predictions = np.array(predictions)
    targets = np.array(targets)
    #return np.sqrt(((predictions - targets) ** 2))
    return ((predictions - targets) ** 2)


def plot_scatter_and_line(qpp_scores, y_train):
    """
    Input (qpp_scores) : a column (Series) containing QPP scores
    """
    # QPP scores = inputs/independent variables
    # AP scores = outputs/responses/dependent variables
    x = qpp_scores.to_numpy().reshape(-1,1) # any no. of rows x 1 column
    model = LinearRegression().fit(x, y_train)
    return model

def compute_regression(best_qpp_approach):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, help = "./PATH/TO/CSV/INPUT")
    parser.add_argument("--k", type = str, choices= ["100", "1000"], help = "retrieval depth")
    parser.add_argument("--qpp_type", type = str, choices= ["pre", "post"], help= "pre / post")
    parser.add_argument("--dataset", type = str, choices = ["trec678rb", "trecdl", "trecdl19", 
                        "trecdl20", "clueweb09b"], help = "trec678rb / trecdl19 / trecdl20 / clueweb09b", required=True)
    parser.add_argument("--ols_type", type = str, choices= ["ols", "lars-cv", "enet", "ridge", "lasso-cv"], help= "ols types, ols / lars-cv / enet / ridge / lasso-cv")


    args = parser.parse_args()
    input_dir = str(args.input)
    dataset_type = str(args.dataset)
    qpp_type = str(args.qpp_type)
    ols_type = str(args.ols_type)

    k = str(args.k)

    input_path = input_dir + "/" + dataset_type + "-" + qpp_type + "-ret.csv"

    df = pd.read_csv(input_path, skiprows = 3)

    if qpp_type == "post":
        qpp_approaches = ["nqc", "wig", "clarity", "uef_nqc", "uef_wig", 
                        "uef_clarity", "neuralqpp", "qppbertpl", "deepqpp", "bertqpp"]
        best_qpp = ["qppbertpl"]
    
    else:
        #qpp_approaches = ["MaxIDF", "AvgIDF", "AvQC", "AVQCG", "SumSCQ", "MaxSCQ",
        #                  "AvgSCQ", "SumVAR", "AvgVAR", "MaxVAR", "AvP", "AvNP"]
        qpp_approaches = ["MaxIDF", "AvgIDF", "SumSCQ", "MaxSCQ",
                        "AvgSCQ", "SumVAR", "AvgVAR", "MaxVAR", "AvP", "AvNP"]
        best_qpp = ["MaxIDF"]
    
    # unsupervised only
    #qpp_approaches = ["nqc", "wig", "clarity", "uef_nqc", "uef_wig", 
    #                  "uef_clarity", "neuralqpp"]
    #best_qpp = ["deepqpp"]
    #best_qpp = ["qppbertpl"]
    best_qpp = [best_qpp_approach]
    #best_qpp.append(best_qpp_approach)
    #best_qpp = ["MaxIDF"] 
    #best_qpp = ["bertqpp"]    
    #qpp_approaches = ["deepqpp"]
    
    # Discarding the row containing the avg. map
    df = df[df["QID"] != "MAP"]

    #ap_real_pred = pd.read_csv("querywise-ap-qppscores.csv", header=3, index_col=0)
    #ap_real_pred = pd.read_csv("data/qpp-fusion-trecdl-dl19.csv", header=3, index_col=0)
    ap_real_pred = pd.read_csv(input_path, header=3, index_col=0)

    scaler = MinMaxScaler()
    #for qpp_approach in qpp_approaches:
    ap_qpp_pred = pd.DataFrame()
    # min-max normalization of all qpp predictors    
    ap_real_pred[qpp_approaches] = scaler.fit_transform(ap_real_pred[qpp_approaches])
    ap_qpp_pred[best_qpp] = scaler.fit_transform(ap_real_pred[best_qpp])

    # In case you don't want to normalize 
    #ap_real_pred[qpp_approaches] = ap_real_pred[qpp_approaches]
    #ap_qpp_pred[best_qpp] = ap_real_pred[best_qpp]
    #print(f"k : {k}")
    if k == "100":
        y = ap_real_pred['ap@100']
    else:
        y = ap_real_pred['ap@1000']

    # Split <tredl (train), trecdl (test)>
    ap_real_pred_train = ap_real_pred.iloc[:len(ap_qpp_pred)]
    ap_real_pred_test = ap_real_pred.iloc[:len(ap_real_pred)]

    if k == "100":
        y_train = ap_real_pred_train['ap@100']
        y_test = ap_real_pred_test['ap@100']
    else:
        y_train = ap_real_pred_train['ap@1000']
        y_test = ap_real_pred_test['ap@1000']

    x_train = ap_real_pred_train[qpp_approaches]
    #print(x_train.shape)

    fold = 0 
    avg_rmse_best_qpp = 0
    avg_rmse_combined_qpp = 0
    avg_p_value = 0 

    #print("Splitting into two halves")
    for seed in range(15): 
        kf = KFold(n_splits = 2, shuffle = True, random_state = seed + 2652124)
        #kf = RepeatedKFold(n_repeats = 15, n_splits = 2, random_state = 2652124)
        
        combined_qpp_sq_error_list = []
        indv_qpp_sq_error_list = []
        y_test_list = []

        combined_qpp_predicted_y_list = []
        indv_qpp_predicted_y_list = []

        for train_index, test_index in kf.split(ap_real_pred):
            fold += 1
            ap_real_pred_train, ap_real_pred_test = ap_real_pred.iloc[train_index], \
                ap_real_pred.iloc[test_index]

            #print(ap_real_pred_train.shape)
            #print(ap_real_pred_test.shape)
            #print(f"Fold : {fold}")
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            x_train = ap_real_pred_train[qpp_approaches]   

            best_qpp_predictor_train = ap_real_pred_train[best_qpp]
            best_qpp_predictor_test = ap_real_pred_test[best_qpp]

            #print(f"shape of x_train {x_train.shape}")

            #regr_model = ElasticNetCV(cv = 5, random_state = 0)
            #regr_model = ElasticNet(alpha=0.001, l1_ratio=0.6, random_state = 0)
            #regr_model = LassoCV(cv = 5, random_state = 0, max_iter = 50000,)
            #regr_model = Lasso(alpha= 0.001, random_state = 0)
            #regr_model = Lasso(alpha= 0.0001)
            #regr_model = Lars(n_nonzero_coefs = 7)
            #regr_model =  LarsCV(cv=5)
            #regr_model = LassoCV(cv = 5, random_state = 0)
            # Trying Support Vector Regression
            #regr_model = make_pipeline(StandardScaler(), SVR(epsilon=0.1))
            #regr_model = make_pipeline(StandardScaler(), NuSVR(C=1, nu=0.1))
            #regr_model = KernelRidge(alpha=.2)
            #regr_model = Ridge(alpha=0.8)
            #regr_model = RidgeCV(cv = 5)
            #regr_model = LassoLars(alpha=1.0)
            #regr_model = LassoLarsCV(cv=5)
            #regr_model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
            #regr_model = GammaRegressor()
            #regr_model = PoissonRegressor()
            regr_model = LinearRegression()

            #print(f"Shape of x_train : {x_train.shape}")
            #print(f"Shape of y_train : {y_train.shape}")

            regr_model = regr_model.fit(x_train, y_train)

            #best_qpp_regr_model = ElasticNetCV(cv = 5, random_state = 0)
            #best_qpp_regr_model = ElasticNet(alpha=0.001, l1_ratio=0.6, random_state = 0)
            #best_qpp_regr_model = LassoCV(cv = 5, random_state = 0)
            #best_qpp_regr_model = Lasso(alpha= 0.001, random_state = 0)
            #best_qpp_regr_model = Lars(n_nonzero_coefs=1)
            # Trying Support Vector Regression
            #best_qpp_regr_model = make_pipeline(StandardScaler(), SVR(epsilon=0.2)) 
            #best_qpp_regr_model  = make_pipeline(StandardScaler(), NuSVR(C=1, nu=0.1))
            #best_qpp_regr_model = KernelRidge(alpha=0.2)
            #best_qpp_regr_model  = LarsCV(cv=5)
            #best_qpp_regr_model = Ridge(alpha=0.8)
            #best_qpp_regr_model = RidgeCV(cv = 5)
            #best_qpp_regr_model = LassoLars(alpha=1.0)
            #best_qpp_regr_model = LassoLarsCV(cv=5)
            #best_qpp_regr_model = RidgeCV(alphas=[1e-3, 1e-2, 1e-1, 1])
            #best_qpp_regr_model = GammaRegressor()
            #best_qpp_regr_model = PoissonRegressor()
            best_qpp_regr_model = LinearRegression()

            best_qpp_regr_model = best_qpp_regr_model.fit(best_qpp_predictor_train, y_train)

            x_test = ap_real_pred_test[qpp_approaches]
            #print(f"Shape of best qpp predictor: {best_qpp_predictor_train.shape}")
            #exit(1)
            #print(x_test.shape)

            y_predicted = regr_model.predict(x_test)
            y_best_qpp_predicted = best_qpp_regr_model.predict(best_qpp_predictor_test)

            y_predicted = np.clip(y_predicted, 0, 1)
            y_best_qpp_predicted = np.clip(y_best_qpp_predicted, 0, 1)

            rmse = root_mean_squared_error(y_test, y_predicted)
            #print(rmse, regr_model.score(x_test, y_test))
            combined_qpp_predicted_y_list.extend(y_predicted)   

            combined_qpp_sq_error = squared_error(y_predicted, y_test)

            #print(f"combined qpp sq error : {combined_qpp_sq_error}")
            #print(f"y_predicted {y_predicted}")
            #print(f"y_test values: {y_test}")

            # Commenting below as we don't want to print all these for all different folds
            #print(f"Predicted y vector: {y_predicted}")
            #print(f"Input x_test: {x_test}")
            #print(f"Actual y_test value: {y_test}")

            #print(f"No. features seen during training : {regr_model.n_features_in_}")
            #print(f"Features seen during training : {regr_model.feature_names_in_}")
            #print(f"Parameter vector : {regr_model.coef_}")
            #print(f"Intercept : {regr_model.intercept_}")

            #print(f"Alphas:  {regr_model.alphas_}")
            #print(f"active : {regr_model.active_}")
            #print(f"Alpha:  {regr_model.alpha_}")
            #print(f"Coeff path: {regr_model.coef_path_}")
            
            combined_qpp_sq_error_list.extend(combined_qpp_sq_error)

            #print(f"best qpp approach: {str(best_qpp)}")
            #print(root_mean_squared_error(y_test, y_best_qpp_predicted), best_qpp_regr_model.score(best_qpp_predictor_test, y_test))
        
            indv_qpp_sq_error = squared_error(y_test, y_best_qpp_predicted)
            #print(f"Shape of indv error : {indv_qpp_sq_error.shape}")
            indv_qpp_sq_error_list.extend(indv_qpp_sq_error)

            indv_qpp_predicted_y_list.extend(y_best_qpp_predicted)
            y_test_list.extend(y_test)

            #print("indv QPP squared error: ")
            # Perform the t-test
            #t_statistic, p_value = ttest_ind(indv_qpp_sq_error, combined_qpp_sq_error)
            t_statistic, p_value = ttest_ind(indv_qpp_sq_error, combined_qpp_sq_error, alternative='greater')
            #print("t-statistic:", t_statistic)
            #print("p-value:", p_value)

        #print("t-test and pvalue of the entire list")
        #print("Shape of the two lists")
        #print(f"Shape of entire list : {len(combined_qpp_sq_error_list)}")
        #print(f"Shape of {best_qpp} : {len(indv_qpp_sq_error_list)}")
        #t_statistic, p_value = ttest_ind(combined_qpp_sq_error_list, indv_qpp_sq_error_list)
        t_statistic, p_value = ttest_ind(indv_qpp_sq_error_list, combined_qpp_sq_error_list, alternative='greater')
        
        #print("In how many cases combind QPP predictor winning : ")
        predictor_status = np.array(np.array(indv_qpp_sq_error_list) > np.array(combined_qpp_sq_error_list)).sum()
        #print(predictor_status)

        #print("t-statistic:", t_statistic)
        #print("p-value:", p_value)
    
        avg_rmse_best_qpp += root_mean_squared_error(indv_qpp_predicted_y_list, y_test_list)
        avg_rmse_combined_qpp += root_mean_squared_error(combined_qpp_predicted_y_list, y_test_list)
        avg_p_value += p_value

        #print(f"RMSE of best QPP {best_qpp} : {root_mean_squared_error(indv_qpp_predicted_y_list, y_test_list)}")
        #print(f"RMSE of the combined predictor : {root_mean_squared_error(combined_qpp_predicted_y_list, y_test_list)}")

    entire_set = fold /2 
    avg_rmse_best_qpp = avg_rmse_best_qpp / entire_set
    avg_rmse_combined_qpp = avg_rmse_combined_qpp / entire_set 
    avg_p_value = avg_p_value / entire_set

    print(f"avg. rmse of {best_qpp} qpp : {avg_rmse_best_qpp:>.4f}")
    print(f"avg. rmse of combined qpp : {avg_rmse_combined_qpp:>.4f}")
    print(f"avg. p-value : {avg_p_value:>.4f}")
    print(f"Total number of folds generated {fold}")


if __name__ == '__main__':
    
    #qpp_approaches = ["MaxIDF", "AvgIDF", "AvQC", "AVQCG", "SumSCQ", "MaxSCQ",
    #                  "AvgSCQ", "SumVAR", "AvgVAR", "MaxVAR", "AvP", "AvNP"]
    
    qpp_approaches = ["MaxIDF", "AvgIDF", "SumSCQ", "MaxSCQ",
                      "AvgSCQ", "SumVAR", "AvgVAR", "MaxVAR", "AvP", "AvNP"]
    
    # TODO : when you want to run for post please comment above line and uncomment below
    #qpp_approaches = ["nqc", "wig", "clarity", "uef_nqc", "uef_wig", 
    #                  "uef_clarity", "neuralqpp", "qppbertpl", "deepqpp", "bertqpp"]
    
    for qpp_approach in qpp_approaches:
        compute_regression(qpp_approach)
