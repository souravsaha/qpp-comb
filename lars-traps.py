"""
Compute multiple regression with lars-traps
Split strategy: leave one out
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
from sklearn.model_selection import LeaveOneOut
from smare import smare

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

def weight_vector_with_traps(coeff_path, intercept, no_of_predictors):
    
    #print(f"length of the coeff path: {len(coeff_path[0])-1}")
    #print(f"coeff path: {coeff_path}")
    column_num = 0
    for col in range(coeff_path.shape[1]):
        #print(f"vector {coeff_path[no_of_predictors:, col]}")
        if np.any(coeff_path[no_of_predictors:, col] != 0) :
            column_num = col
            break
    if column_num > 0:
        column_num -= 1 
    else:
        column_num = 0
    #print(f"column no. {column_num}")
    weight_vector = coeff_path[:no_of_predictors, column_num]
    #print(f"weight vector {weight_vector}")
    return weight_vector, intercept

def predict_y_values(weight_vector, x_test, intercept):
    # print(f"x_test {x_test.to_string()}")
    # print(f"new shape: {np.array(x_test).size}")
    # print(f"intercept: {intercept}")
    # print(f"weight_vector: {weight_vector}")
    
    # print(f"shape of weight vector {np.array(weight_vector).reshape(1, -1).transpose().size}")

    y_predict = np.array(x_test) @ np.array(weight_vector) + intercept     

    # print(f"y_predict {y_predict}")

    return y_predict

def compute_regression():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type = str, help = "./PATH/TO/CSV/INPUT")
    parser.add_argument("--k", type = str, choices= ["100", "1000"], help = "retrieval depth")
    parser.add_argument("--qpp_type", type = str, choices= ["pre", "post"], help= "pre / post")
    parser.add_argument("--dataset", type = str, choices = ["trec678rb", "trec678", "trecdl", "trecdl19", 
                        "trecdl20", "clueweb09b"], help = "trec678rb / trec678 / trecdl19 / trecdl20 / clueweb09b", required=True)

    args = parser.parse_args()    
    input_dir = str(args.input)
    dataset_type = str(args.dataset)
    qpp_type = str(args.qpp_type)

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
    #best_qpp = [best_qpp_approach]
    #best_qpp.append(best_qpp_approach)
    best_qpp = ["MaxIDF"] 
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
    avg_kendall = 0
    avg_pearson = 0
    avg_smare = 0


    loo = LeaveOneOut()

    combined_qpp_sq_error_list = []
    indv_qpp_sq_error_list = []
    y_test_list = []

    combined_qpp_predicted_y_list = []
    indv_qpp_predicted_y_list = []

    for train_index, test_index in loo.split(ap_real_pred):

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

        #regr_model = LinearRegression()
        #regr_model = LassoCV(cv = 5, random_state = 0, max_iter = 50000)
        #regr_model = ElasticNetCV(cv = 5, random_state = 0, max_iter = 50000)
        #regr_model =  LarsCV(cv =5)
        regr_model =  Lars(fit_intercept=False)

        #print(f"Shape of x_train : {x_train.shape}")
        # generate random predictors
        row, col = x_train.shape
        random_predictors = np.random.rand(row, 6)      # as mentioned in Hauff et al.
        bias_vector = np.ones((row, 1))
        #print(f"Shape of random predictors: {random_predictors.shape}")      
        #print(f"Shape of y_train : {y_train.shape}")
        x_train = np.hstack((x_train, bias_vector))
        x_train = np.hstack((x_train, random_predictors))
        #print(f"Shape of new x_train : {x_train.shape}")
        regr_model = regr_model.fit(x_train, y_train)

        best_qpp_regr_model = LinearRegression()

        best_qpp_regr_model = best_qpp_regr_model.fit(best_qpp_predictor_train, y_train)

        x_test = ap_real_pred_test[qpp_approaches]
        #print(f"Shape of best qpp predictor: {best_qpp_predictor_train.shape}")
        #exit(1)
        #print(x_test.shape)
        weight_vector, intercept = weight_vector_with_traps(regr_model.coef_path_, regr_model.intercept_, len(qpp_approaches) + 1)
        row, _ = x_test.shape
        bias_vector = np.ones((row, 1))
        new_x_test = np.hstack((x_test, bias_vector))
        y_predicted = predict_y_values(weight_vector, new_x_test, intercept)
        #print(f"Input x_test: {x_test}")
        #print(f"Predicted y: {y_predicted}")
        #exit(1)

        #y_predicted = regr_model.predict(x_test)
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

        #print(f"Coef path: {regr_model.coef_path_}")
        #print(f"Alphas:  {regr_model.alphas_}")
        #print(f"Alpha:  {regr_model.alpha_}")
            
        combined_qpp_sq_error_list.extend(combined_qpp_sq_error)

        #print(f"best qpp approach: {str(best_qpp)}")
        #print(root_mean_squared_error(y_test, y_best_qpp_predicted), best_qpp_regr_model.score(best_qpp_predictor_test, y_test))
        
        indv_qpp_sq_error = squared_error(y_test, y_best_qpp_predicted)
        #print(f"Shape of indv error : {indv_qpp_sq_error.shape}")
        indv_qpp_sq_error_list.extend(indv_qpp_sq_error)

        indv_qpp_predicted_y_list.extend(y_best_qpp_predicted)
        y_test_list.extend(y_test)

        #print(f"indv QPP squared error: {indv_qpp_sq_error}")
        #print(f"comb. QPP squared error: {combined_qpp_sq_error}")
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

    corr,_ = stats.kendalltau(combined_qpp_predicted_y_list, y_test_list)
    avg_kendall += corr

    corr, _ = stats.pearsonr(combined_qpp_predicted_y_list, y_test_list)
    avg_pearson += corr
    # DONE: add for SMARE too
    temp = pd.DataFrame()
    temp['ap'] = y_test_list
    temp['predictor'] = combined_qpp_predicted_y_list
    avg_smare += smare(temp['ap'], temp['predictor'])
    
    print(f"RMSE of best QPP {best_qpp} : {root_mean_squared_error(indv_qpp_predicted_y_list, y_test_list)}")
    print(f"RMSE of the combined predictor : {root_mean_squared_error(combined_qpp_predicted_y_list, y_test_list)}")


    print(f"avg. rmse of {best_qpp} qpp : {avg_rmse_best_qpp:>.4f}")
    print(f"avg. rmse of combined qpp : {avg_rmse_combined_qpp:>.4f}")
    print(f"avg. p-value : {avg_p_value:>.4f}")
    print(f"Total number of folds generated {fold}")

    print(f"avg. kendall : {avg_kendall:>.4f}")
    print(f"avg. pearson : {avg_pearson:>.4f}")
    print(f"avg. smare : {avg_smare:>.4f}")



if __name__ == '__main__':
    
    #qpp_approaches = ["MaxIDF", "AvgIDF", "SumSCQ", "MaxSCQ",
    #                  "AvgSCQ", "SumVAR", "AvgVAR", "MaxVAR", "AvP", "AvNP"]
    
    #qpp_approaches = ["nqc", "wig", "clarity", "uef_nqc", "uef_wig", 
    #                    "uef_clarity", "neuralqpp", "qppbertpl", "deepqpp", "bertqpp"]
    
    #for qpp_approach in qpp_approaches:
    compute_regression()