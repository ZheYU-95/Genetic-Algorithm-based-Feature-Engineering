from math import inf
from dask.base import compute
from numpy.core.numeric import NaN
from pandas.core.frame import DataFrame
import numpy as np
import pandas as pd
import copy
import random 
import sys
import logging, coloredlogs
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import tribes_competition as tc
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
from dask import delayed
from dask.distributed import LocalCluster, Client
import tc_distributed_pro as tcdp
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from boruta import BorutaPy
import math
import pathlib


logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_new = logging.getLogger(__name__)
output_file_handler = logging.FileHandler("log/baseline.log")
# output_file_handler = logging.FileHandler("log/R/output_" + dataset[7:-3] + 'log')
output_file_handler.setFormatter(formatter)
logger_new.addHandler(output_file_handler)
coloredlogs.install(level = 'DEBUG')
# pathlib.Path(datapath).mkdir(parents=True, exist_ok=True)


# dataset = 'data/R/Housing_Boston.csv'
# dat = tcdp.load_data(dataset, logger=logger_new, art='R')
# datapath = 'result/' + dataset[5:-4]
# prev_dat = pd.read_csv(datapath + '/gen1.csv')
# prev_dat.drop(prev_dat.columns[0], axis=1, inplace=True)
# prev_dat.drop(prev_dat.columns[-1], axis=1, inplace=True)
# cur_dat = pd.read_csv(datapath + '/gen2.csv')
# cur_dat.drop(cur_dat.columns[0], axis=1, inplace=True)
# cur_dat.drop(cur_dat.columns[-1], axis=1, inplace=True)
# # cur_dat[dat.columns[-1]] = dat.iloc[:, -1]
# total_dat = pd.concat([prev_dat, cur_dat], axis=1)
# cur_dat2 = pd.read_csv(datapath + '/gen3.csv')
# cur_dat2.drop(cur_dat2.columns[0], axis=1, inplace=True)
# total_dat = pd.concat([total_dat, cur_dat2], axis=1)
# total_dat.to_csv(datapath + '/total' + '.csv')

# total_dat = pd.read_csv('result/Housing_Boston/total.csv')
# total_dat.drop(total_dat.columns[0], axis=1, inplace=True)

# """ find the best number of the selected features """
# max_inc, coef1, coef2 = 0, 0, 0
# for i in range(1, 10):
#     i /= 10
#     num_best_features = round(i * total_dat.shape[1])
#     best_features_cands = tcdp.bestFeatures(total_dat, num_best_features, art='R')
#     # for j in range(1, 10):
#         # j /= 10
#     j = 3
#     size_limit = max(2, round(j * dat.shape[1]))
#     best_features = pd.DataFrame(best_features_cands)
#     best_features[total_dat.columns[-1]] = total_dat.iloc[:, -1]
#     while best_features.shape[1] > size_limit:
#         best_features = tcdp.featureSelection(best_features, art='R')
#     best_features.drop(best_features.columns[-1], axis=1, inplace=True)
#     init_fitness, cur_fitness = tcdp.scoreCompare(dat, best_features, art='R')
#     increase = (cur_fitness - init_fitness) / init_fitness
#     if increase > max_inc:
#         max_inc, coef1, coef2 = increase, i, j
# logger_new.info("After grid search of best coeffs, with coef1: %f coef2: %f, the increase maximized by %f" %(coef1, coef2, max_inc))


# dat = tcdp.load_data('data/R/Housing_Boston.csv', art='R')
# total_dat = pd.read_csv('result/Housing_Boston/total.csv')
# print(total_dat.shape[1])
# total_dat.drop(total_dat.columns[0], axis=1, inplace=True)

# Select best K features from the total candidate features
# num_best_features = round(3 * dat.shape[1])
# num_best_features = round(0.5 * total_dat.shape[1])
# best_features_cands = tcdp.bestFeatures(total_dat, num_best_features, art='R', logger=logger_new)
# logger_new.info("Finish selecting best %d features according to their importance in the first round coarsely" %(num_best_features))
# init_fitness, cur_fitness = tcdp.scoreCompare(dat, best_features_cands, art='R', logger=logger_new)
# logger_new.debug("Compared with the initial one, the fitness increased by %s" %(str((cur_fitness-init_fitness)/init_fitness)))

# # reduce the features number on the basis of a higher score compared to the initial one
# print(dat.shape[1])
# size_limit = max(2, round(3 * dat.shape[1]))
# print("The limit of the final gen's size is %d" %(size_limit))
# best_features = pd.DataFrame(best_features_cands)
# best_features[total_dat.columns[-1]] = total_dat.iloc[:, -1]
# while best_features.shape[1] > size_limit:
#     best_features = tcdp.featureSelection(best_features, art='R', logger=logger_new)
# best_features.drop(best_features.columns[-1], axis=1, inplace=True)
# logger_new.info("Reduce the size of the final selected features to %d by featureSelection while keeping score not dropping" %(best_features.shape[1]))
# best_features.to_csv('result/Housing_Boston/final' + '.csv')

# init_fitness, cur_fitness = tcdp.scoreCompare(dat, best_features, art='R', logger=logger_new)
# increase = (cur_fitness - init_fitness) / init_fitness
# logger_new.debug("Compared with the initial one, the fitness increased by %s" %(str(increase)))





""" calculate initial baseline scores """
# datasets = ['data/winequality_white.csv']
datasets = ['data/R/Openml_616.csv']
# datasets = ['data/Higgs_Boson.csv', 'data/Amazon_employee_access.csv', 'data/Amazon_employee_access.csv', 
          #   'data/SpectF.csv', 'data/German_Credit.csv', 'data/AP_Omentum_Ovary.csv', 
          #   'data/Lymphography.csv', 'data/Ionosphere_cleaned.csv', 'data/CreditCard_default.csv', 'data/messidor_features.csv', 
          #   'data/winequality_red.csv', 'data/winequality_white.csv', 'data/SpamBase.csv']
# datasets = ['data/R/Housing_Boston.csv','data/R/Airfoil.csv', 'data/R/Openml_618.csv', 'data/R/Openml_589.csv', 'data/R/Openml_616.csv', 
#            'data/R/Openml_607.csv', 'data/R/Openml_620.csv', 'data/R/Openml_637.csv', 'data/R/Openml_586.csv']


for dataset in datasets:
     dat = tcdp.load_data(dataset, art='R')
    #  dat = tcdp.pd.read_csv(dataset)
     print(dat.shape[1])
     # num = 50
     # group = tcdp.bestFeatures(dat, num, art='C')
     # res = tcdp.calculateFitness(dat, group, art='C')
     # res, v = tcdp.compute(res)[0]
     # print("For dataset %s which selects %d features from original, its mean score is %f, standard variance is %f" %(str(dataset[5:-4]), num, res, v))
     
     group = dat.drop(dat.columns[-1], axis=1, inplace=False)
     datapath = 'result/' + dataset[5:-4]
    #  datapath = 'result/R' + dataset[7:-4]
     res = tcdp.eval(dat.iloc[:, :-1],dat.iloc[:, -1], art='R')
     res, v = compute(res)[0]
     logger_new.info("For dataset %s, mean score is %f, standard variance is %f" %(str(dataset[5:-4]), res, v))






def relative_absolute_error(y_true: pd.Series, y_pred: pd.Series):
    y_true_mean = y_true.mean()
    n = len(y_true)
    # Relative Absolute Error
    # err = math.sqrt(sum(np.square(y_true - y_pred)) / math.sqrt(sum(np.square(y_true-y_true_mean))))
    err = sum(abs(y_true - y_pred)) / sum(abs(y_true - y_true_mean))
    return err
score = make_scorer(relative_absolute_error, greater_is_better=True)

""" tune parameters of RF Classifier """
# dataset = 'data/winequality_white.csv'
dataset = 'data/R/Openml_616.csv'
if dataset[5] == 'R':
    scoring_func = score
else:
    scoring_func = 'f1_weighted'


# dat = tcdp.load_data(dataset)
# X = dat.iloc[:, :-1]
# y = dat.iloc[:, -1]
# param_test1 = {'n_estimators':range(10,201,10)}
# gsearch1 = GridSearchCV(estimator=RandomForestRegressor(random_state=10, n_jobs=-1),
#                         param_grid=param_test1, scoring=scoring_func,cv=5)
# gsearch1.fit(X,y)
# print(1-gsearch1.cv_results_['mean_test_score'], gsearch1.cv_results_['params'])

# param_test2 = {'max_depth':range(10,201,10), 'min_samples_split':range(10,201,10)}
# gsearch2 = RandomizedSearchCV(estimator=RandomForestRegressor(n_estimators= 20, random_state=10, n_jobs=-1),
#                         param_distributions=param_test2, scoring=scoring_func, cv=5)
# gsearch2.fit(X,y)
# print(1-gsearch2.cv_results_['mean_test_score'], gsearch2.cv_results_['params'])

# param_test3 = {'min_samples_split':range(20,120,10), 'min_samples_leaf':range(10,60,10)}
# gsearch3 = RandomizedSearchCV(estimator=RandomForestClassifier(n_estimators= 30, max_depth=50, random_state=10, n_jobs=-1),
#                         param_distributions=param_test3, scoring=scoring_func, cv=5)
# gsearch3.fit(X,y)
# print(gsearch3.cv_results_['mean_test_score'], gsearch3.cv_results_['params'])
