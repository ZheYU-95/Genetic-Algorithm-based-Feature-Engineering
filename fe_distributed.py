from joblib import logger
import pandas as pd
import transforms_distributed as tfd
# import tc_distributed as tcd
import tc_distributed_pro as tcdp
from dask.distributed import LocalCluster, Client, progress
from dask_jobqueue import HTCondorCluster
import logging, coloredlogs
import sys, os
import copy
import pathlib
import warnings

dataset = 'data/Lymphography.csv'
# dataset = 'data/R/Openml_589.csv'
datapath = 'result/' + dataset[5:-4]
art_new, start_idx = 'C', 5
if dataset[5] == 'R':
    art_new, start_idx = 'R', 7

logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_new = logging.getLogger(__name__)
output_file_handler = logging.FileHandler("log/output_" + dataset[start_idx:-3] + 'log')
output_file_handler.setFormatter(formatter)
logger_new.addHandler(output_file_handler)
coloredlogs.install(level = 'DEBUG')
pathlib.Path(datapath).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")


repeat = 4
una_oprs = tfd.unary_operators
bina_oprs = tfd.binary_operators
weights = [repeat] * (len(una_oprs) + len(bina_oprs))
# import data
dat = tcdp.load_data(dataset, logger=logger_new, art=art_new)
# the number of next generation's features not over inflation*num_of_curr_gen to form an up-side-down pyramid structure
# magic_number: inflation, cur_limit, total_limit, num_best_features
inflation = 10
cur_size = dat.shape[1]-1
# total_limit can be much larger, which denpends on the device
cur_limit = 2000
total_limit = 1000
cur_dat = dat
prev_gen = None
# only for the dataset with too many original features
dat_sel = None

if dat.shape[1] > 10000:
    # for AP_Omentum_Ovary
    group = dat.drop(dat.columns[-1], axis=1, inplace=False)
    res = tcdp.eval(dat.iloc[:, :-1], dat.iloc[:, -1], art=art_new)
    res, v = tcdp.compute(res)[0]
    logger_new.info("For original dataset %s, mean score is %f, standard variance is %f" %(str(dataset[start_idx:-4]), res, v))
    num = 50
    dat_sel = tcdp.bestFeatures(dat, num, art=art_new)
    cur_dat = dat_sel
    cur_dat[dat.columns[-1]] = dat.iloc[:, -1]
    group = cur_dat.drop(cur_dat.columns[-1], axis=1, inplace=False)
    res = tcdp.calculateFitness(dat, group, art=art_new)
    res, v = tcdp.compute(res)[0]
    logger_new.info("For dataset %s which selects %d features from original, its mean score is %f, standard variance is %f" 
                    %(str(dataset[start_idx:-4]), num, res, v))

# generate and store candidate features
for i in range(repeat):
    cur_dat, gen = tcdp.updateDat(cur_dat, prev_gen=prev_gen, oprs_weights=weights, art=art_new, logger=logger_new)
    cur_gen, cur_size = tcdp.constrainFeaturesNum(cur_dat, min(inflation*cur_size, cur_limit), art=art_new, logger=logger_new)
    logger_new.debug('The number of features in cur-gen drops from %d to %d' %(cur_dat.shape[1]-1, cur_size))

    prev_gen = pd.concat([prev_gen, gen], axis=1)
    prev_gen[dat.columns[-1]] = dat.iloc[:, -1]
    prev_gen, prev_size = tcdp.constrainFeaturesNum(prev_gen, total_limit, art=art_new, logger=logger_new)
    logger_new.debug('The number of features in prev-gen drops from %d to %d' %(prev_gen.shape[1], prev_size))

    if dat_sel is not None:
        cur_dat = tcdp.addInitalFeatures(cur_gen, prev_gen, dat, logger=logger_new)
    else:
        cur_dat = tcdp.addInitalFeatures(cur_gen, prev_gen, dat, logger=logger_new)
    cur_dat.to_csv(datapath + '/gen' + str(i+1) + '.csv')

total_dat = pd.concat([prev_gen, cur_dat], axis=1)
total_dat.to_csv(datapath + '/total.csv')
# drop the columns with high correlations
# total_dat = tcdp.dropHighCorrelation(total_dat, logger=logger_new)
# total_dat[dat.columns[-1]] = dat.iloc[:, -1]


dat = tcdp.load_data(dataset, logger=logger_new, art=art_new)
total_dat = pd.read_csv(datapath + '/total.csv')
total_dat.drop(total_dat.columns[0], axis=1, inplace=True)
max_inc, coef1, coef2 = 0, 0, 0
for i in range(1, 10):
    i /= 10
    num_best_features = round(i * total_dat.shape[1])
    best_features_cands = tcdp.bestFeatures(total_dat, num_best_features, art=art_new)
    last = dat.shape[1]
    for j in range(9, 0, -1):
        j /= 10
        if dat_sel is not None:
            size_limit = round(j * dat_sel.shape[1])
        else:
            size_limit = round(j * dat.shape[1])
        if size_limit > last:
            continue
        best_features = pd.DataFrame(best_features_cands)
        best_features[total_dat.columns[-1]] = total_dat.iloc[:, -1]
        while best_features.shape[1] > size_limit:
            best_features = tcdp.featureSelection(best_features, art=art_new)
        last = best_features.shape[1]
        best_features.drop(best_features.columns[-1], axis=1, inplace=True)
        init_fitness, cur_fitness = tcdp.scoreCompare(dat, best_features, art=art_new)
        increase = (cur_fitness - init_fitness) / init_fitness
        logger_new.info("with coef1=%f, coef2=%f, the increasing is %f" %(i, j, increase))
        if increase > max_inc:
            max_inc, coef1, coef2 = increase, i, j
logger_new.info("After grid search of best coeffs, with coef1: %f coef2: %f, the increase maximized by %f" %(coef1, coef2, max_inc))


# Select best K features from the total candidate features
# coef1 = 0.9
num_best_features = round(coef1 * total_dat.shape[1])
best_features_cands = tcdp.bestFeatures(total_dat, num_best_features, art=art_new, logger=logger_new)
init_fitness, cur_fitness = tcdp.scoreCompare(dat, best_features_cands, art=art_new, logger=logger_new)
logger_new.info("Finish selecting best %d features (coefficient: %f) in the first round coarsely" %(num_best_features, coef1))
increase = (cur_fitness - init_fitness) / init_fitness
logger_new.debug("After the first round, compared with the initial one, the fitness increased by %s" %(str(increase)))

# reduce the features number on the basis of a higher score compared to the initial one
# coef2 = 0.9
if dat_sel is not None:
    size_limit = round(coef2 * dat_sel.shape[1])
else:
    size_limit = round(coef2 * dat.shape[1])
logger_new.info("The limit of the final gen's size is %d with coeffiecient %f" %(size_limit, coef2))
best_features = pd.DataFrame(best_features_cands)
best_features[total_dat.columns[-1]] = total_dat.iloc[:, -1]
while best_features.shape[1] > size_limit:
    best_features = tcdp.featureSelection(best_features, art=art_new, logger=logger_new)
best_features.drop(best_features.columns[-1], axis=1, inplace=True)
logger_new.info("Reduce the size of the final selected features to %d by featureSelection" %(best_features.shape[1]))

init_fitness, cur_fitness = tcdp.scoreCompare(dat, best_features, art=art_new, logger=logger_new)
increase = (cur_fitness - init_fitness) / init_fitness
logger_new.debug("At last, compared with the initial one, the fitness increased by %s" %(str(increase)))
best_features.to_csv(datapath + '/final' + '.csv')