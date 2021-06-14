import pandas as pd
import numpy as np
import os, sys
from joblib import Parallel, delayed, parallel_backend
from dask.distributed import Client, progress
from dask_jobqueue import HTCondorCluster
import pickle
import sklearn
import warnings
import copy
import logging, coloredlogs
import tribes_competition as tc
import tc_distributed_pro as tcdp
import transforms as tf
import transforms_distributed as tfd



logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_new = logging.getLogger(__name__)
output_file_handler = logging.FileHandler("log/time_compare_German.log")
output_file_handler.setFormatter(formatter)
logger_new.addHandler(output_file_handler)
coloredlogs.install(level = 'DEBUG')
# pathlib.Path(datapath).mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

dataset = 'data/German_Credit.csv'
art_new = 'C'
# import data
dat = tc.load_data(dataset, logger=logger_new, art=art_new)

repeat = 3
una_oprs = tf.unary_operators
bina_oprs = tf.binary_operators
weights = [repeat] * (len(una_oprs) + len(bina_oprs))
# import data
dat = tc.load_data(dataset, logger=logger_new, art=art_new)
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
    num = 50
    dat_sel = tc.bestFeatures(dat, num, art=art_new)
    cur_dat = dat_sel
    cur_dat[dat.columns[-1]] = dat.iloc[:, -1]
    group = cur_dat.drop(cur_dat.columns[-1], axis=1, inplace=False)

# generate and store candidate features
for i in range(repeat):
    cur_dat, gen = tc.updateDat(cur_dat, prev_gen=prev_gen, oprs_weights=weights, art=art_new, logger=logger_new)
    cur_gen, cur_size = tc.constrainFeaturesNum(cur_dat, min(inflation*cur_size, cur_limit), art=art_new, logger=logger_new)
    logger_new.debug('The number of features in cur-gen drops from %d to %d' %(cur_dat.shape[1]-1, cur_size))

    prev_gen = pd.concat([prev_gen, gen], axis=1)
    prev_gen[dat.columns[-1]] = dat.iloc[:, -1]
    prev_gen, prev_size = tc.constrainFeaturesNum(prev_gen, total_limit, art=art_new, logger=logger_new)
    logger_new.debug('The number of features in prev-gen drops from %d to %d' %(prev_gen.shape[1], prev_size))

    if dat_sel is not None:
        cur_dat = tc.addInitalFeatures(cur_gen, prev_gen, dat_sel, logger=logger_new)
    else:
        cur_dat = tc.addInitalFeatures(cur_gen, prev_gen, dat, logger=logger_new)
    cur_dat.to_csv('result/gen' + str(i+1) + '.csv')
