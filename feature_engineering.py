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


"""
# ignore warning
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50
"""

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
coloredlogs.install(level = 'DEBUG')
logger_new = logging.getLogger(__name__)
output_file_handler = logging.FileHandler("log/output.log")
stdout_handler = logging.StreamHandler(sys.stdout)
logger_new.addHandler(output_file_handler)
logger_new.addHandler(stdout_handler)


dataset = 'data/php0iVrYT.csv'
cur_dat = tc.load_data(dataset, logger=logger_new, art='C')
# print(cur_dat)
prev_g = None
for i in range(5):
    cur_dat, prev_g = tc.update_dat(cur_dat, prev_gen=prev_g, logger=logger_new)
    prev_g.to_csv('result/res_' + dataset[5:])
# print(prev_g)
    
# candidates = tc.generateCandidates(dat, art='C', prev_gen=None, logger=logger_new)
# print(candidates)