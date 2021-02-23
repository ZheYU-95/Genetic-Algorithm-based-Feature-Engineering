import pandas as pd
import tc_distributed as tcd
from dask.distributed import LocalCluster, Client, progress
from dask_jobqueue import HTCondorCluster
import logging, coloredlogs
import sys, os
import copy


"""
# ignore warning
warnings.filterwarnings("ignore")
pd.options.display.max_columns = 50
"""

dataset = 'data/php0iVrYT.csv'

logging.basicConfig(level=logging.ERROR)
# logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger_new = logging.getLogger(__name__)

output_file_handler = logging.FileHandler("log/output_" + dataset[5:-3] + 'log')
output_file_handler.setFormatter(formatter)
logger_new.addHandler(output_file_handler)
coloredlogs.install(level = 'DEBUG')

# stdout_handler = logging.StreamHandler(sys.stdout)
# stdout_handler.setFormatter(formatter)
# logger_new.addHandler(stdout_handler)

# cluster = LocalCluster(n_workers=4, threads_per_worker=2, memory_limit='2GB')
# client = Client(cluster)

cur_dat = tcd.load_data(dataset, logger=logger_new, art='C')
prev_g = None
for i in range(4):
    cur_dat, prev_g = tcd.update_dat(cur_dat, prev_gen=prev_g, logger=logger_new)
    cur_dat.to_csv('result/curr_' + dataset[5:])
    prev_g.to_csv('result/prev_' + dataset[5:])
logger_new.warning('+Successfully finish all steps')

""" logger_new.debug("+Concatenate the last loop 's results as the final dataframe")
cur_g = cur_dat.drop(cur_dat.columns[-1], axis=1, inplace=False)
prev_g = pd.concat([prev_g, cur_g], axis=1)
prev_g[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
prev_g.to_csv('result/res_' + dataset[5:])
logger_new.warning('+Successfully finish all steps') """











# %%
