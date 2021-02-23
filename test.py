# %%
from math import inf
from numpy.core.numeric import NaN
from pandas.core.frame import DataFrame
from transforms import LinearRegressorTransform, RandomForestClassifierTransform
import numpy as np
import pandas as pd
import copy
import random 
import sys
import logging, coloredlogs
from sklearn.model_selection import train_test_split
import transforms as tf
import tribes_competition as tc
import dask.array as da
import dask.dataframe as dd
import dask.bag as db
from dask import delayed
from dask.distributed import LocalCluster, Client

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
coloredlogs.install(level = 'DEBUG')
logger_new = logging.getLogger(__name__)







""" dates = pd.date_range('20210101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=list('ABCD'))
df = df.drop('C', axis=1, inplace=False)
df1 = pd.DataFrame(np.random.randn(6, 2), index=dates, columns=list('EF')) """

# df2 = pd.DataFrame(df.iloc[:, :-1])
# df1 = pd.DataFrame(np.random.randn(6, 6), index=dates, columns=list('ABCDEF'))
# df2[df1.columns[-1]] = df1.iloc[:, -1]
# print(pd.concat([df1, df2], axis=1, ignore_index=True))



# label = df.iloc[:, -1].value_counts().index[0]
# print(df.reset_index().iloc[:, 1:])
# print(df.reset_index(drop=False))

from dask import compute
import tc_distributed as tcd
import transforms_distributed as tfd
import dask.bag as db
una_oprs = tfd.unary_operators
bina_oprs = tfd.binary_operators
oprs_list = [una_oprs, bina_oprs]



# dataset = 'data/php0iVrYT.csv'
# dat = tcd.load_data(dataset, art='C')


# seq = dat.columns
# b = db.from_sequence(dat[s] for s in seq)
# b.map(lambda x: np.log(x))
# print(b.compute())
# gps = tcd.tribesCompetition(dat)
# cand_d = delayed(tcd.plunge_eliminate)(dat, gps)
# print(compute(cand_d))
# nxt_g = tcd.generateCandidates(dat)
# print(nxt_g)
# print(lis)
# print(compute(lis))

# %%
a = np.array(['abba' for _ in range(186043)], dtype=object)
a.nbytes / (1024 ** 2)


# %%
# 导入数据：这里刚读入的数据里都有第一列的一列索引，要把他删除掉
cur_dat = pd.read_csv('result/curr_php0iVrYT.csv')
cur_dat.drop(cur_dat.columns[0], axis=1, inplace=True)
cur_gen = pd.DataFrame(cur_dat.iloc[:, :-1], columns=cur_dat.columns[:-1])
print(cur_gen.shape)
# print(cur_gen)
prev_dat = pd.read_csv('result/prev_php0iVrYT.csv')
prev_gen = prev_dat.drop(prev_dat.columns[0], axis=1, inplace=False)
print(prev_gen.shape)
# print(prev_gen)
# cur_gen.to_csv('t.csv', mode='a', header=False)



# %%
# 通过调用操作符生成子代特征
cands_bag = tcd.new_cands(cur_gen, prev_gen=prev_gen)
cands_bag = compute(compute(cands_bag)[0])[0]
print(cands_bag)

# cands_bag = compute(cands_bag)[0]
# lis = []
# for i in range(21):
#      result = cands_bag[i].compute()
#      lis.append(result)
#      print("res for %s opr" %(str(i+1)))
#      # print(result)
# print(lis)



# cands_bag = compute(cands_bag.compute())[0]

# %%
#清洗子代特征
""" cands_lis = []
for cand in cands_bag:
    cand = delayed(tcd.clean_dat)(cand)
    cands_lis.append(cand)
cands_lis = compute(cands_lis)[0]
cands_lis """

# %%
#改写eval函数
""" cands_lis = copy.deepcopy(cands_bag)
cand = cands_lis[-4]
lis1 = ['1', '2']
lis2 = ['3', '4']
lis3 = ['5', '6']
lis = [lis1, lis2, lis3]
scores = []
for i in range(3):
     d = delayed(train_test_split)(cand, cur_dat.iloc[:, -1])
     s = tcd.eval(d)
     scores.append(s)
     lis[i].append(s)
# print(scores)
mean_s = sum(scores)/3
mean_s = compute(mean_s)[0]
print(mean_s)
print(lis)
print(compute(lis)[0])
res = sorted(compute(lis)[0], key=lambda x: x[-1], reverse=False)
print(res) """

# %%
""" cand_dat = copy.deepcopy(cand)
cand_dat[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
sorted_gps = tcd.tribesCompetition(cand_dat)
# print(compute(sorted_gps[-1][:-1])[0])
cand_d = delayed(tcd.plunge_eliminate)(cand_dat, sorted_gps)
cand_d = compute(cand_d)[0]
print(cand_d) """

# %%
""" cnt = 1
res_lis = []
for cand in cands_lis:
     if cand.empty:
          continue
     cand_dat = copy.deepcopy(cand)
     cand_dat[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
     # tC 返回的结果是delayed格式的
     sorted_gps = tcd.tribesCompetition(cand_dat)
     # sorted_gps = compute(sorted_gps)[0]
     cand_d = delayed(tcd.plunge_eliminate)(cand_dat, sorted_gps)
     res_lis.append(cand_d)
     cnt += 1
res_lis = compute(res_lis)[0] """






""" candidates_list = []
# cands_bag = tcd.new_cands(dat, prev_gen)
# print(compute(compute(cands_bag)[0])[0])
for oprs in oprs_list:
     for opr in oprs.values():
          candidates = None
          if oprs == oprs_list[0]:
               candidates = delayed(opr()._exec)(cur_gen)
          elif oprs == oprs_list[1]:
               candidates = delayed(opr()._exec)(cur_gen, prev_gen)
          candidates_list.append(candidates)
          # candidates_list.append(candidates.compute())
print(candidates_list[-1].compute())
# print(compute(candidates_list)[0]) """








""" # dat = tc.load_data(dataset, art='C')
dat = pd.read_csv(dataset)
# dat = tc.clean_dat(dat)
dat = tc.load_data(dataset, art='C')
print(dat)
 """

""" dataset = 'data/php0iVrYT.csv'
dat = tc.load_data(dataset, art='C')
X_train, X_test, y_train, y_test = train_test_split(
            dat.iloc[:, :-1], dat.iloc[:, -1])
# res = tc.eval(X_train, X_test, y_train, y_test)

una_oprs = tf.unary_operators
bina_oprs = tf.binary_operators
oprs_list = [una_oprs, bina_oprs]
cur_gen = pd.DataFrame(dat.iloc[:, :-1], columns=dat.columns[:-1])
prev_gen = None
# opr = oprs_list[0].get('adde')
opr = oprs_list[1].get('div')
for i in range(3):
     # candidates = opr()._exec(cur_gen)
     candidates = opr()._exec(cur_gen, prev_gen)
     # print(candidates)
     candidates[dat.columns[-1]] = dat.iloc[:, -1]
     lis = tc.tribesCompetition(candidates)
     candidates = tc.plunge_eliminate(candidates, lis)
     # print(candidates)
     res_limit = 200
     res = pd.DataFrame()
     res = pd.concat([res, candidates], axis=1)
     res[dat.columns[-1]] = dat.iloc[:, -1]
     while (res.shape[1] > res_limit):
          res = tc.feature_selection(res)
     res = res.iloc[:, :-1]
     prev_gen = pd.concat([prev_gen, cur_gen], axis=1)
     print(prev_gen)
     cur_gen = res
     print(cur_gen)
print(prev_gen)
print(cur_gen) """

# print(np.finfo(np.float32).max)












""" df2 = pd.DataFrame({'age': [5, pd.NaT, np.NaN],
                   'born': [pd.NaT, pd.Timestamp('1939-05-27'), pd.Timestamp('1940-04-25')],
                   'name': ['Alfred', 'Batman', np.inf],
                   'toy': [None, 'Batmobile', 'Joker']})
# print(df2.loc[:, df2.isna().sum()/3 < .5])
print(df2)
print((df2.iloc[:, 0].isna()).any()) """



""" 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
x = RandomForestRegressor()
y = x
if 1 > 0:
     x = RandomForestClassifier()
print(x)
print(y) 
"""

""" a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
k = 4
ind = np.argpartition(a, -k)[-k:] """
# print(df['A'].all()!=0)
# df = df.loc[:, (df != df.iloc[0]).any()]
# df = df.iloc[0]
# print((df != df.iloc[0]).any())  # anys() indicating whether any element is True
# print(df.columns)
# print(df.isna().sum().sum())
# print(df.columns)
# print(df)

# lis = [[1,2,3], [4,5,6,2], [7,8,9], [10,11,12,1]]
# print(lis[0][:-1])
# lis2 = sorted(lis, key=lambda x: x[-1], reverse=False)
# print(lis2)


 

"""
coloredlogs.install(level = 'DEBUG')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info('This is a log info')
logger.debug('Debugging')
logger.warning('Warning exists')
logger.fatal('Fatal')
logger.info('Finish')
"""

 

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
""" 
X = [[ 0.87, -1.34,  0.31 ],
     [-2.79, -0.02, -0.85 ],
     [-1.34, -0.48, -2.55 ],
     [ 1.92,  1.48,  0.65 ]]
y = [0, 1, 0, 1]
clf = RandomForestRegressor().fit(X, y)
selector =  SelectFromModel(clf, threshold= 'mean', prefit=True)
selected = selector.get_support()
# print(selected)
df = pd.DataFrame(np.random.randn(len(dates), len(df.loc[:, selected].columns)), index=dates, columns=df.loc[:, selected].columns)
print(df)
"""

import gopup as gp
""" 
df_index = gp.marco_cmlrd()
# print(df_index)
df_index.to_excel("中国杠杆率.xlsx") 
"""

# %%
