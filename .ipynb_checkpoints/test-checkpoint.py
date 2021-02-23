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

dataset = 'data/php0iVrYT.csv'
# dat = tc.load_data(dataset, art='C')
dat = pd.read_csv(dataset)
# dat = tc.clean_dat(dat)
dat = tc.load_data(dataset, art='C')
print(dat)


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

a = np.array([9, 4, 4, 3, 3, 9, 0, 4, 6, 0])
k = 4
ind = np.argpartition(a, -k)[-k:]
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
