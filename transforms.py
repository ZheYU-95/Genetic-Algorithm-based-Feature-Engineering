from abc import ABCMeta, abstractmethod
import sys
import traceback
from numpy.core.numeric import NaN

from numpy.ma.core import append, reshape
from torch.utils import data
from state import State
import copy

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from concurrent.futures import ThreadPoolExecutor
from scipy.stats import ttest_ind
from multiprocessing import cpu_count
from sklearn.cluster import KMeans, DBSCAN, AffinityPropagation
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel, VarianceThreshold
from sklearn.gaussian_process import GaussianProcessRegressor, GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RationalQuadratic, Exponentiation, RBF
from sklearn.kernel_approximation import RBFSampler
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.linear_model import LinearRegression, SGDClassifier
from sklearn.metrics import roc_auc_score, mean_absolute_error, accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.neighbors import NearestNeighbors
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures, \
    QuantileTransformer, PowerTransformer
from sklearn.svm import SVR, SVC, LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
import xgboost as xgb


class Transform(metaclass=ABCMeta):
    @abstractmethod
    def fit(self, X_train, X_test=None, y_train=None, y_test=None):
        """
        X_train, type of DataFrame
        X_test, type of DataFrame
        """
        pass

def valid_col(col: pd.Series) -> bool:
    """ if there are inf, NaN, number out of range of float32, too many large/small numbers,  this new_col should be discarded"""
    if (col.isna().any() or
        (col == np.inf).any() or
        (col > np.finfo(np.float32).max).any() or
        (col < np.finfo(np.float32).min).any() or
        (abs(col - 0.0) < 0.0001).sum() / len(col) > 0.8):
        return False
    return True

## type A

class Abs(Transform):
    def __init__(self):
        super(Abs, self).__init__()
        self.name = 'abs'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = self._exec(X_train)
        res1 = pd.concat([X_train, tmp1], axis=1)
        if type(X_test) != type(None):
            tmp2 = self._exec(X_test)
            res2 = pd.concat([X_test, tmp2], axis=1)
        else:
            res2 = None
        return res1, res2

    def _exec(self, dat: pd.DataFrame):
        res = pd.DataFrame()
        for col in dat.columns:
            if (dat[col].values < 0).any():
                res[str(col)+ '_abs'] = dat[col].abs()
        return res
        


class Adde(Transform):
    def __init__(self):
        super(Adde, self).__init__()
        self.name = 'adde'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_adde'] = dat[i].map(lambda x: x + np.e)
        return res


class Cos(Transform):
    def __init__(self):
        super(Cos, self).__init__()
        self.name = 'cos'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_cos'] = dat[i].astype(float).map(lambda x: np.cos(x))
        return res


class Degree(Transform):
    def __init__(self):
        super(Degree, self).__init__()
        self.name = 'degree'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_degree'] = np.degrees(dat[i])
        return res


class Exp(Transform):
    """
    only do exp when all the x <= 1
    """

    def __init__(self):
        super(Exp, self).__init__()
        self.name = 'exp'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = self._exec(X_train)
        res1 = pd.concat([X_train, tmp1], axis=1)
        if type(X_test) != type(None):
            tmp2 = self._exec(X_test)
            res2 = pd.concat([X_test, tmp2], axis=1)
        else:
            res2 = None
        return res1, res2

    def _exec(self, dat: pd.DataFrame()):
        res = pd.DataFrame()
        for col in dat.columns:
            if (dat[col].values < 1).all():
                res[str(col)+ '_exp'] = dat[col].map(lambda x: np.exp(x))
        return res


class Ln(Transform):
    def __init__(self):
        super(Ln, self).__init__()
        self.name = 'ln'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for col in dat.columns:
            if (dat[col].values > 0).all():
                res[str(col) + '_ln'] = dat[col].map(lambda x: np.log(x))
        return res


class Negative(Transform):
    def __init__(self):
        super(Negative, self).__init__()
        self.name = 'negative'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_negative'] = -dat[i]
        return res


class Radian(Transform):
    def __init__(self):
        super(Radian, self).__init__()
        self.name = 'radian'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_radian'] = np.radians(dat[i])
        return res


class Reciprocal(Transform):
    def __init__(self):
        super(Reciprocal, self).__init__()
        self.name = 'reciprocal'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_recip'] = dat[i].map(lambda x: 1 / x if x != 0 else x)
        return res


class Sin(Transform):
    def __init__(self):
        super(Sin, self).__init__()
        self.name = 'sin'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_sin'] = dat[i].map(lambda x: np.sin(x))
        return res


class Sigmoid(Transform):
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.name = 'sigmoid'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_sigmoid'] = dat[i].map(lambda x: 1 / (1 + np.exp(-x)))
        return res


class Square(Transform):
    def __init__(self):
        super(Square, self).__init__()
        self.name = 'square'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            new_col = dat[i].map(lambda x: x * x)
            if (not valid_col(new_col)):
                continue
            res[str(i) + '_square'] = new_col
        return res
    
    
""" class Sqrt(Transform):
    def __init__(self):
        super(Sqrt, self).__init__()
        self.name = 'sqrt'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_sqrt'] = dat[i].map(lambda x: x ** (.5))
        return res """


class Tanh(Transform):
    def __init__(self):
        super(Tanh, self).__init__()
        self.name = 'tanh'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat: pd.DataFrame):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_tanh'] = dat[i].map(lambda x: np.tanh(x))
        return res


class Relu(Transform):
    def __init__(self):
        super(Relu, self).__init__()
        self.name = 'relu'
        self.type = 1

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1)
        else:
            tmp2 = None
        return tmp1, tmp2

    def _exec(self, dat):
        res = pd.DataFrame()
        for i in dat.columns:
            res[str(i) + '_relu'] = dat[i].map(lambda x: x * (x > 0))
        return res


class Minmaxnorm(Transform):
    def __init__(self):
        super(Minmaxnorm, self).__init__()
        self.name = 'mmnorm'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train_m = self._exec(X_train)
        X_train = pd.concat([X_train, X_train_m], axis=1)
        if type(X_test) != type(None):
            X_test_m = self._exec(X_test)
            X_test = pd.concat([X_test, X_test_m], axis=1)
        return X_train, X_test
    
    def _exec(self, dat: pd.DataFrame):
        scaler = MinMaxScaler()
        res = pd.DataFrame(scaler.fit_transform(dat), index=dat.index, 
                           columns=[str(i) + '_mmnorm' for i in dat.columns])
        return res
        


class Zscore(Transform):
    def __init__(self):
        super(Zscore, self).__init__()
        self.name = 'zscore'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train_z = self._exec(X_train)
        X_train = pd.concat([X_train, X_train_z], axis=1)
        if type(X_test) != type(None):
            X_test_z = self._exec(X_test)
            X_test = pd.concat([X_test, X_test_z], axis=1)
        return X_train, X_test
    
    def _exec(self, dat: pd.DataFrame):
        scaler = StandardScaler()
        res = pd.DataFrame(scaler.fit_transform(dat), index=dat.index, 
                           columns=[str(i) for i in dat.columns + '_zscore'])
        return res


class KernelApproxRBF(Transform):
    def __init__(self):
        super(KernelApproxRBF, self).__init__()
        self.name = 'kernelapproxrbf'
        self.type = 3

    def fit(self, X_train, X_test=None):
        # rbf sampler
        X_train_t = self._exec(X_train)
        # print(X_train.head())
        X_train = pd.concat([X_train, X_train_t], axis=1)
        if type(X_test) != type(None):
            X_test_t = self._exec(X_test)
            X_test = pd.concat([X_test, X_test_t], axis=1)
        return X_train, X_test
    
    def _exec(self, dat: pd.DataFrame):
        rbf_feature = RBFSampler(gamma=1, n_components=dat.shape[1])
        res = pd.DataFrame(rbf_feature.fit_transform(dat), index=dat.index)
        cols = []
        # for i in range(res.shape[1]):
        for i in dat.columns:
            cols.append(str(i) + '_rbfFeature')
        res.columns = cols
        return res


## type B
class Div(Transform):
    def __init__(self):
        self.name = 'div'
        super(Div, self).__init__()
        self.type = 2

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        tmp2 = None
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1) 
        return tmp1, tmp2

    def _exec(self, dat: pd.DataFrame, prev_dat: pd.DataFrame=None):
        cols = dat.columns
        res = pd.DataFrame()
        for i in range(len(cols)):
            for j in np.arange(i+1, len(cols)):
                if dat[cols[j]].any() == 0:
                    continue
                new_col = dat[cols[i]] / dat[cols[j]]
                if (not valid_col(new_col)):
                    continue
                res['(' + str(cols[i]) + '/' + str(cols[j]) + ')'] = new_col
        if type(prev_dat) != type(None):
            cols2 = prev_dat.columns
            for i in range(len(cols)):
                for j in range(len(cols2)):
                    if prev_dat[cols2[j]].any()==0:
                        continue
                    new_col = dat[cols[i]] / prev_dat[cols2[j]]
                    if (not valid_col(new_col)):
                        continue
                    res['(' + str(cols[i]) + '/' + str(cols2[j]) + ')'] = new_col
        return res


class Minus(Transform):
    def __init__(self):
        super(Minus, self).__init__()
        self.name = 'minus'
        self.type = 2

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        tmp2 = None
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1) 
        return tmp1, tmp2

    def _exec(self, dat: pd.DataFrame, prev_dat: pd.DataFrame=None):
        res = pd.DataFrame()
        cols = dat.columns
        for i in range(len(cols)):
            for j in np.arange(i+1, len(cols)):
                new_col = dat[cols[i]] - dat[cols[j]]
                if (not valid_col(new_col)):
                    continue
                res['(' + str(cols[i]) + '-' + str(cols[j]) + ')'] = new_col
        if type(prev_dat) != type(None):
            cols2 = prev_dat.columns
            for i in range(len(cols)):
                for j in range(len(cols2)):
                    new_col = dat[cols[i]] - prev_dat[cols2[j]]
                    if (not valid_col(new_col)):
                        continue
                    res['(' + str(cols[i]) + '-' + str(cols2[j]) + ')'] = new_col
        return res


class Add(Transform):
    def __init__(self):
        super(Add, self).__init__()
        self.name = 'add'
        self.type = 2

    def fit(self, X_train, X_test=None):
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        tmp2 = None
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1) 
        return tmp1, tmp2

    def _exec(self, dat: pd.DataFrame, prev_dat: pd.DataFrame=None):
        cols = dat.columns
        res = pd.DataFrame()
        for i in range(len(cols)):
            for j in np.arange(i, len(cols)):
                new_col = dat[cols[i]] + dat[cols[j]]
                if (not valid_col(new_col)):
                    continue
                res['(' + str(cols[i]) + '+' + str(cols[j]) + ')'] = new_col
        if type(prev_dat) != type(None):
            cols2 = prev_dat.columns
            for i in range(len(cols)):
                for j in range(len(cols2)):
                    new_col = dat[cols[i]] + prev_dat[cols2[j]]
                    if (not valid_col(new_col)):
                        continue
                    res['(' + str(cols[i]) + '+' + str(cols2[j]) + ')'] = new_col
        return res


class Product(Transform):
    def __init__(self):
        super(Product, self).__init__()
        self.name = 'product'
        self.type = 2

    def fit(self, X_train, X_test=None):
        # poly = PolynomialFeatures(degree=2)
        # X_train = pd.DataFrame(poly.fit_transform(X_train), columns=poly.get_feature_names(input_features=X_train.columns), index=X_train.index)
        # X_test = pd.DataFrame(poly.fit_transform(X_test), columns=poly.get_feature_names(input_features=X_test.columns), index=X_test.index)
        tmp1 = pd.concat([X_train, self._exec(X_train)], axis=1)
        tmp2 = None
        if type(X_test) != type(None):
            tmp2 = pd.concat([X_test, self._exec(X_test)], axis=1) 
        return tmp1, tmp2
    
    def _exec(self, dat: pd.DataFrame, prev_dat: pd.DataFrame=None):
        cols = dat.columns
        res = pd.DataFrame()
        for i in range(len(cols)):
            for j in np.arange(i+1, len(cols)):
                new_col = dat[cols[i]] * dat[cols[j]]
                if not valid_col(new_col):
                    continue
                res['(' + str(cols[i]) + '*' + str(cols[j]) + ')'] = new_col
        if type(prev_dat) != type(None):
            cols2 = prev_dat.columns
            for i in range(len(cols)):
                for j in range(len(cols2)):
                    new_col = dat[cols[i]] * prev_dat[cols2[j]]
                    if not valid_col(new_col):
                        continue
                    res['(' + str(cols[i]) + '*' + str(cols2[j]) + ')'] = new_col
        return res


# type C, multi -> 1
class Timeagg(Transform):
    def __init__(self):
        super(Timeagg, self).__init__()
        self.name = 'timeagg'
        self.type = 1
        self.actions = [
            Percentile25(),
            Percentile50(),
            Percentile75(),
            Std(),
            Maximum(),
            Minimum(), ]
        self.type = 3

    def fit(self, X_train, X_test=None):
        for i in self.actions:
            X_train, X_test = i.fit(X_train, X_test)
        return X_train, X_test


class Percentile25(Transform):
    def __init__(self):
        super(Percentile25, self).__init__()
        self.name = 'p25'
        self.type = 3

    def fit(self, X_train, X_test=None):
        # 第p个百分位数是这样一个值，它使得至少有p%的数据项小于或等于这个值
        X_train.loc[:, '-'.join(X_train.columns) + '_25%'] = np.percentile(X_train, 25, axis=1)
        if type(X_test) != type(None):
            X_test.loc[:, '-'.join(X_test.columns) + '_25%'] = np.percentile(X_test, 25, axis=1)
        return X_train, X_test


class Percentile50(Transform):
    def __init__(self):
        super(Percentile50, self).__init__()
        self.name = 'p50'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train.loc[:, '-'.join(X_train.columns) + '_50%'] = np.percentile(X_train, 50, axis=1)
        if type(X_test) != type(None):
            X_test.loc[:, '-'.join(X_test.columns) + '_50%'] = np.percentile(X_test, 50, axis=1)
        return X_train, X_test


class Percentile75(Transform):
    def __init__(self):
        super(Percentile75, self).__init__()
        self.name = 'p75'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train.loc[:, '-'.join(X_train.columns) + '_75%'] = np.percentile(X_train, 75, axis=1)
        if type(X_test) != type(None):
            X_test.loc[:, '-'.join(X_test.columns) + '_75%'] = np.percentile(X_test, 75, axis=1)
        return X_train, X_test


class Std(Transform):
    def __init__(self):
        super(Std, self).__init__()
        self.name = 'std'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train.loc[:, '-'.join(X_train.columns) + '_std'] = np.std(X_train, axis=1)
        if type(X_test) != type(None):
            X_test.loc[:, '-'.join(X_test.columns) + '_std'] = np.std(X_test, axis=1)
        return X_train, X_test


class Maximum(Transform):
    def __init__(self):
        super(Maximum, self).__init__()
        self.name = 'max'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train.loc[:, '-'.join(X_train.columns) + '_max'] = np.max(X_train, axis=1)
        if type(X_test) != type(None):
            X_test.loc[:, '-'.join(X_test.columns) + '_max'] = np.max(X_test, axis=1)
        return X_train, X_test


class Minimum(Transform):
    def __init__(self):
        super(Minimum, self).__init__()
        self.name = 'min'
        self.type = 3

    def fit(self, X_train, X_test=None):
        X_train.loc[:, '-'.join(X_train.columns) + '_min'] = np.min(X_train, axis=1)
        if type(X_test) != type(None):
            X_test.loc[:, '-'.join(X_test.columns) + '_min'] = np.min(X_test, axis=1)
        return X_train, X_test


class Autoencoder(Transform):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.name = 'autoencoder'
        self.type = 3

    def fit(self, X_train, X_test):
        return X_train, X_test
        # autoencoder
        index_train = X_train.index
        index_test = X_test.index
        dim_in = X_train.shape[1]
        encoding_dim = 32
        input_dat = Input(shape=(dim_in,))
        encoded = Dense(encoding_dim, activation='relu')(input_dat)
        decoded = Dense(dim_in, activation='sigmoid')(encoded)
        autoencoder = Model(input_dat, decoded)
        encoder = Model(input_dat, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='mse')
        autoencoder.fit(X_train, X_train,
                        epochs=50,
                        batch_size=16,
                        shuffle=True,
                        validation_data=(X_test, X_test),
                        verbose=True)
        X_train_ae = pd.DataFrame(encoder.predict(X_train), index=index_train,
                                  columns=['ae_' + str(i) for i in range(encoding_dim)])
        X_test_ae = pd.DataFrame(encoder.predict(X_test), index=index_test,
                                 columns=['ae_' + str(i) for i in range(encoding_dim)])
        X_train = pd.concat([X_train, X_train_ae], axis=1)
        X_test = pd.concat([X_test, X_test_ae], axis=1)
        return X_train, X_test


# type D, need help of y
### fail noch the fit method for type D and type E
class Clustering(Transform):
    def __init__(self):
        """
        分组做clustering
        """
        super(Clustering, self).__init__()
        self.name = 'clustering'
        self.type = 4

    def fit(self, X_train, X_test=None):
        # preprocessing before transformation
        # 没有分开做，主要是因为麻烦
        num_c = 16
        
        cluster = KMeans(n_clusters=num_c).fit(X_train)
        X_train['clustering_whole'] = cluster.predict(X_train)
        X_test['clustering_whole'] = cluster.predict(X_test)
        return X_train, X_test


class Binning(Transform):
    def __init__(self):
        """
        clustering for each feature, need predefined k
        only _execute when there are less than 100 cols
        """
        super(Binning, self).__init__()
        self.name = 'binning'
        self.type = 2  # should be type 4, set it to type 2 to limit the number of columns

    def fit(self, X_train, X_test=None):
        # preprocessing before transformation
        # replace inf with na
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        # remove columns with more than 1/3 na
        X_train = X_train.loc[:, (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
        X_test = X_test.loc[:, (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
        # fill na
        if X_train.isna().sum().sum() > 0:
            columns = X_train.columns
            index = X_train.index
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_train = imp.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns=columns, index=index)
        if X_test.isna().sum().sum() > 0:
            columns = X_test.columns
            index = X_test.index
            # imp = IterativeImputer(max_iter=10, random_state=0)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_test = imp.fit_transform(X_test)
            X_test = pd.DataFrame(X_test, columns=columns, index=index)
        # do clustering for each column
        for i in X_train.columns:
            if len(set(X_train[[i]])) <= 16:
                cluster = KMeans(n_clusters=len(set(X_train[[i]]))).fit(X_train[[i]])
            else:
                cluster = KMeans(n_clusters=16).fit(X_train[[i]])
            X_train['binning_' + str(i)] = cluster.predict(X_train[[i]])
            X_test['binning_' + str(i)] = cluster.predict(X_test[[i]])
        return X_train, X_test


class Diff(Transform):
    def __init__(self):
        """
        diff between the columns
        """
        super(Diff, self).__init__()
        self.name = 'diff'
        self.type = 2

    def fit(self, X_train, X_test=None):
        X_train = self.get_diff(X_train)
        X_test = self.get_diff(X_test)
        return X_train, X_test

    def get_diff(self, dat):
        dat_diff = dat.diff(axis=1)
        dat_diff.columns = ['diff_' + str(i) for i in dat_diff.columns]
        return pd.concat([dat, dat_diff], axis=1)


class TimeBinning(Transform):
    def __init__(self):
        super(TimeBinning, self).__init__()
        self.name = 'timeBinning'
        self.type = 4

    def fit(self, X_train, X_test=None):
        return X_train, X_test


class TempWinAgg(Transform):
    def __init__(self):
        super(TempWinAgg, self).__init__()
        self.name = 'tempWinAgg'
        self.type = 4
        self.win = 5

    def fit(self, X_train, X_test=None):
        X_train = self.get_tempWinAgg(X_train)
        X_test = self.get_tempWinAgg(X_test)
        return X_train, X_test

    def get_tempWinAgg(self, dat):
        cols = dat.columns
        dat_std = dat.rolling(self.win, axis=1).std()
        dat_std.columns = ['tempWinAgg_std_' + str(i) for i in cols]
        dat_max = dat.rolling(self.win, axis=1).max()
        dat_max.columns = ['tempWinAgg_max_' + str(i) for i in cols]
        dat_min = dat.rolling(self.win, axis=1).min()
        dat_min.columns = ['tempWinAgg_min_' + str(i) for i in cols]
        dat_mean = dat.rolling(self.win, axis=1).mean()
        dat_mean.columns = ['tempWinAgg_mean_' + str(i) for i in cols]
        return pd.concat([dat, dat_std, dat_max, dat_min, dat_mean], axis=1)


class SpatialAgg(Transform):
    def __init__(self):
        """
        AffinityPropagation
        """
        super(SpatialAgg, self).__init__()
        self.name = 'spatialAgg'
        self.type = 4

    def fit(self, X_train, X_test=None):
        # DBSCAN
        num_cols = X_train.shape[1]
        step = 4
        # shift 1
        if num_cols < step:
            block_start = list(np.arange(0, num_cols, 1))
        else:
            block_start = list(np.arange(0, num_cols, int(num_cols / step)))
        block_end = block_start[1:]
        block_end.append(num_cols)
        for i, j in zip(block_start, block_end):
            if i >= j:
                continue
            cluster = AffinityPropagation().fit(X_train.iloc[:, i: j])
            X_train['spatialAgg_step_' + str(i) + '_' + str(j)] = cluster.predict(X_train.iloc[:, i: j])
            X_test['spatialAgg_step_' + str(i) + '_' + str(j)] = cluster.predict(X_test.iloc[:, i: j])
        # shift 2
        block_end2 = block_start[2:]
        block_end2.append(num_cols)
        block_start2 = block_start[:-1]
        for i, j in zip(block_start2, block_end2):
            if i >= j:
                continue
            cluster = AffinityPropagation().fit(X_train.iloc[:, i: j])
            X_train['spatialAgg_step_' + str(i) + '_' + str(j)] = cluster.predict(X_train.iloc[:, i: j])
            X_test['spatialAgg_step_' + str(i) + '_' + str(j)] = cluster.predict(X_test.iloc[:, i: j])
        # shift 3
        block_end3 = block_start[3:]
        block_end3.append(num_cols)
        block_start3 = block_start[:-2]
        for i, j in zip(block_start3, block_end3):
            if i >= j:
                continue
            cluster = AffinityPropagation().fit(X_train.iloc[:, i: j])
            X_train['spatialAgg_step_' + str(i) + '_' + str(j)] = cluster.predict(X_train.iloc[:, i: j])
            X_test['spatialAgg_step_' + str(i) + '_' + str(j)] = cluster.predict(X_test.iloc[:, i: j])
        # whole
        cluster = AffinityPropagation().fit(X_train)
        X_train['spatialAgg_whole'] = cluster.predict(X_train)
        X_test['spatialAgg_whole'] = cluster.predict(X_test)
        return X_train, X_test


class SpatioAgg(Transform):
    def __init__(self):
        super(SpatioAgg, self).__init__()
        self.name = 'spatioAgg'
        self.type = 4

    def fit(self, X_train, X_test=None):
        return X_train, X_test


class KTermFreq(Transform):
    def __init__(self):
        super(KTermFreq, self).__init__()
        self.name = 'ktermFreq'
        self.type = 4

    def fit(self, X_train, X_test=None):
        for i in X_train.columns:
            tmp = X_train[i].value_counts()
            X_train['ktermfreq_' + str(i)] = X_train[i].map(lambda x: tmp[x] if x in tmp.index else 0)
            X_test['ktermfreq_' + str(i)] = X_test[i].map(lambda x: tmp[x] if x in tmp.index else 0)
        return X_train, X_test


class QuanTransform(Transform):
    def __init__(self):
        super(QuanTransform, self).__init__()
        self.name = 'quanTransform'
        self.type = 4

    def fit(self, X_train, X_test=None):
        # preprocessing before transformation
        # replace inf with na
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        # remove columns with more than 1/3 na
        X_train = X_train.loc[:, (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
        X_test = X_test.loc[:, (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
        # fill na
        if X_train.isna().sum().sum() > 0:
            columns = X_train.columns
            index = X_train.index
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_train = imp.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns=columns, index=index)
        if X_test.isna().sum().sum() > 0:
            columns = X_test.columns
            index = X_test.index
            # imp = IterativeImputer(max_iter=10, random_state=0)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_test = imp.fit_transform(X_test)
            X_test = pd.DataFrame(X_test, columns=columns, index=index)
        # Quantile Transformer
        scaler = QuantileTransformer()
        X_train_z = pd.DataFrame(scaler.fit_transform(X_train), index=X_train.index,
                                 columns=['quanTransform' + str(i) for i in X_train.columns])
        X_test_z = pd.DataFrame(scaler.fit_transform(X_test), index=X_test.index,
                                columns=['quanTransform' + str(i) for i in X_test.columns])
        X_train = pd.concat([X_train, X_train_z], axis=1)
        X_test = pd.concat([X_test, X_test_z], axis=1)
        return X_train, X_test


class NominalExpansion(Transform):
    def __init__(self):
        super(NominalExpansion, self).__init__()
        self.name = 'nominalExpansion'
        self.type = 2
        self.degree = 2

    def fit(self, X_train, X_test=None):
        # preprocessing before transformation
        # replace inf with na
        try:
            X_train = X_train.replace([np.inf, -np.inf], np.nan)
            X_test = X_test.replace([np.inf, -np.inf], np.nan)
            # remove columns with more than 1/3 na
            X_train = X_train.loc[:,
                      (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
            X_test = X_test.loc[:,
                     (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
            # fill na
            if X_train.isna().sum().sum() > 0:
                columns = X_train.columns
                index = X_train.index
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                X_train = imp.fit_transform(X_train)
                X_train = pd.DataFrame(X_train, columns=columns, index=index)
            if X_test.isna().sum().sum() > 0:
                columns = X_test.columns
                index = X_test.index
                # imp = IterativeImputer(max_iter=10, random_state=0)
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                X_test = imp.fit_transform(X_test)
                X_test = pd.DataFrame(X_test, columns=columns, index=index)
            # poly transform
            if X_train.shape[1] <= 10:
                poly = PolynomialFeatures(degree=3, interaction_only=True)
                poly = PolynomialFeatures(degree=3, interaction_only=True)
            else:
                poly = PolynomialFeatures(degree=self.degree, interaction_only=True)
                poly = PolynomialFeatures(degree=self.degree, interaction_only=True)
            X_train = pd.DataFrame(poly.fit_transform(X_train),
                                   columns=poly.get_feature_names(input_features=[str(i) for i in X_train.columns]),
                                   index=X_train.index)
            X_test = pd.DataFrame(poly.fit_transform(X_test),
                                  columns=poly.get_feature_names(input_features=[str(i) for i in X_test.columns]),
                                  index=X_test.index)
        except Exception as e:
            ex_type, ex_value, ex_traceback = sys.exc_info()
            # Extract unformatter stack traces as tuples
            trace_back = traceback.extract_tb(ex_traceback)
            # Format stacktrace
            stack_trace = list()
            for trace in trace_back:
                stack_trace.append(
                    "File : %s , Line : %d, Func.Name : %s, Message : %s" % (trace[0], trace[1], trace[2], trace[3]))
            print('Error appear:')
            print(State.get_state()) 
            self.logger.error(e, exc_info=True)
            print("Stack trace : %s" % stack_trace)
            return X_train, X_test
        return X_train, X_test


class IsoMap(Transform):
    def __init__(self):
        super(IsoMap, self).__init__()
        self.name = 'isomap'
        self.type = 3

    def fit(self, X_train, X_test=None):
        # replace inf with na
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        # remove columns with more than 1/3 na
        X_train = X_train.loc[:, (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
        X_test = X_test.loc[:, (X_train.isna().sum() / len(X_train) < .3) & (X_test.isna().sum() / len(X_test) < .3)]
        # fill na
        if X_train.isna().sum().sum() > 0:
            columns = X_train.columns
            index = X_train.index
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_train = imp.fit_transform(X_train)
            X_train = pd.DataFrame(X_train, columns=columns, index=index)
        if X_test.isna().sum().sum() > 0:
            columns = X_test.columns
            index = X_test.index
            # imp = IterativeImputer(max_iter=10, random_state=0)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            X_test = imp.fit_transform(X_test)
            X_test = pd.DataFrame(X_test, columns=columns, index=index)
        # print('#'*10)
        # print(X_test)
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_z = pd.DataFrame(scaler.transform(X_train), index=X_train.index, columns=X_train.columns)
        X_test_z = pd.DataFrame(scaler.transform(X_test), index=X_test.index, columns=X_test.columns)
        # print(X_test_z)
        # isomap
        nc = 16
        if min(X_train.shape) < nc:
            nc = min(X_train.shape)
        embedding = Isomap(n_components=nc).fit(X_train_z)
        X_train_t = pd.DataFrame(embedding.transform(X_train_z), index=X_train_z.index,
                                 columns=['isomap_' + str(i) for i in range(nc)])
        X_test_t = pd.DataFrame(embedding.transform(X_test_z), index=X_test_z.index,
                                columns=['isomap_' + str(i) for i in range(nc)])
        # print(X_test_t)
        X_train = pd.concat([X_train, X_train_t], axis=1)
        X_test = pd.concat([X_test, X_test_t], axis=1)
        # print(X_test)
        return X_train, X_test


### type E: feature engineering with model

class DecisionTreeClassifierTransform(Transform):
    def __init__(self):
        super(DecisionTreeClassifierTransform, self).__init__()
        self.name = 'decisionTreeClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        ### extract nodes sample
        reg = DecisionTreeClassifier(max_depth=4).fit(X_train, y_train)
        # extract information from tree
        dec_paths = reg.decision_path(X_train).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_train_samples = pd.DataFrame(tmp / tmp[0][0], index=X_train.index,
                                       columns=['dt_n_cla_' + str(i) for i in range(tmp.shape[1])])
        dec_paths = reg.decision_path(X_test).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_test_samples = pd.DataFrame(tmp / tmp[0][0], index=X_test.index,
                                      columns=['dt_n_cla_' + str(i) for i in range(tmp.shape[1])])
        X_train = pd.concat([X_train, X_train_samples], axis=1)
        X_test = pd.concat([X_test, X_test_samples], axis=1)
        ### train dt model and predict
        reg = DecisionTreeClassifier().fit(X_train, y_train)
        # predict
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_cla_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['dt_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = DecisionTreeClassifier().fit(X_train, y_train == X_train_pred['dt_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_cla_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        ### extract information from tree
        return X_train, X_test


class DecisionTreeRegressorTransform(Transform):
    def __init__(self):
        super(DecisionTreeRegressorTransform, self).__init__()
        self.name = 'decisionTreeRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        ### extract nodes sample
        reg = DecisionTreeRegressor(max_depth=4).fit(X_train, y_train)
        # extract information from tree
        dec_paths = reg.decision_path(X_train).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_train_samples = pd.DataFrame(tmp / tmp[0][0], index=X_train.index,
                                       columns=['dt_n_reg_' + str(i) for i in range(tmp.shape[1])])
        dec_paths = reg.decision_path(X_test).toarray()
        tmp = dec_paths.sum(axis=0) * dec_paths
        X_test_samples = pd.DataFrame(tmp / tmp[0][0], index=X_test.index,
                                      columns=['dt_n_reg_' + str(i) for i in range(tmp.shape[1])])
        X_train = pd.concat([X_train, X_train_samples], axis=1)
        X_test = pd.concat([X_test, X_test_samples], axis=1)
        ### train dt model and predict
        reg = DecisionTreeClassifier().fit(X_train, y_train)
        # predict
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_reg_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = DecisionTreeClassifier().fit(X_train, y_train - X_train_pred['dt_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['dt_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['dt_reg_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        ### extract information from tree
        return X_train, X_test


class LeakyInfo(Transform):
    def __init__(self):
        super(LeakyInfo, self).__init__()
        self.name = 'leakyInfo'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        tmp_train = copy.deepcopy(X_train)
        tmp_test = copy.deepcopy(X_test)
        if X_train.shape[1] >= 10:
            tmp_train, tmp_test = self._feature_selection(tmp_train, tmp_test, y_train)
        out_train = self._do_leaky(tmp_train)
        out_test = self._do_leaky(tmp_test)
        tmp1 = pd.concat([X_train, out_train], axis=1)
        tmp2 = pd.concat([X_test, out_test], axis=1)
        return tmp1, tmp2

    def _feature_selection(self, train, test, y_train):
        # feature selection
        clf = RandomForestRegressor().fit(train, y_train)
        fs = SelectFromModel(clf, threshold=-np.inf, prefit=True, max_features=10)
        supp = fs.get_support()
        train = pd.DataFrame(fs.transform(train), columns=train.columns[supp], index=train.index)
        test = pd.DataFrame(fs.transform(test), columns=test.columns[supp], index=test.index)
        return train, test

    def _do_leaky(self, dat):
        out = pd.DataFrame()
        print(dat.shape)
        for i in dat.columns:
            cols = dat.columns.to_list()
            cols.remove(i)
            dat_x = dat[cols]
            dat_y = dat[i]
            svr = SVR()
            svr.fit(dat_x, dat_y)
            out[str(i) + '_leaky'] = svr.predict(dat_x) - dat_y
        return out


class LinearRegressorTransform(Transform):
    def __init__(self):
        super(LinearRegressorTransform, self).__init__()
        self.name = 'linearRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = LinearRegression().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['linear_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['linear_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = LinearRegression().fit(X_train, y_train - X_train_pred['linear_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['linear_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['linear_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class MLPClassifierTransform(Transform):
    def __init__(self):
        super(MLPClassifierTransform, self).__init__()
        self.name = 'mlpClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        train_dataset = Data_rb_cla(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        model = MLP(X_train.shape[1], hidden_size=32, output_size=len(y_train.unique()))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.CrossEntropyLoss()
        model = self._train_model(model, epochs=100, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
        X_train_pred = pd.DataFrame(model.layers(torch.from_numpy(X_train.to_numpy()).float()).data.numpy(),
                                    index=X_train.index,
                                    columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        X_test_pred = pd.DataFrame(model.layers(torch.from_numpy(X_test.to_numpy()).float()).data.numpy(),
                                   index=X_test.index,
                                   columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        return X_train, X_test

    def _train_model(self, model, epochs, train_loader, optimizer, loss_fn):
        for epoch in range(epochs):
            model.train()
            losses = []
            for i, (Xs, ys) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(Xs)
                loss = loss_fn(outputs, ys)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            # print(np.mean(losses))
        return model


class MLPRegressorTransform(Transform):
    def __init__(self):
        """
        the different between MLPReg and MLPCla is the loss function and y value
        """
        super(MLPRegressorTransform, self).__init__()
        self.name = 'mlpRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        train_dataset = Data_rb_reg(X_train, y_train)
        train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
        model = MLP(X_train.shape[1], hidden_size=32, output_size=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = nn.L1Loss()
        model = self._train_model(model, epochs=100, train_loader=train_loader, optimizer=optimizer, loss_fn=loss_fn)
        X_train_pred = pd.DataFrame(model.layers(torch.from_numpy(X_train.to_numpy()).float()).data.numpy(),
                                    index=X_train.index,
                                    columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        X_test_pred = pd.DataFrame(model.layers(torch.from_numpy(X_test.to_numpy()).float()).data.numpy(),
                                   index=X_test.index,
                                   columns=['mlp_pred_' + str(i) + '_' + str(X_train.shape[1]) for i in range(32)])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        return X_train, X_test

    def _train_model(self, model, epochs, train_loader, optimizer, loss_fn):
        for epoch in range(epochs):
            model.train()
            losses = []
            for i, (Xs, ys) in enumerate(train_loader):
                optimizer.zero_grad()
                outputs = model(Xs)
                loss = loss_fn(outputs, ys)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            print(np.mean(losses))
        return model


class MLP(torch.nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=1):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        self.ln = nn.Linear(hidden_size, output_size)

    def forward(self, inp):
        '''
        inp shape of torch tensor
        '''
        out = self.ln(self.layers(inp))
        return out


class Data_rb_cla(Dataset):
    def __init__(self, Xs, ys):
        # input is type of pandas
        self.Xs = torch.from_numpy(Xs.to_numpy()).float()
        self.ys = torch.from_numpy(ys.to_numpy()).long()

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]


class Data_rb_reg(Dataset):
    def __init__(self, Xs, ys):
        # input is type of pandas
        self.Xs = torch.from_numpy(Xs.to_numpy()).float()
        self.ys = torch.from_numpy(ys.to_numpy()).float()

    def __len__(self):
        return len(self.Xs)

    def __getitem__(self, idx):
        return self.Xs[idx], self.ys[idx]


class NearestNeighborsClassifierTransform(Transform):
    def __init__(self):
        super(NearestNeighborsClassifierTransform, self).__init__()
        self.name = 'nearestNeighborsClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        neigh = NearestNeighbors()
        neigh.fit(X_train)
        neigh.kneighbors(n_neighbors=30)
        neigh.fit(X_train)
        indx = pd.DataFrame(neigh.kneighbors(X_train)[1], index=X_train.index)
        nnx = indx.apply(lambda x: pd.Series([(y_train.iloc[x] == i).sum() for i in range(len(y_train.unique()))]),
                         axis=1)
        nnx.columns = ['nn_' + str(i) for i in nnx.columns]
        indy = pd.DataFrame(neigh.kneighbors(X_test)[1], index=X_test.index)
        nny = indy.apply(lambda x: pd.Series([(y_train.iloc[x] == i).sum() for i in range(len(y_train.unique()))]),
                         axis=1)
        nny.columns = ['nn_' + str(i) for i in nny.columns]
        X_train = pd.concat([X_train, nnx], axis=1)
        X_test = pd.concat([X_test, nny], axis=1)
        return X_train, X_test


class NearestNeighborsRegressorTransform(Transform):
    def __init__(self):
        super(NearestNeighborsRegressorTransform, self).__init__()
        self.name = 'nearestNeighborsRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        neigh = NearestNeighbors()
        neigh.fit(X_train)
        neigh.kneighbors(n_neighbors=5)
        neigh.fit(X_train)
        indx = pd.DataFrame(neigh.kneighbors(X_train)[1], index=X_train.index)
        X_train['nn_target'] = indx.apply(lambda x: y_train.iloc[x].mean(), axis=1)
        indy = pd.DataFrame(neigh.kneighbors(X_test)[1], index=X_test.index)
        X_test['nn_target'] = indy.apply(lambda x: y_train.iloc[x].mean(), axis=1)
        return X_train, X_test


class SVRTransform(Transform):
    def __init__(self):
        super(SVRTransform, self).__init__()
        self.name = 'svrTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = SVR().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svr_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svr_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = SVR().fit(X_train, y_train - X_train_pred['svr_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svr_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svr_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class SVCTransform(Transform):
    def __init__(self):
        super(SVCTransform, self).__init__()
        self.name = 'svcTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = SVC().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svc_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svc_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['svc_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = SVC().fit(X_train, y_train == X_train_pred['svc_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['svc_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['svc_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauDotWhiteRegressorTransform(Transform):
    def __init__(self):
        super(GauDotWhiteRegressorTransform, self).__init__()
        self.name = 'gauDotWhiteRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaudotwhite_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_reg_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train - X_train_pred['gaudotwhite_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['gaudotwhite_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_reg_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauDotClassifierTransform(Transform):
    def __init__(self):
        super(GauDotClassifierTransform, self).__init__()
        self.name = 'gauDotClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = DotProduct() + WhiteKernel()
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaudotwhite_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_cla_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['gaudotwhite_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train == X_train_pred['gaudotwhite_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['gaudotwhite_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaudotwhite_cla_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauExpRegressorTransform(Transform):
    def __init__(self):
        super(GauExpRegressorTransform, self).__init__()
        self.name = 'gauExpRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = Exponentiation(RationalQuadratic(), exponent=2)
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_reg_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train - X_train_pred['gauexp_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_reg_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauExpClassifierTransform(Transform):
    def __init__(self):
        super(GauExpClassifierTransform, self).__init__()
        self.name = 'gauExpClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = Exponentiation(RationalQuadratic(), exponent=2)
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_cla_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['gauexp_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train == X_train_pred['gauexp_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gauexp_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gauexp_cla_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauRBFRegressorTransform(Transform):
    def __init__(self):
        super(GauRBFRegressorTransform, self).__init__()
        self.name = 'gauRBFRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = 1.0 * RBF(1.0)
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_reg_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_reg_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessRegressor(kernel=kernel).fit(X_train, y_train - X_train_pred['gaurbf_reg_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_reg_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_reg_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class GauRBFClassifierTransform(Transform):
    def __init__(self):
        super(GauRBFClassifierTransform, self).__init__()
        self.name = 'gauRBFClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        kernel = 1.0 * RBF(1.0)
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_cla_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_cla_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['gaurbf_cla_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = GaussianProcessClassifier(kernel=kernel).fit(X_train, y_train == X_train_pred['gaurbf_cla_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['gaurbf_cla_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['gaurbf_cla_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class RandomForestClassifierTransform(Transform):
    def __init__(self):
        super(RandomForestClassifierTransform, self).__init__()
        self.name = 'randomForestClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = RandomForestClassifier().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfc_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfc_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['rfc_pred']).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = RandomForestClassifier().fit(X_train, y_train == X_train_pred['rfc_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfc_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfc_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class RandomForestRegressorTransform(Transform):
    def __init__(self):
        super(RandomForestRegressorTransform, self).__init__()
        self.name = 'randomForestRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = RandomForestRegressor().fit(X_train, y_train)
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfr_pred'])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfr_pred'])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = RandomForestRegressor().fit(X_train, y_train - X_train_pred['rfr_pred'])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['rfr_pred_diff'])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['rfr_pred_diff'])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class XGBClassifierTransform(Transform):
    def __init__(self):
        super(XGBClassifierTransform, self).__init__()
        self.name = 'xgbClassifierTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = xgb.XGBClassifier().fit(X_train, y_train)
        ppname = str(time.time())[-3:]
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['xgb_cla_pred' + ppname])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_cla_pred' + ppname])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        if len((y_train == X_train_pred['xgb_cla_pred' + ppname]).unique()) == 1:
            return X_train, X_test
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = xgb.XGBClassifier().fit(X_train, y_train == X_train_pred['xgb_cla_pred' + ppname])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['xgb_cla_pred_diff' + ppname])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_cla_pred_diff' + ppname])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test


class XGBRegressorTransform(Transform):
    def __init__(self):
        super(XGBRegressorTransform, self).__init__()
        self.name = 'xgbRegressorTransform'
        self.type = 5

    def fit(self, X_train, X_test, y_train, y_test):
        # train linear model and predict
        reg = xgb.XGBRegressor().fit(X_train, y_train)
        ppname = str(time.time())[-3:]
        X_train_pred = pd.DataFrame(reg.predict(X_train), index=X_train.index, columns=['xgb_reg_pred' + ppname])
        X_test_pred = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_reg_pred' + ppname])
        X_train = pd.concat([X_train, X_train_pred], axis=1)
        X_test = pd.concat([X_test, X_test_pred], axis=1)
        # add diff prediction
        X_train = X_train.astype(float)
        X_test = X_test.astype(float)
        # deal with inf.
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        if X_train.isna().sum().sum() > 0:
            X_train = X_train.fillna(method='ffill')
        reg = xgb.XGBRegressor().fit(X_train, y_train - X_train_pred['xgb_reg_pred' + ppname])
        X_train_pred_diff = pd.DataFrame(reg.predict(X_train), index=X_train.index,
                                         columns=['xgb_reg_pred_diff' + ppname])
        X_test_pred_diff = pd.DataFrame(reg.predict(X_test), index=X_test.index, columns=['xgb_reg_pred_diff' + ppname])
        X_train = pd.concat([X_train, X_train_pred_diff], axis=1)
        X_test = pd.concat([X_test, X_test_pred_diff], axis=1)
        return X_train, X_test
    
    
unary_operators = {'abs':Abs, 'adde':Adde, 'cos':Cos, 'degree':Degree, 'exp':Exp, 'ln':Ln, 'negative':Negative, 
                   'radian':Radian, 'reciprocal':Reciprocal, 'sin':Sin, 'sigmoid':Sigmoid, 'square':Square, """ 'sqrt':Sqrt, """ 
                   'tanh':Tanh, 'relu':Relu, 'mmnorm':Minmaxnorm, 'zscore':Zscore, 'kernelapproxrbf':KernelApproxRBF}

binary_operators = {'div':Div, 'minus':Minus, 'add':Add, 'product':Product}

ctype_operators = {'timeagg':Timeagg, 'p25':Percentile25, 'p50':Percentile50, 'p75':Percentile75, 
                    'max':Maximum, 'min':Minimum, 'autoencoder':Autoencoder}

dtype_operators = {'clustering':Clustering}

etype_operators = {}



