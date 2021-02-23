import copy
import numpy as np
import pandas as pd
import logging
import pickle
import random
import time

from sklearn.model_selection import train_test_split

from transforms import *


class Pipeline():
    def __init__(self, data, art='C', logger=None, seed=0):
        """
        self attribute is not safe here because of Parallel
        ==========
        inputs:
        ==========
        data, type of dataframe, target dataset
        numeric_cols, type of list of string
            numeric columns in the dataset
        actions, type of list of string
            transformation sequence
        logger, type of Logger
        seed, type of int
            random seed used in the train test split
        """
        if logger:
            self.logger = logger
        else:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            self.logger = logging.getLogger(__name__)
        self.dat = data  # save the original data that used in the convertion later
        self.seed = seed
        self.art = art         # ??
        self.max_db = 1000
        self.database = {}  # database used to save history converted data
        # keys is ','.join(actions) value is type of dict


    def clean_data(self, train: pd.DataFrame, test: pd.DataFrame=None):
        self.logger.info('Clean Data, number of inf and nun are for train set: (%d, %d)' % (
            (train == np.inf).sum().sum(), train.isna().sum().sum()))
        """ 补充一个删掉有很多极小值的 """
        if type(test) != type(None):
            self.logger.info('Clean Data, number of inf and nun are: (%d, %d) for test set' % (
                (test == np.inf).sum().sum(), test.isna().sum().sum()))
        # set type to float32 at first && deal with inf.
        train = train.astype(np.float32)
        train = train.replace([np.inf, -np.inf], np.nan)
        if type(test) != type(None):
            test = test.astype(np.float32)
            test = test.replace([np.inf, -np.inf], np.nan)
        # remove columns half of na
        train = train.dropna(axis=1, thresh=len(train) * .5)
        # remove costant columns
        train = train.loc[:, (train != train.iloc[0]).any()]
        # train = train.loc[:, (train.isna().sum()/len(train) < .3)&(test.isna().sum()/len(test) < .3)]
        # fillna
        if train.isna().sum().sum() > 0:
            self.logger.info('- start to fill na for the new feature')
            columns_index = train.columns
            # imp = IterativeImputer(max_iter=10, random_state=0)
            imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            # train = train.fillna(train.mean())
            index_train = train.index
            tmp = imp.fit_transform(train)
            if tmp.shape[1] != train.shape[1]:
                tmp = train.fillna(0)
            train = pd.DataFrame(tmp, columns=columns_index, index=index_train)
        if type(test) != type(None):
            test = test.loc[:, train.columns]
            if test.isna().sum().sum() > 0:
                imp = SimpleImputer(missing_values=np.nan, strategy='mean')
                columns_test = test.columns
                index_test = test.index
                tmp = imp.fit_transform(test)
                if tmp.shape[1] != test.shape[1]:
                    tmp = test.fillna(0)
                test = pd.DataFrame(
                    tmp, columns=columns_test, index=index_test)
                # test = test.fillna(test.mean())
        self.logger.info('End with Data cleaning, number of inf and nun are for train set: (%d, %d)' % (
            (train == np.inf).sum().sum(), train.isna().sum().sum()))
        if type(test) != type(None):
            self.logger.info('End with Data cleaning, number of inf and nun are: (%d, %d) for test set' % (
                (test == np.inf).sum().sum(), test.isna().sum().sum()))
            return train, test
        return train


    def _feature_selection(self, train, test, y_train):
        self.logger.info('The number of columns exceed max number, try columns selection first')
        # feature selection, recognize which kind of task Classification or Regression 
        if self.art == 'C':
            clf = RandomForestClassifier().fit(train, y_train)
        else:
            clf = RandomForestRegressor().fit(train, y_train)
        fs = SelectFromModel(clf, threshold= 'mean', prefit=True)
        supp = fs.get_support()    # Get a mask, or integer index, of the features selected
        # fs.transform(X) : Reduce X to the selected features
        train = pd.DataFrame(fs.transform(train), columns = train.loc[:, supp].columns, index = train.index)
        test = pd.DataFrame(fs.transform(test), columns = test.loc[:, supp].columns, index = test.index)
        self.logger.info('End with columns selection, number of columns now is: %s' %(str(train.shape[1])))
        return train, test

    
    def reset_seed(self, seed):
        # this process will clear the buffer
        self.seed = seed
        self.database = {}


    def _save_data_to_database(self, actions, train, test, y_train, y_test):
        # check condition, maximal number or left memory
        tmpkey = ', '.join(actions)
        if tmpkey in self.database.keys():
            self.logger.info('- data set already exist')
            #print('actions already exist %s'%str(actions))
            return 0
        self.logger.info('Save data set to database: %s'%(actions))
        dict_data = {'hit_count': 1, 
                     'X_train': copy.deepcopy(train), 'X_test': copy.deepcopy(test), 
                     'y_train': copy.deepcopy(y_train), 'y_test': copy.deepcopy(y_test)}
        if len(self.database) <= self.max_db:
            self.database[tmpkey] = dict_data
        else:
            self.logger.info('Database is full, remove the one with smallest count')
            tg = sorted(self.database.items(), key = lambda x: x[1]['hit_count'])[0][0]
            del self.database[tg]
            self.database[tmpkey] = dict_data
        return 1


    def _convert_action_to_transform(self, action):
        transfo = unary_operators.get(action)
        if transfo == None:
            transfo = binary_operators.get(action)
            if transfo == None:
                transfo = ctype_operators.get(action)
        if transfo == None:
            self.logger.error('Target transform %s not found in the method set, jump over the method'%action)
        else:
            self.logger.info('- load method %s'%action)
        return transfo

    
    def insert_action(self, action):
        self.transforms.append(self._convert_action_to_transform(action))
        self.dat = self.dat.dropna(axis = 1, how = 'all')
        if self.dat.isna().sum().sum() > 0:
            if self.logger:
                self.logger.info('- na value exist, start to fill na with iterative imputer: '+ str(self.dat.isna().sum().sum()))
            # include na value, impute with iterative Imputer or simple imputer
            columns = self.dat.columns
            imp = IterativeImputer(max_iter=10, random_state=0)
            # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
            dat = imp.fit_transform(self.dat)
            dat = pd.DataFrame(dat, columns = columns)

    
    def _load_actions(self, actions):
        """
        check if transformation existes, if so, load the data set
        convert actions name to real transformation functions
        ==========
        outputs:
        ==========
            tfs, list of transform left
            dat, base dataset
        """
        self.logger.info('Load actions: %s'%(actions))
        # check if transformation is already exists
        actions_left = actions
        dat = self.dat
        X_train, X_test, y_train, y_test = train_test_split(dat.iloc[:, :-1], dat.iloc[:, -1], 
                                                            test_size = 0.33, random_state = self.seed)
        for i in np.arange(len(actions), -1, -1):
            ## check database existence
            tmp = actions[:i]
            tmp = ', '.join(tmp) # should be same as saving the dataset
            if tmp in self.database.keys():
                self.logger.info('- part of sequence existed, direct use the dataset in the database')
                self.logger.info('- transformation of existed data set: %s'%(tmp))
                self.logger.info('- transformation still lack of: %s'%(actions[i:]))
                actions_left = actions[i:] # reset actions list
                self.database[tmp]['hit_count'] += 1 # add hit count
                X_train = copy.deepcopy(self.database[tmp]['X_train'])
                X_test  = copy.deepcopy(self.database[tmp]['X_test'])
                y_train = copy.deepcopy(self.database[tmp]['y_train'])
                y_test  = copy.deepcopy(self.database[tmp]['y_test'])
                break
        tfs = [self._convert_action_to_transform(i) for i in actions_left]
        return tfs, X_train, X_test, y_train, y_test


    def _set_transform(self, actions):
            self.logger.info('Set transforms')
            return self._load_actions(actions)


    
    def run(self, actions):
        init_perf = self.game.init_performance
        top_five = []
        for i in range(2):
            counter = 1
            for i in range(25):
                self.logger.info('X'*70)
                self.logger.info('Round:\t%d'%(counter))
                self.logger.info('X'*70)
                self.episode()
                counter+=1
                top_five.append(self.game.record_top_five)
            self._update_init_performance()
        return init_perf, top_five, self.game.record_best_features, self.game.buffer, self.root.edges