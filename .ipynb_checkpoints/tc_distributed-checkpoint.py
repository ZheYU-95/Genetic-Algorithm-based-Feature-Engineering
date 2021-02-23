import numpy as np
import pandas as pd
import traceback
import copy
import logging
import random
import sys
import math
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.inspection import permutation_importance
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
from dask import delayed
from dask.distributed import LocalCluster, Client
import transforms_distributed as tfd


def clean_dat(dat: pd.DataFrame, logger=None) -> pd.DataFrame:
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    logger.debug('Clean Data, number of inf and nun are for dataset: (%d, %d)' % (
            (dat == np.inf).sum().sum(), dat.isna().sum().sum()))
    logger.info(' -Set type to float32 at first && deal with inf.')
    dat = dat.astype(np.float32)
    dat = dat.replace([np.inf, -np.inf], np.nan)
    logger.info(' -Remove columns with half of nan')
    dat = dat.dropna(axis=1, thresh=dat.shape[0] * .5)
    logger.info(' -Remove costant columns')
    dat = dat.loc[:, (dat != dat.iloc[0]).any()]
    logger.info(' -Remove columns with too many so small numbers')
    for col in dat.columns:
        if (abs(dat[col] - 0.0) < 0.0001).sum() / dat.shape[0] > 0.8:
            print((abs(dat[col] - 0.0) < 0.0001).sum())
            dat.drop(col, axis=1, inplace=True)
    if dat.isna().sum().sum() > 0:
        logger.info(' -Start to fill the columns with nan')
        # imp = IterativeImputer(max_iter=10, random_state=0)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # dat = dat.fillna(dat.mean())
        tmp = imp.fit_transform(dat)
        if tmp.shape[1] != dat.shape[1]:
            tmp = dat.fillna(0)
        dat = pd.DataFrame(tmp, columns=dat.coulumns, index=dat.index)
    logger.info(' -Remove rows with any nan in the end')
    dat = dat.dropna(axis=0, how='any')
    logger.debug('End with Data cleaning, number of inf and nun are for dataset: (%d, %d)' 
                 % ((dat == np.inf).sum().sum(), dat.isna().sum().sum()))
    return dat


def subsampling(dat: pd.DataFrame):
    """ when number of instance too large, only use 10000 data to do the feature engineering """
    if dat.shape[0] > 10000:
        return dat.sample(n=10000, random_state=1).reset_index(drop=True)
    else:
        return dat


def data_preprocessing(dat: pd.DataFrame, art='C', y=None, logger=None, remove=True):
    """
    Encoding + remove columns with more than 1/2 na if remove==True + remove columns with all na + imputation
    if art == 'C', will do LabelEncoding first for the target column
    ================
    Parameter:
    ================
    dat - type of DataFrame
    art - type of string
        either C for classifcation of R for regression. indicates the type of problem 
    y - type of string
        the name of the target column; if None, set the last column of the data set as target
        considering only one column for label
    logger - type of Logger
    remove - type of bollean
        whether remove the columns with na value more than half length or not
    =================
    Output
    =================
    dat - type of Dataframe 
        the dataframe after preprocessing
    cols - type of list of string
        the name of the numerical columns
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    logger.info('- Start preprocessing function')
    # replace original indeices with default ones
    dat = dat.reset_index(drop=True)

    if art == 'C':
        logger.info('- start to label target feature y for classification task')
        dat.iloc[:, -1] = LabelEncoder().fit_transform(dat.iloc[:, -1])
        logger.info('+ end with label encoding the target feature')
    if remove:
        # remove columns with more than 1/2 na
        dat = dat.loc[:, dat.isna().sum()/len(dat) < .5]
        logger.info('Following features are removed from the dataframe because half of their value are NA: %s' %
                    (dat.columns[dat.isna().sum()/len(dat) > .5].to_list()))
    # Encoding
    oe = OneHotEncoder(drop='first')
    # get categorical columns
    if y:
        dat_y = dat[[y]]
        cols = dat.columns.to_list()
        cols.remove(y)
        dat_x = dat[cols]
    else:
        dat_y = dat[[dat.columns[-1]]]
        dat_x = dat[dat.columns[:-1]]
    dat_categ = dat_x.select_dtypes(include=['object'])
    # get kterm of categ features
    for i in dat_categ.columns:
        # save output to dat
        tmp = dat_x[i].value_counts()
        dat_x[i + '_kterm'] = dat_x[i].map(lambda x: tmp[x] if x in tmp.index else 0)
    # float columns including the k term cols
    dat_numeric = dat_x.select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    # onehot encoding and label encoding
    dat_categ_onehot = dat_categ.iloc[:, dat_categ.apply(
        lambda x: len(x.unique())).values < 8]
    dat_categ_label = dat_categ.iloc[:, dat_categ.apply(
        lambda x: len(x.unique())).values >= 8]
    flag_onehot = False
    flag_label = False
    # oe
    if dat_categ_onehot.shape[1] > 0:
        logger.info('- start to do onehot to the following categoric features: %s' %
                    (str(dat_categ_onehot.columns.to_list())))
        dat_onehot = pd.DataFrame(oe.fit_transform(dat_categ_onehot.astype(str)).toarray(),
                                  columns=oe.get_feature_names(dat_categ_onehot.columns))
        logger.info('+ end with onehot')
        flag_onehot = True
    else:
        dat_onehot = None
    # le
    if dat_categ_label.shape[1] > 0:
        logger.info('- start to do label encoding to the following categoric features: %s' %
                    (str(dat_categ_label.columns.to_list())))
        dat_categ_label = dat_categ_label.fillna('NULL')
        dat_label = pd.DataFrame(columns=dat_categ_label.columns)
        for i in dat_categ_label.columns:
            dat_label[i] = LabelEncoder().fit_transform(
                dat_categ_label[i].astype(str))
        logger.info('+ end with label encoding')
        flag_label = True
    else:
        dat_label = None
    # scaling
    # combine
    dat_new = pd.DataFrame()
    if flag_onehot and flag_label:
        dat_new = pd.concat([dat_numeric, dat_onehot, dat_label], axis=1)
    elif flag_onehot:
        dat_new = pd.concat([dat_numeric, dat_onehot], axis=1)
    elif flag_label:
        dat_new = pd.concat([dat_numeric, dat_label], axis=1)
    else:
        dat_new = dat_numeric
    dat_new = pd.concat([dat_new, dat_y], axis=1)
    # imputation
    dat_new = dat_new.dropna(axis=1, how='all')
    if dat_new.isna().sum().sum() > 0:
        logger.info('- na value exist, start to fill na with iterative imputer: ' +
                    str(dat_new.isna().sum().sum()))
        # include na value, impute with iterative Imputer or simple imputer
        columns = dat_new.columns
        imp = IterativeImputer(max_iter=10, random_state=0)
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dat_new = imp.fit_transform(dat_new)
        dat_new = pd.DataFrame(dat_new, columns=columns)
    dat_numeric = dat_new.iloc[:, :-1].select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    logger.info('+ end with fill na')
    return dat_new, dat_numeric.columns


def balanced_sampling(dat: pd.DataFrame, logger=None):
    """ balanced sample data from each class to avoid huge data size bewtween different classes """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    # upsampling
    logger.info('Balanced sampling')
    subsample = []
    num_of_each_class = dat.iloc[:, -1].value_counts().to_numpy()
    if num_of_each_class.std()*1.0 / num_of_each_class.mean() < 0.1:
        logger.info('- The given data is balance.')
        # the dataset is balanced
        return dat
    logger.info('- Given dataset is unbalance')
    logger.info('- Sampling data from each class to generate a new dataset')
    n_smp = num_of_each_class.max()
    for label in dat.iloc[:, -1].value_counts().index:
        samples = dat[dat.iloc[:, -1] == label]
        num_samples = len(samples)
        index_range = range(num_samples)
        # take all from the set
        indexes = list(np.random.choice(index_range, size=num_samples, replace=False))
        indexes2 = list(np.random.choice(
            index_range, size=n_smp-num_samples, replace=True))  # add random items
        indexes.extend(indexes2)
        subsample.append(samples.iloc[indexes, :])
    logger.info('+ end with sampling')
    out = pd.concat(subsample)
    out = out.sample(frac=1).reset_index(drop=True)  # shuffle and re index
    return out


def load_data(dataset_name, sep=',', header='infer', index_col=None, logger=None, art='C'):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    logger.info(' -Load data from file: %s' % (dataset_name))
    dat = pd.read_csv(dataset_name, sep=sep, header=header, index_col=index_col)
    dat = dat.reset_index(drop=True)
    dat = clean_dat(dat, logger=logger)
    logger.info(' -Check data size')
    dat.columns = [str(i) for i in dat.columns]
    logger.info(' -Check data balance')
    if art == 'C':
        # up sampling, if art of task is classification
        dat = balanced_sampling(dat)
        # sub sampling, if number of data point >= 10000
        dat = subsampling(dat)
    else:
        # sub sampling, if number of data point >= 10000
        dat = subsampling(dat)
    logger.info(' -start to do data preprocessing')
    print(dat)
    dat, numeric_cols = data_preprocessing(dat, art=art, logger=logger)
    logger.info(' +End with data preprocessing')
    # print(dat.head())
    logger.debug(' -successfully load data!')
    
    N = 1
    # dat = dd.from_pandas(dat, npartitions=N, chunksize=None)
    return dat


@delayed
def eval(X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: np.array, y_test: np.array, art='C', logger=None):
    """
    output:
    fitness: type of float, performance of the model;
    fe: type of tuple, contains: name of the most important feature and data
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')

    fitness = 0.0
    if X_train.isna().sum().sum() > 0:
        print('NA value exist')
        print(X_train)
        X_train.to_csv('data/pro.csv')
    model = None
    try:
        logger.info('Get final fitness')
        # feature selection miss
        #print('finish fillna and feature transformation', str(time.time()))
        if art == 'C':
            logger.info('- start to run classification model')
            #model = SVC()
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            logger.info('+ end with model running and return evaluate result')
            # evaluate the fitness of model based on current selected training features
            # fitness = roc_auc_score(y_test, predict) # two class
            # fitness = accuracy_score(y_test, predict) # or f1_score
            if len(set(predict)) > 2 or len(y_test) > 2:
                fitness = f1_score(y_test, predict, average='weighted')
            else:
                fitness = f1_score(y_test, predict)
        else:
            logger.info('- start to run regression model')
            model = RandomForestRegressor()
            model.fit(X_train, y_train)
            predict = model.predict(X_test)
            logger.info('- end with model running and return evaluate result')
            # evaluate
            fitness = 1 - mean_absolute_error(y_test, predict)
    except Exception as e:
        ex_type, ex_value, ex_traceback = sys.exc_info()
        # Extract unformatter stack traces as tuples
        trace_back = traceback.extract_tb(ex_traceback)
        # Format stacktrace
        stack_trace = list()
        for trace in trace_back:
            stack_trace.append("File : %s , Line : %d, Func.Name : %s, Message : %s" % (
                trace[0], trace[1], trace[2], trace[3]))
        print('Error appear:')
        # print(State.state)
        logger.error(e, exc_info=True)
        print("Stack trace : %s" % stack_trace)

    # get the most important k features
    res = permutation_importance(
        model, X_train, y_train, n_repeats=5, random_state=0)
    fe_importances = res.importances_mean
    """ select the most important k features """
    k = math.ceil(X_train.shape[1] * .3)
    ind_fe = np.argpartition(fe_importances, -k)[-k:]
    name_col = X_train.columns[ind_fe]
    # ind_fe = model.feature_importances_
    # name_col = X_train.columns[np.argmax(idx_fe)]
    logger.debug('successfully evaluating fitness')
    return fitness, (name_col, X_train[name_col])


def partition(lis: list, n: int):
    """ randomly split a list into n groups """
    # prevent destroying the original dataset
    lis_cp = copy.deepcopy(lis)
    random.shuffle(lis_cp)
    if len(lis) > n:
        return [lis_cp[i::n] for i in range(n)]
    else:
        return [[lis_cp[i]] for i in range(len(lis))]


def tribesCompetition(candidate_dat: pd.DataFrame, art='C', logger=None) -> list:
    """ 
    candidate_dat: cleaned data after preprocessing, subsampling and balanced-sampling 
                   and generating all possible individuals

    labels placed at the last column

    ##output## -> list of groups, each group includes columns of features names and its score, which at the last column
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')

    cols = candidate_dat.columns[:-1].tolist()
    """ gp_num: """
    gp_num = 2
    # logger.info('Start randomly splitting feature columns into %s groups with almost the same size' % (str(gp_num)))
    gps = partition(cols, gp_num)
    logger.info(
        'Start evaluating the groups fitnesses 1-by-1 and finding out the best & worst groups')
    # dic reserve gp:mean_score pairs
    gps_list = list()
    for gp in gps:
        gp_dat = pd.DataFrame(candidate_dat.loc[:, gp], columns=gp)
        # evaluate the groups' fitness/performances and find out the best & worst one
        scores = []
        """ repeat_times: """
        repeat_times = 3
        for i in range(repeat_times):
            X_train, X_test, y_train, y_test = train_test_split(gp_dat, candidate_dat.iloc[:, -1])
            s = eval(X_train, X_test, y_train, y_test, art=art, logger=logger)[0]
            # score += s
            scores.append(s)
        mean_score = delayed(sum)(scores) / repeat_times
        # logger.info('The mean performance of the given features combination %s is:\t %s'%(str(gp), str(mean_score)))
        # the last column in gp is its score, which is used for sorting
        gp.append(mean_score)
        gps_list.append(gp)
    sorted_gps_list = delayed(sorted)(gps_list, key=lambda x: x[-1], reverse=False)
    gp_min = sorted_gps_list[0]
    gp_max = sorted_gps_list[-1]
    logger.debug('The groups with the worst %s and best %s performances are found.\t ' % (str(gp_min), str(gp_max)))
    return sorted_gps_list
    # return gps_list


def plunge_eliminate(dat: pd.DataFrame, sorted_gps: list, art='C', logger=None) -> pd.DataFrame():
    """
    ###

    function: subgroup the weakest and best seperately and 
    try to replace the worst subgroup in the best with the best subgroup in the weakest

    ###

    dat: generated children features derived from the same operator, combined with label column, waiting for plunge and die out

    sorted_gps: each element in list is a list including features group combined with its score

    return: selected features without label column any more
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')

    logger.debug('start to plunge and dying out groups')
    # excluding the label column in list element firstly
    worst = sorted_gps[0][:-1]
    best = sorted_gps[-1][:-1]

    gp_w = pd.DataFrame(dat[worst])
    gp_w[dat.columns[-1]] = dat.iloc[:, -1]
    logger.info('start to get subgroups performances ranking from the worst group')
    sorted_gp_w = tribesCompetition(gp_w, art=art, logger=logger)
    # worst_in_gp_w = pd.DataFrame(gp_w.loc[:, sorted_gp_w[0][:-1]], columns=sorted_gp_w[0][:-1])
    best_in_gp_w = pd.DataFrame(
        gp_w.loc[:, sorted_gp_w[-1][:-1]], columns=sorted_gp_w[-1][:-1])

    gp_b = pd.DataFrame(dat[best])
    gp_b[dat.columns[-1]] = dat.iloc[:, -1]
    logger.info('start to get subgroups performances ranking from the best group')
    sorted_gp_b = tribesCompetition(gp_b, art=art, logger=logger)
    # worst_in_gp_b = pd.DataFrame(gp_b.loc[:, sorted_gp_b[0][:-1]], columns=sorted_gp_b[0][:-1])
    # best_in_gp_b = pd.DataFrame(gp_b.loc[:, sorted_gp_b[-1][:-1]], columns=sorted_gp_b[-1][:-1])
    no_worst_in_gp_b = pd.DataFrame(
        gp_b.loc[:, sorted_gp_b[1:][:-1]], columns=sorted_gp_b[1:][:-1])

    new_gp_b = pd.concat([best_in_gp_w, no_worst_in_gp_b], axis=1)
    score = 0.0
    """ repeat_times: """
    repeat_times = 3
    for i in range(repeat_times):
        X_train, X_test, y_train, y_test = train_test_split(new_gp_b, gp_b.iloc[:, -1])
        score += eval(X_train, X_test, y_train, y_test, art=art, logger=logger)[0]
    mean_score = score / repeat_times
    init_score = sorted_gp_b[-1][-1]
    logger.info('finish plunge and eliminating')
    if (mean_score > init_score):
        return new_gp_b
    else:
        return gp_b.iloc[:, :-1]


def feature_selection(dat: pd.DataFrame, art='C', logger=None) -> pd.DataFrame():
    """ 
    dat: features + label column
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    logger.info(
        'the number of columns exceed max number(%d), try columns selection first' % (2000))

    # feature selection
    if art == 'C':
        model = RandomForestClassifier()
    else:
        model = RandomForestRegressor()
    clf = model.fit(dat.iloc[:, :-1], dat.iloc[:, -1])
    fs = SelectFromModel(clf, threshold='mean', prefit=True)
    supp = fs.get_support()
    df = pd.DataFrame(fs.transform(
        dat.iloc[:, :-1]), columns=dat.iloc[:, :-1].columns[supp], index=dat.index)
    df['target'] = dat.iloc[:, -1]
    logger.debug('End with columns selection, number of columns now is: %s' % (
        str(dat.iloc[:, :-1].shape[1])))
    return df


def generateCandidates(cur_dat: pd.DataFrame, prev_gen: pd.DataFrame = None, art='C', logger=None) -> pd.DataFrame():
    """ 
    input -> cur_dat: newly generated candidates features and label column; prev_gen: father features;
    output -> generated candidates after competition and selection
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    logger.info('start to generate children candidates')
    una_oprs = tfd.unary_operators
    bina_oprs = tfd.binary_operators
    oprs_list = [una_oprs, bina_oprs]
    cur_gen = pd.DataFrame(cur_dat.iloc[:, :-1], columns=cur_dat.columns[:-1])
    res = pd.DataFrame()
    for oprs in oprs_list:
        for opr in oprs.values():
            candidates = pd.DataFrame()
            if oprs == oprs_list[0]:
                candidates = opr()._exec(cur_gen)
            elif oprs == oprs_list[1]:
                candidates = opr()._exec(cur_gen, prev_gen)
            
            if candidates.empty:
                logger.debug(
                'no possible children candidates for opr %s , skip following steps' % (str(opr)))
                continue
            
            cand_dat = copy.deepcopy(candidates)
            cand_dat[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
            logger.debug(
                'all possible children candidates for opr %s have been generated and added label column' % (str(opr)))
            
            sorted_gps = tribesCompetition(cand_dat, art=art, logger=logger)
            candidates = plunge_eliminate(cand_dat, sorted_gps, art=art, logger=logger)
            
            res = pd.concat([res, candidates], axis=1)
            
            logger.debug(
                'tribes competition finished and the selected concatenated to the dataset, try to limit size of dataset if needed')
            """ res_limit:  """
            res_limit = 200
            res[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
            while (res.shape[1] > res_limit):
                res = feature_selection(res, art=art, logger=logger)
            res.drop(res.columns[-1], axis=1, inplace=True)
    logger.debug(' remaining children candidates are ready')
    logger.warning('This round generates %s children features' %(str(len(res.columns))))
    return res







def update_dat(dat: pd.DataFrame, prev_gen: pd.DataFrame=None, art='C', logger=None):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    
    nxt_g = generateCandidates(dat, prev_gen, art=art, logger=logger)
    nxt_g = clean_dat(nxt_g, logger)
    
    nxt_dat = copy.deepcopy(nxt_g)
    nxt_dat[dat.columns[-1]] = dat.iloc[:, -1]
    
    """ prev_limit: an adaptive hyperparameter """
    prev_limit = 200
    cur_gen = dat.drop(dat.columns[-1], axis=1, inplace=False)
    prev_gen = pd.concat([prev_gen, cur_gen], axis=1)
    prev_gen[dat.columns[-1]] = dat.iloc[:, -1]
    while prev_gen.shape[1] > prev_limit:
        prev_gen = feature_selection(prev_gen, art=art, logger=logger)
    prev_gen.drop(prev_gen.columns[-1], axis=1, inplace=True)
    
    logger.warning('There are %s remaining features in prev_generation' %(str(len(prev_gen.columns))))
    return nxt_dat, prev_gen


