from operator import le
import numpy as np
import pandas as pd
import traceback
import copy
import logging
import random
import sys
import math
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression, f_classif, f_regression
# from sklearn.inspection import permutation_importance
import dask.array as da
import dask.bag as db
import dask.dataframe as dd
from dask import delayed, compute
from dask.distributed import LocalCluster, Client
import transforms_distributed as tfd
# from boruta import BorutaPy

una_oprs = tfd.unary_operators
bina_oprs = tfd.binary_operators
oprs_names = tfd.operators_names


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
    remove - type of boolean
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
    
    logger.info('Start data preprocessing')
    # replace original indeices with default ones
    dat = dat.reset_index(drop=True)

    if art == 'C':
        logger.info('Start to label target feature y for classification task')
        dat.iloc[:, -1] = LabelEncoder().fit_transform(dat.iloc[:, -1])
        logger.info('End with label encoding the target feature')
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
    dat_categ_onehot = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values < 8]
    dat_categ_label = dat_categ.iloc[:, dat_categ.apply(lambda x: len(x.unique())).values >= 8]
    flag_onehot = False
    flag_label = False
    # oe
    if dat_categ_onehot.shape[1] > 0:
        logger.info('Start to do onehot to the following categoric features: %s' %
                    (str(dat_categ_onehot.columns.to_list())))
        dat_onehot = pd.DataFrame(oe.fit_transform(dat_categ_onehot.astype(str)).toarray(),
                                  columns=oe.get_feature_names(dat_categ_onehot.columns))
        logger.info('End with onehot')
        flag_onehot = True
    else:
        dat_onehot = None
    # le
    if dat_categ_label.shape[1] > 0:
        logger.info('Start to do label encoding to the following categoric features: %s' %
                    (str(dat_categ_label.columns.to_list())))
        dat_categ_label = dat_categ_label.fillna('NULL')
        dat_label = pd.DataFrame(columns=dat_categ_label.columns)
        for i in dat_categ_label.columns:
            dat_label[i] = LabelEncoder().fit_transform(dat_categ_label[i].astype(str))
        flag_label = True
        logger.info('End with label encoding')
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
        logger.info('Nan value exist, start to fill na with iterative imputer: ' +
                    str(dat_new.isna().sum().sum()))
        # include na value, impute with iterative Imputer or simple imputer
        columns = dat_new.columns
        imp = IterativeImputer(max_iter=10, random_state=0)
        # imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        dat_new = imp.fit_transform(dat_new)
        dat_new = pd.DataFrame(dat_new, columns=columns)
    dat_numeric = dat_new.iloc[:, :-1].select_dtypes(include=['float32', 'float64', 'int32', 'int64'])
    logger.info('End with filling nan')
    return dat_new, dat_numeric.columns


def clean_dat(dat: pd.DataFrame, logger=None) -> pd.DataFrame:
    if dat.empty:
        return dat
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    logger.debug('Clean Data, number of inf and nan are for dataset: (%d, %d)' % (
            (dat == np.inf).sum().sum(), dat.isna().sum().sum()))
    logger.info('Set type to float32 at first && deal with inf.')
    dat = dat.astype(np.float32)
    dat = dat.replace([np.inf, -np.inf], np.nan)
    logger.info('Remove columns with half of nan')
    dat = dat.dropna(axis=1, thresh=dat.shape[0] * .5)
    logger.info('Remove costant columns')
    dat = dat.loc[:, (dat != dat.iloc[0]).any()]
    
    dat = dat.loc[:, (dat==0).mean() < .8]
    logger.info('Remove columns with too many so small numbers')
    
    if dat.isna().sum().sum() > 0:
        logger.info('Start to fill the columns with nan')
        # imp = IterativeImputer(max_iter=10, random_state=0)
        imp = SimpleImputer(missing_values=np.nan, strategy='mean')
        # dat = dat.fillna(dat.mean())
        tmp = imp.fit_transform(dat)
        if tmp.shape[1] != dat.shape[1]:
            tmp = dat.fillna(0)
        dat = pd.DataFrame(tmp, columns=dat.columns, index=dat.index)
    logger.info('Remove rows with any nan in the end')
    dat = dat.dropna(axis=0, how='any')
    logger.debug('End with Data cleaning, number of inf and nan are for dataset: (%d, %d)' 
                 % ((dat == np.inf).sum().sum(), dat.isna().sum().sum()))
    return dat


def subsampling(dat: pd.DataFrame):
    """ when number of instance too large, only use 10000 data to do the feature engineering """
    if dat.shape[0] > 10000:
        return dat.sample(n=10000, random_state=1).reset_index(drop=True)
    else:
        return dat


def balanced_sampling(dat: pd.DataFrame, logger=None):
    """ balanced sample data from each class to avoid huge data size bewtween different classes """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    
    # upsampling
    logger.info('Start balanced sampling')
    subsample = []
    num_of_each_class = dat.iloc[:, -1].value_counts().to_numpy()
    if num_of_each_class.std()*1.0 / num_of_each_class.mean() < 0.1:
        logger.info('The given data is balance.')
        # the dataset is balanced
        return dat
    logger.info('Given dataset is unbalance')
    logger.info('Sampling data from each class to generate a new dataset')
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
    logger.info('End with sampling')
    out = pd.concat(subsample)
    out = out.sample(frac=1).reset_index(drop=True)  # shuffle and re index
    return out


def load_data(dataset_name, sep=',', header='infer', index_col=None, logger=None, art='C'):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    logger.info('Load data from file: %s' % (dataset_name))
    dat = pd.read_csv(dataset_name, sep=sep, header=header, index_col=index_col)
    # encode columns with strings and 
    dat, numeric_cols = data_preprocessing(dat, art=art, logger=logger)
    logger.info('End with data preprocessing')
    dat = dat.reset_index(drop=True)
    dat = clean_dat(dat, logger=logger)
    logger.info('Check data size')
    dat.columns = [str(i) for i in dat.columns]
    logger.info('Check data balance')
    # sub sampling, if number of data point >= 10000
    dat = subsampling(dat)
    logger.info('Finish subsampling without balanced_sampling!')
    # N = 1
    # dat = dd.from_pandas(dat, npartitions=N, chunksize=None)
    logger.debug('Successfully load data!')
    return dat


@delayed
def eval(X: pd.DataFrame, y: pd.DataFrame, art='C', logger=None):
    """
    output:
    fitness: type of float, performance of the model;
    fe: type of tuple, contains: name of the most important feature and data
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    # X_train, X_test, y_train, y_test = compute(train_test_d_dat)[0]
    # logger.info('Start to evaluate fitness')
    fitness, std_var = 0.0, 0.0
    if X.isna().sum().sum() > 0:
        print('NA value exist')
        print(X)
        X.to_csv('data/pro.csv')
    try:
        if art == 'C':
            #model = SVC()
            model = RandomForestClassifier(n_estimators= 200, random_state=10, n_jobs=-1)
            scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
            # 'roc_auc'
            fitness = scores.mean()
            std_var = np.std(scores)
            # logger.info('End with model running and return evaluate result')
            # evaluate
            # trained_model = model.fit(X_train, y_train)
            # fitness = roc_auc_score(y_test, predict) # two class
            # fitness = accuracy_score(y_test, predict) # or f1_score
            """ # predict = model.predict(X_test)
            if len(set(predict)) > 2 or len(y_test) > 2:
                # fitness = f1_score(y_test, predict, average='weighted')
                fitness = f1_score(y, predict, average='weighted')
            else:
                # fitness = f1_score(y_test, predict)
                fitness = f1_score(y, predict) """
        else:
            # logger.info('Start to run regression model')
            model = RandomForestRegressor(n_estimators=100, random_state=10, n_jobs=-1)
            def relative_absolute_error(y_true: pd.Series, y_pred: pd.Series):
                y_true_mean = y_true.mean()
                n = len(y_true)
                # Relative Absolute Error
                # err = math.sqrt(sum(np.square(y_true - y_pred)) / math.sqrt(sum(np.square(y_true-y_true_mean))))
                err = sum(abs(y_true - y_pred)) / sum(abs(y_true - y_true_mean))
                return err
            score = make_scorer(relative_absolute_error, greater_is_better=True)
            scores = cross_val_score(model, X, y, cv=5, scoring=score)
            # fitness = scores.mean()
            fitness = 1.0 - scores.mean()
            std_var = np.std(scores)
            # logger.info('End with model running and return evaluate result')
            # evaluate
            # trained_model = model.fit(X_train, y_train)
            # predict = model.predict(X_test)
            # fitness = 1 - mean_absolute_error(y_test, predict)
            # # predict = cross_val_predict(trained_model, X, y, cv=5)
            # # fitness = 1 - mean_absolute_error(y, predict)
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

    """ 
    # get the most important k features
    res = permutation_importance(model, X_train, y_train, n_repeats=5, random_state=0)
    fe_importances = res.importances_mean
    # select the most important k features
    k = math.ceil(X_train.shape[1] * .3)
    ind_fe = np.argpartition(fe_importances, -k)[-k:]
    name_col = X_train.columns[ind_fe]
    """
    # ind_fe = model.feature_importances_
    # name_col = X_train.columns[np.argmax(idx_fe)]
    # logger.debug('Successfully evaluate fitness')
    return fitness, std_var  #, (name_col, X_train[name_col])


def calculateFitness(dat: pd.DataFrame, tribe: pd.DataFrame, art='C', logger=None):
    """ 
    calculate the mean score for one tribe, prepared for further competitions 
    returns a delayed obj, which needs computing-func to get a real number
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    s = eval(tribe, dat.iloc[:, -1], art=art, logger=logger)
    logger.info('Finish calculating the fitness and its standard variance of current tribe/group.')
    return s


def partition(lis: list, n: int):
    """ randomly split a list into n groups """
    # prevent destroying the original dataset
    lis_cp = copy.deepcopy(lis)
    random.shuffle(lis_cp)
    if len(lis) > n:
        return [lis_cp[i::n] for i in range(n)]
    else:
        return [[lis_cp[i]] for i in range(len(lis))]


@delayed
def innerCompetition(dat: pd.DataFrame, groups_num: int, art='C', logger=None) -> list:
    """ 
    dat: cleaned data after preprocessing, subsampling and balanced-sampling 
                   and generating all possible individuals

    labels placed at the last column

    ##output## -> list of groups, each group includes columns of features names and its score, which at the last column
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')

    cols = dat.columns[:-1].tolist()
    # logger.info('Start randomly splitting feature columns into %s groups with almost the same size' % (str(gp_num)))
    gps = partition(cols, groups_num)
    # logger.info('Start evaluating the groups fitnesses 1-by-1')
    
    # dic reserve gp:mean_score pairs
    gps_list = list()
    for gp in gps:
        group = pd.DataFrame(dat.loc[:, gp], columns=gp)
        score = calculateFitness(dat, group, art=art, logger=logger)
        # the last column in gp is its score, which is used for sorting
        gp.append(score)
        gps_list.append(gp)
    # x: [col1, col2,..., (fitness, variance)]
    sorted_gps_list = sorted(compute(gps_list)[0], key=lambda x: x[-1][0], reverse=False)
    # gp_min = sorted_gps_list[0]
    # gp_max = sorted_gps_list[-1]
    # logger.debug('The groups with the worst %s and best %s performances are found.\t ' % (str(gp_min), str(gp_max)))
    logger.debug('Finish competition inner one tribe and return group list with acsending order of their scores.' )
    return sorted_gps_list


def plunge(weak: pd.DataFrame, strong: pd.DataFrame, init_score, art='C', logger=None):
    """ try forming a new group combined the best part of weak_group with not the worst part of strong_group, 
        judge if it can replace the previous best group, and alway eliminate the worst group """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    # magic_number: groups_num=2
    asc_weak = innerCompetition(weak, groups_num=2, art=art, logger=logger)
    cols_best_in_w = compute(asc_weak[-1][:-1])[0]
    best_in_weak = pd.DataFrame(weak.loc[:, cols_best_in_w], columns=cols_best_in_w)
    
    asc_strong = innerCompetition(strong, groups_num=2, art=art, logger=logger)
    cols_not_worst_in_strong = compute(asc_strong[1:][:-1])[0]
    not_worst_in_strong = pd.DataFrame(strong.loc[:, cols_not_worst_in_strong], columns=cols_not_worst_in_strong)
    # cols_worst_in_strong = compute(asc_strong[0][:-1])[0]
    # cols_best_in_strong = compute(asc_strong[-1][:-1])[0]
    new_group = pd.concat([best_in_weak, not_worst_in_strong], axis=1)
    # new_tribe = clean_dat(new_tribe, logger)
    res = calculateFitness(strong, new_group, art=art, logger=logger)
    score = compute(res)[0][0]
    logger.debug('Finish competition between 2 groups, the better one plunges better part in worse to get better, if possible')
    # logger.debug("init_score: %s" %(str(init_score)))
    if (score > init_score):
        return new_group
    else:
        return strong.iloc[:, :-1]


def plungeInnerTribe(dat: pd.DataFrame, sorted_gps: list, art='C', logger=None):
    """
    function: subgroup the weakest and best seperately and 
    try to replace the worst subgroup in the best with the best subgroup in the weakest
    dat: generated children features derived from the same operator, combined with label column, waiting for plunge and die out
    sorted_gps: each element in list is a list including features group combined with its score
    return: selected features without label column any more
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # logger.debug('Start to plunge and eliminate groups')
    # excluding the label column in list element firstly
    best = compute(sorted_gps[-1][:-1])[0]
    gp_b = pd.DataFrame(dat[best])
    gp_b[dat.columns[-1]] = dat.iloc[:, -1]
    # magic_number: search_range
    search_range = min(20, round(len(sorted_gps)/2))
    group = pd.DataFrame(gp_b)
    for i in range(-2, -search_range, -1):
        not_best = compute(sorted_gps[i][:-1])[0]
        gp_not_best = pd.DataFrame(dat[not_best])
        gp_not_best[dat.columns[-1]] = dat.iloc[:, -1]
        # init_score = compute(innerCompetition(group, groups_num=2, art=art, logger=logger)[-1][-1])[0][0]
        init_score = compute(eval(group.iloc[:, :-1], group.iloc[:, -1], art=art, logger=logger))[0][0]
        group = plunge(gp_not_best, group, init_score, art=art, logger=logger)
    logger.info('Finish plunging inner a tribe.')
    return group


def interTribesCompetition(dat: pd.DataFrame, tribes_list: list, art='C', logger=None):
    """ give scores to tribes and return the opr index list sorted by acsending scores  """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # logger.info('Start to generate children candidates in new_cands function')
    
    # oprs_scores list records [index_of_opr, score_of_this_tribe]
    oprs_scores = []
    idx = 0
    for tribe in tribes_list:
        score = calculateFitness(dat, tribe, art=art, logger=logger)
        oprs_scores.append([idx, score])
        idx += 1
    asc_rank = sorted(compute(oprs_scores)[0], key=lambda x: x[1][0], reverse=False)
    logger.debug('Finish competitions inter tribes and according to their fitness update their weights')
    return asc_rank


def plungeInterTribes(dat: pd.DataFrame, tribes_list: list, oprs_ranks: list, oprs_weights: list, art='C', logger=None):
    """ according to the tribes' scores go plunging between them, and update oprs' weights """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # logger.info('Start to generate children candidates in new_cands function')
    
    """ initial weights, weights updating rule """
    # magic_number: dis
    dis = round(-0.5 * len(oprs_ranks))
    for idx, rank in oprs_ranks:
        """ how to update the weights """
        oprs_weights[idx] += 0.5 * dis
        # dis<0 shows the opr is at lower zone, its weight should be decreased  
        if dis < 0:
            oprs_weights[idx] = max(0, oprs_weights[idx])
        dis += 1
    """ inter tribes plunging strategy: """
    final_tribes_list = []
    left = 0
    right = len(oprs_ranks)-1
    while left < right:
        weak_tribe = tribes_list[oprs_ranks[left][0]]
        weak_tribe[dat.columns[-1]] = dat.iloc[:, -1]
        strong_tribe = tribes_list[oprs_ranks[right][0]]
        strong_tribe[dat.columns[-1]] = dat.iloc[:, -1]
        
        init_score = oprs_ranks[right][1][0]
        tribe = delayed(plunge)(weak_tribe, strong_tribe, init_score, art=art, logger=None)
        final_tribes_list.append(tribe)
        left += 1
        right -= 1
    if (left == right): final_tribes_list.append(tribes_list[oprs_ranks[left][0]])
    
    logger.debug('Finish plunging inter tribes.')
    return final_tribes_list


def featureSelection(dat: pd.DataFrame, art='C', logger=None) -> pd.DataFrame():
    """ 
    dat: features + label column
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    feature_names = dat.columns[:-1]
    x = dat.iloc[:, :-1].values
    y = dat.iloc[:, -1].values.ravel()
    # instantiate random forest
    if art == 'C':
        model = RandomForestClassifier(n_estimators= 80, random_state=10, n_jobs=-1)
        # model = ExtraTreesClassifier(n_estimators=100)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=10, n_jobs=-1)
        # model = ExtraTreesRegressor(n_estimators=100)
    # fit boruta
    # boruta_selector = BorutaPy(model, n_estimators = 'auto', verbose=0, random_state=1)
    # boruta_selector.fit(x, y)
    # store results
    # boruta_ranking = boruta_selector.ranking_
    # selected_features = np.array(feature_names)[boruta_ranking <= max_size]
    # supp = boruta_selector.support_
    # X_filtered = boruta_selector.transform(x)
    
    clf = model.fit(x, y)
    # Meta-transformer for selecting features based on importance weights.
    fs = SelectFromModel(clf, threshold='mean', prefit=True)
    supp = fs.get_support()
    X_filtered = fs.transform(x)
    df = pd.DataFrame(X_filtered, columns=dat.iloc[:, :-1].columns[supp], index=dat.index)
    df['target'] = dat.iloc[:, -1]
    logger.debug('End with columns selection, number of columns now is: %s' %(str(df.iloc[:, :-1].shape[1])))
    return df


def newCandidates(cur_dat: pd.DataFrame, prev_gen: pd.DataFrame, oprs_weights=None, logger=None):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    # calculate probability of opr according to their weights
    total_w = sum(oprs_weights)
    oprs_prs = [i/total_w for i in oprs_weights]
    # randomly select ratio of all oprs with non-zero probability, otherwise the number of non-zero probability oprs can't reach size of lis_size
    # magic_number: lis_size
    ratio = 0.9
    effective_cnt = len(oprs_prs) - oprs_prs.count(0)
    lis_size = round(ratio * effective_cnt)
    # not repeatedly select a number of oprs according to the prob
    oprs_list = np.random.choice(oprs_names, lis_size, replace=False, p=oprs_prs)
    logger.debug('Randomly select %d operators to be used to generate children candidates' %(lis_size))
    
    cur_gen = pd.DataFrame(cur_dat.iloc[:, :-1], columns=cur_dat.columns[:-1])
    candidates_list = []
    for operator in oprs_list:
        candidates = None
        if operator in una_oprs.keys():
            opr = una_oprs[operator]
            candidates = delayed(opr()._exec)(cur_gen)
        elif operator in bina_oprs.keys():
            opr = bina_oprs[operator]
            candidates = delayed(opr()._exec)(cur_gen, prev_gen)
        candidates_list.append(candidates)
    cands_bag = db.from_sequence(candidates_list, partition_size=None, npartitions=None)
    logger.debug('Successfully generate new children features candidates')
    return cands_bag


def getNextGeneration(cur_dat: pd.DataFrame, prev_gen: pd.DataFrame=None, oprs_weights=None, art='C', logger=None) -> pd.DataFrame():
    """ 
    input -> cur_dat: newly generated candidates features and label column; prev_gen: father features;
    output -> generated candidates after competition and selection
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    cands_bag = newCandidates(cur_dat, prev_gen=prev_gen, oprs_weights=oprs_weights, logger=logger)
    cands_bag = compute(compute(cands_bag)[0])[0]
    logger.debug('Finish computing children candidates in getNextGen')
    
    cands_lis = []
    for cand in cands_bag:
        cand = delayed(clean_dat)(cand, logger)
        cands_lis.append(cand)
    cands_lis = compute(cands_lis)[0]
    logger.debug('Finish Data Cleaning after new_cands function in getNextGen')
    
    cnt = 1
    res_lis = []
    for cand in cands_lis:
        if cand.empty:
            logger.info('No possible children candidates for the %sth operator' % (str(cnt)))
            continue
        cand_dat = copy.deepcopy(cand)
        cand_dat[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
        # inner tribes competition, each opr forms a tribe
        """ How many groups should be splitted? It should match the expansion of data with O(n^2) """
        gp_num = round(math.sqrt(cand_dat.shape[1]))
        sorted_gps = innerCompetition(cand_dat, groups_num=gp_num, art=art, logger=logger)
        cand_d = delayed(plungeInnerTribe)(cand_dat, sorted_gps, art=art, logger=logger)
        res_lis.append(cand_d)
        logger.info('Successfully plunge&eliminate after tribesCompet. the candidates for the %sth operator' %(str(cnt)))
        cnt += 1
    res_lis = compute(res_lis)[0]
    # res_tup = compute(res_lis)
    # res_lis = res_tup[0]
    logger.debug('Here remain the children surviving from Competion and Elimination')
    
    # inter tribes competition and update the weight of each opr
    ranks = interTribesCompetition(cur_dat, res_lis, art=art, logger=logger)
    final_lis =  plungeInterTribes(cur_dat, res_lis, ranks, oprs_weights, art=art, logger=logger)
    final_lis = compute(final_lis)[0]
    
    res = pd.DataFrame()
    for cand_d in final_lis:
        res = pd.concat([res, cand_d], axis=1)
        # if res is too large, parallel process needs considering
        # cand_d.to_csv('result/temp_res.csv', mode='a', header=False)
    
    # Try to limit size of the whole dataset if needed
    """ cur_limit:  """
    # cur_limit = 200
    # res[cur_dat.columns[-1]] = cur_dat.iloc[:, -1]
    # while (res.shape[1] > cur_limit):
        # logger.info('The number of current_gen columns %d exceed cur_limit %d, columns selection first' % (res.shape[1], cur_limit))
        # res = featureSelection(res, art=art, logger=logger)
    # res.drop(res.columns[-1], axis=1, inplace=True)
    # logger.debug('Number of newly generated features is constrained under cur_limit= %d' %(cur_limit))
    
    logger.debug('This round generates %d children features.' %(len(res.columns)))
    return res


def updateDat(dat: pd.DataFrame, prev_gen: pd.DataFrame=None, oprs_weights=None, art='C', logger=None):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    # coloredlogs.install(level='DEBUG')
    
    nxt_g = getNextGeneration(dat, prev_gen, art=art, oprs_weights=oprs_weights, logger=logger)
    # nxt_g = clean_dat(nxt_g, logger)
    nxt_dat = copy.deepcopy(nxt_g)
    nxt_dat[dat.columns[-1]] = dat.iloc[:, -1]
    cur_gen = dat.drop(dat.columns[-1], axis=1, inplace=False)
    return nxt_dat, cur_gen


def constrainFeaturesNum(dat: pd.DataFrame, limit: int, art='C', logger=None):
    """ 
    The features with the same number of oprs are in the same generation. The size of each generation is like reversed pyramid.
    constrain number of features in prev and the total number of features in all generations.
    """
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    """ try different methods here to get an appropriate approach """
    gen = pd.DataFrame()
    if dat.shape[1] > limit:
        gen = bestFeatures(dat, limit, art=art, logger=logger)
    else: gen = dat.drop(dat.columns[-1], axis=1, inplace=False)
    features_num = dat.shape[1] - 1
    # while dat.shape[1] > limit:
    #     dat = featureSelection(dat, art=art, logger=logger)
    # gen = dat.drop(dat.columns[-1], axis=1, inplace=False)
    # logger.debug('After feature selection, the number of features drops from %d to %d' %(features_num, gen.shape[1]))
    
    return gen, gen.shape[1]


def dropHighCorrelation(dat: pd.DataFrame, logger=None):
    """ drop the children features with high correlations """
    
    gen = dat.drop(dat.columns[-1], axis=1, inplace=False)
    # Create correlation matrix
    corr_matrix = gen.corr().abs()
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    # Find index of feature columns with correlation greater than 0.98
    to_drop = [column for column in upper.columns if any(upper[column] > 0.98)]
    gen.drop(gen[to_drop], axis=1, inplace=True)
    
    logger.debug('Finish dropping feature columns with correlation greater than 0.98, the number of features drops from %d to %d' %(dat.shape[1]-1, gen.shape[1]))
    return gen


def addInitalFeatures(cur_features: pd.DataFrame, pre_features:pd.DataFrame, init_dat: pd.DataFrame, logger=None):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
        
    # add initial features in generation to create a generation gap, keep the diversity of the whole
    # The error: 'cannot reindex from a duplicate axis' is attributed to the same columns added with the existing one
    cols = init_dat.columns[:-1]
    cnt = 0
    for col in cols:
        if col not in cur_features.columns and col not in pre_features.columns:
            cur_features = pd.concat([cur_features, init_dat.loc[:, col]], axis=1)
            cnt += 1
    cur_features[init_dat.columns[-1]] = init_dat.iloc[:, -1]

    logger.debug('Add %d initial features into current generation' %(cnt))
    return cur_features


def bestFeatures(dat: pd.DataFrame, size: int, art='C', logger=None)->pd.DataFrame:
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    # cols = dat.columns[:-1].tolist()
    # cur_size = dat.shape[1]-1
    # groups_num = round(cur_size/size)
    # asc_groups = innerCompetition(dat, groups_num, art=art, logger=logger)
    features = pd.DataFrame(dat.iloc[:, :-1], columns=dat.columns[:-1])
    targets = pd.DataFrame(dat.iloc[:, -1])
    if features.shape[1] <= size:
        return features
    # score_function = None
    # if art == 'C': score_function = mutual_info_classif
    if art == 'C': score_function = f_classif
    # else: score_function = mutual_info_regression
    else: score_function = f_regression
    # Create and fit selector
    selector = SelectKBest(score_function, k=size)
    selector.fit(features, targets.values.ravel())
    # Get columns to keep and create new dataframe with those only
    cols = selector.get_support(indices=True)
    features_selected = features.iloc[:,cols]
    # selected = SelectKBest(score_function, k=size).fit_transform(features, targets)
    logger.debug('Select best %d features in the final features.' %(size))
    return features_selected


def scoreCompare(init_dat: pd.DataFrame, cur_features: pd.DataFrame, art='C', logger=None):
    if logger == None:
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logger = logging.getLogger(__name__)
    
    cur_fitness = calculateFitness(init_dat, cur_features, art=art, logger=logger)
    cur_fitness, cur_std = compute(cur_fitness)[0]
    init_fitness = calculateFitness(init_dat, init_dat.iloc[:, :-1], art=art, logger=logger)
    init_fitness, init_std = compute(init_fitness)[0]
    logger.debug("Initial fitness is %s with the standard variance %s" %(str(init_fitness), str(init_std)))
    logger.debug("After algorithm the final fitness with selected features is %s with standard variance %s" %(str(cur_fitness), str(cur_std)))
    """ should add significance test """
    return init_fitness, cur_fitness