Genetic Algorithm based Feature Engineering

data存放 https://www.openml.org/s/218/data 上的数据集；
log记录每个数据集的运行状态
result保存运行结束后每个数据集产生的最新一代的特征curr，和所有保存下来的前辈们的特征prev；子文件夹是按照各个数据集的名字，其内部是生成的每一代特征

一共有两版代码：
普通版：包含所有操作运算（单目/双目/多目/聚合）的transforms.py, 实现算法的主体部分tribes_competition.py, 运行数据集的主函数feature_engineering.py
并行版：主要使用了dask.delayed函数，实现并行for循环内部操作，减少运行时间，transforms_distributed.py, tc_distributed.py, fe_distributed.py/.ipynb

两版所有主要内容均保持了一致，目前代码tribes_competition部分仍有优化空间，transfoms里面目前也仅仅考虑了单目双目运算

第三版：tc_distributed_pro.py 增加完善了算法中的几个新功能，如群间竞争、群间掠夺、去除相似性高的特征、各个子代平滑增长如金字塔、添加初始特征

本方法的优势：大部分代码段实现并行化提速；扩大探索范围；每个属性的转换路径可以各不相同，传统的是一个最优方法作用到所有特征上

functions:

interTribesCompetition:
给各个种群打分评价,返回种群分数升序的opr索引号列表
plungeInterTrbes:
初始权重为预期生成子代的总代数，即循环次数 
更新权重策略:对下半区的opr每次权重减小，越差的见效的越多，上半区的权重每次增加，越好的增加越多
群间掠夺的策略是:最好的尝试掠夺下半区最差的，次好的对应次差的，可以保证越到后面越接近，可以因此保留到一些偶然分到下半区但值得保留的opr的特征
newCandidates:
根据各个opr的权重所代表的概率，随机选择所有oprs中前 ratio 的概率非零的oprs，否则可能出现概率非零的opr个数不满足所选lis_size个数
根据出现概率随机选择一定数量的opr，不出现重复 
constrainFeaturesNum: 
根据特征所使用的opr的数目分代，即一次循环为一代，使用一个opr的为第一代，构建一个opr数目与特征数目成正比的反金字塔结构
存储所有的prev特征此外也用作限制prev里面特征总数量，即既限制每一代中又限制所有代中特征的数目
tuneParameters: 
可以把结果画图，更直观用于解释调参过程
addInitialFeatures: 
每隔多代就向里面补充一遍初始特征，创建一个 generation gap，减少因为意外导致的特征的丢失的影响，以及子特征与初始结合可能会有更好的效果
如果补充太密集会导致 cannot reindex from a duplicate axis 问题，原因是在prev中的初始特征还没有被淘汰掉，相同的初始和他进行运算会报错


登录 remote desktop connection： login-l.sdil.kit.edu
向sdil平台上传数据的命令行：scp *.csv ujzgk@login-l.sdil.kit.edu:/smartdata/ujzgk/
https://notebooks.sdil.kit.edu/
设置服务器环境：
setup-anaconda
conda env list
source activate env_name
conda create --name=new_env --clone=current_env --copy



各个超参数一览：
Higgs_Boson: model = RandomForestClassifier(n_estimators= 200, random_state=10, n_jobs=-1)
SpectF: model = RandomForestClassifier(n_estimators= 40, max_depth=5, min_samples_split=50, 
                                           min_samples_leaf=30, random_state=10, n_jobs=-1)
messidor_features: model = RandomForestClassifier(n_estimators=170, max_sdepth=50, min_samples_split=30, random_state=10, n_jobs=-1)
Ionosphere_cleaned: model = RandomForestClassifier(n_estimators=80, random_state=10, n_jobs=-1)
Lymphography/winequality_white/winequality_red: model = RandomForestClassifier(n_estimators=150, random_state=10, n_jobs=-1)
German_Credit: model = RandomForestClassifier(n_estimators= 20, max_depth=100, min_samples_split=100, random_state=10, n_jobs=-1)

