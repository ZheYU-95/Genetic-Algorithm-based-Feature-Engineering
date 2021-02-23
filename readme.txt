Genetic Algorithm based Feature Engineering

data存放 https://www.openml.org/s/218/data 上的数据集；
log记录每个数据集的运行状态
result保存运行结束后每个数据集产生的最新一代的特征curr，和所有保存下来的前辈们的特征prev

一共有两版代码：
普通版：包含所有操作运算（单目/双目/多目/聚合）的transforms.py, 实现算法的主体部分tribes_competition.py, 运行数据集的主函数feature_engineering.py
并行版：主要使用了dask.delayed函数，实现并行for循环内部操作，减少运行时间，transforms_distributed.py, tc_distributed.py, fe_distributed.py/.ipynb

两版所有主要内容均保持了一致，目前代码tribes_competition部分仍有优化空间，transfoms里面目前也仅仅考虑了单目双目运算
