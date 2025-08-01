import argparse
import numpy as np
import pyarrow.parquet as pq
import lightgbm as lgb
import pickle

# 读入命令行
psr = argparse.ArgumentParser()
psr.add_argument("-i", dest="ipt", nargs="+", type=str, help="input parquet files")
psr.add_argument("-o", dest="opt", type=str, help="output pkl file")
args = psr.parse_args()

# 第一个模型gbm.s训练
table_s = pq.read_table(args.ipt[0])
x_s = table_s["x"].to_numpy()
y_s = table_s["y"].to_numpy()
r_s = np.sqrt(x_s**2 + y_s**2)
nEV_s = table_s["nEV"].to_numpy()
nPE_s = table_s["nPE"].to_numpy()

log_nEV_s = np.log(nEV_s + 1e-8)
X_s = np.stack([x_s, y_s, r_s, r_s**2, log_nEV_s], axis=1)
target_s = nPE_s
gbm_s = lgb.LGBMRegressor(
    objective = "poisson",
    metric = "poisson",

    # 学习过程
    n_estimators = 1500,
    learning_rate = 0.02,

    # 基础结构
    min_child_samples = 50,
    num_leaves = 63,
    max_depth = 6,

    # 正则化
    reg_alpha = 3.0,
    reg_lambda = 5.0,
    min_split_gain = 0.05,
    colsample_bytree = 0.8,
    subsample = 0.8,
    subsample_freq = 1,
    verbose = -1
    )
gbm_s.fit(X_s, target_s)
print(gbm_s)

# 第二个模型gbm.t训练,比第gbm.s多了时间参数t
table_t = pq.read_table(args.ipt[1])
x_t = table_t["x"].to_numpy()
y_t = table_t["y"].to_numpy()
r_t = np.sqrt(x_t**2 + y_t**2)
t_t = table_t["t"].to_numpy()
nEV_t = table_t["nEV"].to_numpy()
nPE_t = table_t["nPE"].to_numpy()

t_bins = np.unique(t_t)
t_binwidth = t_bins[1] - t_bins[0]

log_nEV_t = np.log(nEV_t + 1e-8)
X_t = np.stack([x_t, y_t, r_t, r_t**2, t_t, log_nEV_t], axis=1)  
target_t = nPE_t
gbm_t = lgb.LGBMRegressor(
    objective = "poisson",
    metric = "poisson",

    # 学习过程
    n_estimators = 2000,
    learning_rate = 0.01,
    
    # 基础结构
    min_child_samples = 30,
    num_leaves = 127,
    max_depth = 7,

    # 正则化
    reg_alpha = 1.0,
    reg_lambda = 2.0,
    min_split_gain = 0.01,
    colsample_bytree = 0.8,
    subsample = 0.8,
    subsample_freq = 5,
    verbose = -1
    )
gbm_t.fit(X_t, target_t)
print(gbm_t)

# 输出得到的模型
with open(args.opt, "wb") as fout:
    pickle.dump({"model_s": gbm_s, "model_t": gbm_t, "t_binwidth": t_binwidth}, fout)
