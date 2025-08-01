import argparse
import numpy as np
import h5py
import os
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm
from config import *

psr = argparse.ArgumentParser()
psr.add_argument("-g", dest="geo", type=str, help="geometry file")
psr.add_argument("-i", dest="ipt", nargs="+", type=str, help="input h5 files")  # 支持多个输入文件
psr.add_argument("-o", dest="opt", nargs="+", type=str, help="output parquet files")  # 输出两个 parquet 文件
psr.add_argument("-b", dest="Bins", type=int, help="number of spatial bins")
psr.add_argument("-t", dest="T_Bins", type=int, help="number of timing bins")
args = psr.parse_args()

# 读取PMT几何数据
with h5py.File(args.geo, 'r') as h5file_r:
    geo = h5file_r["Geometry"][...]
geo_theta = np.deg2rad(geo["theta"][:])

# 计算分bin
# y_bins = np.arange(args.Bins+1) / args.Bins
# x_bins = np.concatenate((-y_bins[::-1], y_bins[1:]))
# t_bins = np.arange(int(args.T_Bins)+1) / int(args.T_Bins) * T_MAX
def get_spatial_bins(n_bins):
    """空间分bin：0-0.8均匀分bin，0.8-1.0加密"""
    linear_part = np.linspace(0, 0.8, int(n_bins * 0.8) + 1)
    edge_part = np.linspace(0.8, 1.0, n_bins - int(n_bins * 0.8) + 1)
    return np.unique(np.concatenate([linear_part, edge_part]))

def get_time_bins(n_bins):
    """时间分bin：0-80ns密集，80-T_MAX稀疏"""
    early = np.linspace(0, 80, int(n_bins * 0.3) + 1)
    late = np.linspace(80, T_MAX, n_bins - int(n_bins * 0.3) + 1)
    return np.unique(np.concatenate([early, late]))
#     """
#     基于PE时间分布的连续加权分bin：
#     - 0-100ns：快速上升峰（指数权重）
#     - 100-400ns：长尾衰减（线性+指数混合）
#     - 400-1000ns：背景区（均匀分布）
#     """
#     def weight_function(t):
#         """混合权重函数"""
#         return np.where(
#             t < 60,
#             0.1,
#             np.where(
#                 t < 120,
#                 0.9,
#                 np.where(
#                     t < 400,
#                     0.1 + 0.8*np.exp(-(t-80)/150),  # 100-400ns混合衰减
#                     0.05  # 400+ns基础权重
#                 )
#             )
#         )

#     # 生成采样点
#     t_samples = np.linspace(0, T_MAX, 10000)
#     weights = weight_function(t_samples)
    
#     # 构建累积分布
#     cdf = np.cumsum(weights)
#     cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])  # 归一化
    
#     # 生成分bin边界
#     quantiles = np.linspace(0, 1, n_bins + 1)
#     bin_edges = np.interp(quantiles, cdf, t_samples)
    
#     return np.unique(bin_edges)

# def generate_centered_inverse_bins(left_bound, right_bound, n_bins):
#     """生成以0为中心的平方反比分bin（严格约束在[left_bound, right_bound]）"""
#     center = 0.0
#     r_min = abs(center - left_bound) # 0.1017
#     r_max = abs(center - right_bound) # 2.1017
    
#     # 生成半径采样点（从r_min到r_max）
#     r_samples = np.linspace(r_min, r_max, 1000)
    
#     # 平方反比权重（软化处理）
#     weights = 1 / (r_samples**2)
#     cdf = np.cumsum(weights)
#     cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0]) # 归一化
#     # 生成分bin边界（从r_min到r_max）
#     divi = np.linspace(0, 1, 2*n_bins+1)
#     bin_edges = np.interp(divi, cdf, r_samples)
    
    
#     # 转换为x坐标（从中心向两侧扩展）
    
#     constrained_bins = 19.5/17.7-bin_edges
#     bins = constrained_bins[::-1]

#     return bins

# # 参数设置
# left_bound = 1.8 / 17.7 # ≈0.1017
# right_bound = 37.2 / 17.7 # ≈2.1017


# # 生成分bin边界
# x_bins = generate_centered_inverse_bins(left_bound, right_bound, args.Bins)

# y_bins = np.arange(args.Bins+1) / args.Bins

# 在数据处理部分应用
y_bins = get_spatial_bins(args.Bins)
x_bins = np.concatenate((-y_bins[::-1], y_bins[1:]))
t_bins = get_time_bins(args.T_Bins)


# 累加所有输入文件
nev, npe, x_sum, y_sum = 0, 0, 0, 0
for fname in tqdm(args.ipt):
    fname_base = os.path.basename(fname)
    z_val = float(os.path.splitext(fname_base)[0])
    with h5py.File(fname, 'r') as h5file_r:
        PETruth = h5file_r["PETruth"][...]
    z = z_val / R0
    r = np.full((N,), z, dtype='float32')
    θ = geo_theta
    px = r * np.cos(θ)
    py = r * np.sin(θ)
    x_sum += np.histogram2d(px, py, bins=[x_bins, y_bins], weights=px)[0]
    y_sum += np.histogram2d(px, py, bins=[x_bins, y_bins], weights=py)[0]
    nev += np.histogram2d(px, py, bins=[x_bins, y_bins])[0]
    npe += np.histogramdd(([px[PETruth['ChannelID']], py[PETruth['ChannelID']], PETruth['PETime']]), bins=[x_bins, y_bins, t_bins])[0]

nev_t = np.repeat(nev[:, :, np.newaxis], args.T_Bins, axis=2)
use_s = nev > 0
use_t = nev_t > 0

nev_s = nev[use_s]
x_sum_s = x_sum[use_s]
y_sum_s = y_sum[use_s]
x_mean_s = x_sum_s / nev_s
y_mean_s = y_sum_s / nev_s
npe_s = np.sum(npe, axis=2)[use_s]

nev_t = nev_t[use_t]
npe_t = npe[use_t]
x_sum_t = np.repeat(x_sum[:, :, np.newaxis], args.T_Bins, axis=2)[use_t]
y_sum_t = np.repeat(y_sum[:, :, np.newaxis], args.T_Bins, axis=2)[use_t]
x_mean_t = x_sum_t / nev_t
y_mean_t = y_sum_t / nev_t

pt_ini = (t_bins[1:] + t_bins[:-1]) / 2
t_mean = np.tile(pt_ini, (args.Bins*2, args.Bins, 1))[use_t]

table_s = pa.Table.from_arrays([x_mean_s.astype('float32'), y_mean_s.astype('float32'),
                                nev_s.astype('uint32'), npe_s.astype('uint32')],
    names=["x", "y", "nEV", "nPE"])

table_t = pa.Table.from_arrays([x_mean_t.astype('float32'), y_mean_t.astype('float32'), t_mean.astype("float32"), 
                                nev_t.astype('uint32'), npe_t.astype('uint32')],
    names=["x", "y", "t", "nEV", "nPE"])

pq.write_table(table_s, args.opt[0], compression="ZSTD")
pq.write_table(table_t, args.opt[1], compression="ZSTD")


