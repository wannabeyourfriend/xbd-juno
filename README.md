# 2025 大作业第二阶段：JUNO calibration probe

在[一阶段大作业](https://git.tsinghua.edu.cn/physics-data/projects/tpl_junosap)中，我们完成了JUNO探测器的模拟和分析，对JUNO的物理过程有了初步了解，并且通过模拟结果绘制了probe函数的图像。如果你忘了什么是probe函数，请及时回顾，务必明确$r$和$\theta$的定义。

我们通常将含时的 probe 函数记为 $R(r,\theta,t)$, 该函数表示在相对坐标 $(r, \theta)$ 下随时间 $t$ 变化的 PMT 上接收到的期望 PE 数. 将时间维度积分后, 得到不含时间的 probe 函数 $\lambda(r,\theta)$. 即: 这两个函数之间的关系是: $\lambda(r,\theta)$ = $\int_0^TR(r,\theta,t)\mathrm{d}t$, 这里 $T$ 是一个足够大的时间窗口, 我们取 1000ns.

第二阶段的任务是这样的，我们准备了一些训练集，每个训练集中都包含10000个固定在z轴上某处的顶点，文件名即对应顶点在z轴上的坐标（单位mm），如：`1000.h5`代表所有顶点都位于 x=0, y=0, z=1000 的位置。运行 `make datas` 即可下载到 data 文件夹中。你需要用这些训练集, 得到一个 probe 函数。

每个 `h5` 文件包含 **`ParticleTruth`** 以及 **`PETruth`** 两个 **dataset**, 其格式为:

**`ParticleTruth`**

包含 10000 个顶点.

| 名称      | 说明         |
| --------- | ------------ |
| `EventID` | 事件编号     |
| `x`       | 顶点坐标x/mm |
| `y`       | 顶点坐标y/mm |
| `z`       | 顶点坐标z/mm |

**`PETruth`**

| 名称        | 说明           |
| ----------- | -------------- |
| `EventID`   | 事件编号       |
| `ChannelID` | PMT 编号       |
| `PETime`    | PE 击中时间/ns |

几何文件沿用一阶段的`geo.h5`，其格式为：

**`Geometry`**

| 名称        | 说明                     |
| ----------- | ------------------------ |
| `ChannelID` | PMT 编号                 |
| `theta`     | 球坐标 $\theta$ **角度** |
| `phi`       | 球坐标 $\phi$ **角度**   |

## 任务说明

相信大家还记得课上介绍的GAM和GBM方法。我们已经给大家准备好了这两种方法的基本模板，当然目前的效果很不理想，你们可以以这两个模板为基础进行优化（更推荐），也可以自行探索其他方法（时间可能比较紧张）。总之，你需要造出一个 probe 函数, 尽量使评分更高. 可以添加额外的文件。

为了大家更好地理解这两种方法，我们为大家准备了一些[参考文献](https://learn.tsinghua.edu.cn/f/wlxt/kcgg/wlkc_ggb/teacher/beforeViewJs?wlkcid=2024-2025-3150523888&id=26ef84e898048e4301981c7476774c3e)。可能的优化方向比如：调整模型参数、调整用于模型训练的网格数据的分bin策略等。

## 注意事项

1. 请不要私自更改 `coefficient.py` 与 `draw.py` 来作弊, 否则成绩无效。例如，`probe_{gam}{gbm}.py` 中的 `class Probe` 继承了 `coefficient.py` 中的 `class ProbeBase`, 因此除了两个抽象函数, 其它函数也可以重写. 因此你的确可以通过直接重写 `validate` 函数给自己一个高分, 这样的情况直接判零分。

2. 训练集和测试集的顶点类型不同，因而需要一个标度因子去修正probe函数，因此你可以看到`probe_{gam}{gbm}.py`从`config.py`中import了FACTOR这个变量。我们要求这个量对于不同方法（包括你探索的新方法）和不同函数（`get_mu`和`get_lc`）都要保持一致。因此，你只能在`config.py`中尝试修改这个值来提高评分，而不能在其他任何位置修改FACTOR这个变量。

3. 我们将提供一个测试集供大家在本地评分作为参考. 运行`make concat.h5`即可下载测试集。但是, **为了防止大家对着测试集过拟合, 本地评测分数及CI排行仅供参考, 最终的排名将由隐藏测例来决定!**

4. `gam.R`中有一长段代码，请同学们不要修改，脚本中已具体说明。

5. 如无特别需求，不要将训练模型等中间文件push上来。超过10M的大文件绝对禁止push，如果违反，扣光白盒分。

## Makefile说明

- 运行 `make score_{GAM}{GBM}%` 可以得到相应模型的分数。这里%的形式是 **数字\_数字**，用来指定用于模型训练的网格数据的分bin数，如：`20_50` 意味着空间上划分20个bin，时间上划分50个bin。更详细内容可以参阅代码。

- 运行 `make draw/{GAM}{GBM}%.pdf` 可以得到相应的图。%含义和之前相同。

- 我们定义了默认target，`make score` 和 `make draw` 就可以评分和画图，`Makefile`中已经通过`default_method`和`default_bins`为该target指定了默认模型，你可以修改它。排行榜上的分数以`make score`结果为准。

## 附: 评分说明

给定顶点和 PMT 的相对坐标 $(r,\theta)$, 我们可以得到一个 PE 时间序列 $\vec{z}=(t_1,t_2,...t_k)$, 这里 $k$ 是序列长度. 这也是一个非齐次泊松过程, 其似然函数为:

$P(\vec{z}|r,\theta)=e^{-\int_0^TR(r,\theta,t)\mathrm{d}t}\prod_{j=1}^kR(r,\theta,t_j)$

如果我们有 $N$ 次采样, 那么似然函数就是:

$\mathcal{L}=\prod_{i=1}^NP(\vec{z_i}|r_i,\theta_i)=\prod_{i=1}^Ne^{-\int_0^TR(r_i,\theta_i,t)\mathrm{d}t}\prod_{j=1}^{k_i}R(r_i,\theta_i,t_{ij})$

取对数得到:

$\log \mathcal{L}=\sum_{i=1}^N(-\int_0^TR(r_i,\theta_i,t)\mathrm{d}t + \sum_{j=1}^{k_i}\log R(r_i,\theta_i,t_{ij}))$

评分时, 会将你的 probe 函数 $R(r,\theta,t)$ 输入到 $\log \mathcal{L}$ 中。所以评分实际上是个 log likelihood，因此可能有负值。分数越大越好。

各位提交后，可以在 Build - Pipelines 页面手动触发 CI 评测并上传分数，见[排行榜](https://leaderboard.thudep.com/)。此外，CI 的镜像基于[`python:3.13.5-bookworm`](https://github.com/adamanteye/images/blob/physics-data/Dockerfile)并且打包了数据集，如果镜像中不存在你想要的库，请联系助教修改镜像或在 Makefile 中用 pip 添加清华源进行安装。

## My solutions

使用virtual env来管理env环境



setup environment
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```

download dataset
```bash
make datas
```
train and validate model





