# 领先-滞后相关性因子策略

验证并交易一个假设："某只股票（leader）今天上涨后，另一只股票（follower）在
之后第 `lag` 个交易日上涨的概率，显著高于它自身平时的基准概率。" 如果能在历史
数据里稳健地找到这样一批 (leader, follower) 配对，就用它构建一个日频、可在
QMT 里跑回测/模拟盘/实盘的多头轮动策略。

## 目录结构

```
leadlag/            纯 Python 因子挖掘 + 回测引擎（只依赖 pandas/numpy，任何环境都能跑）
  data.py             数据适配层：xtquant.xtdata / CSV / 合成数据，统一转成宽表
  factor.py           领先-滞后挖掘：条件概率、显著性检验（FDR）、样本外验证、部署裁剪
  backtest.py          把挖掘出的配对表变成每日信号，模拟组合净值（含交易成本）
  metrics.py           年化收益/夏普/最大回撤等绩效指标
  stats.py             不依赖 scipy 的正态分布 p 值 + Benjamini-Hochberg FDR 实现
research/
  run_mining.py       离线挖掘 CLI：读数据 -> 训练/测试集切分 -> 挖掘 -> 样本外验证 -> 产出 pairs.csv
  run_backtest.py     回测 CLI：读 pairs.csv + 行情，跑组合回测，输出绩效和净值曲线
strategy/
  leadlag_strategy.py  QMT ContextInfo 策略脚本，读 pairs.csv，可直接粘贴进策略编辑器跑回测/模拟盘/实盘
docs/
  QMT_API_NOTES.md    本项目用到的 QMT 内置 Python API 要点摘录（附官方文档页码）
tests/                 用合成数据做的单元测试，验证挖掘/回测数学逻辑没有 bug
```

数据流：**离线挖掘（你自己的电脑/服务器，配 xtquant.xtdata）→ 产出
`pairs.csv` + `pairs.meta.json` → 拷进 `strategy/` 目录 → 粘贴进 QMT 策略编辑器
跑回测/模拟盘/实盘**。挖掘和实盘用的是两套不同的 API（原因见
`docs/QMT_API_NOTES.md` 第 1 节），但共享同一套 `leadlag/` 数学逻辑。

## 快速开始（不需要任何行情数据，纯跑通流程）

```bash
pip install -r requirements.txt

# 1. 在一个合成的、内置了几组"真实"领先-滞后关系的市场上挖掘
python research/run_mining.py --source synthetic --output research/output/pairs.csv

# 2. 用挖掘出的配对表回测（默认只在挖掘时没见过的样本外窗口上评估）
python research/run_backtest.py --pairs research/output/pairs.csv --source synthetic
```

`tests/` 里的单元测试更严格：一份测试验证在有真实注入信号的合成市场里，挖掘
流程能找回大部分注入的配对且样本外方向不反转；另一份测试验证在纯噪声市场里
（没有任何真实关系）几乎挖不出任何"显著"配对——这是在检验统计显著性检验和
多重检验校正确实在起作用，而不是随便找一堆巧合相关。跑法：

```bash
python -m unittest discover -s tests -v
```

## 接入真实 A 股数据

```bash
python research/run_mining.py --source xtdata --start 20180101 \
    --output research/output/pairs.csv
```

要求：这台机器上跑着 QMT/MiniQMT 客户端且已登录（`xtquant.xtdata` 通过本地
客户端读行情），默认股票池是 `沪深A股`，全市场挖掘一次的耗时和内存量级见下面
"性能与规模"一节。也可以用 `--source csv` 接入你自己已有的行情文件（宽表格式：
日期索引 + 每列一只股票的收盘价/开盘价）。

## 部署到 QMT 实盘/模拟盘

1. `pairs.csv` / `pairs.meta.json` 拷贝到 `strategy/leadlag_strategy.py` 同目录。
2. 打开 `strategy/leadlag_strategy.py`，把 `ACCOUNT_ID` 改成你自己的资金账号。
3. 整个文件粘贴进 QMT 策略编辑器，选日线周期，先跑"回测"核对效果（起止时间、
   初始资金、手续费在回测参数里设置），再切换到模拟盘/实盘。
4. 策略脚本里的关键限制、每个 API 调用为什么这么写，都记在
   `docs/QMT_API_NOTES.md`，遇到"这行代码是不是编的"的疑问先查那份文档。

## 方法论 / 统计陷阱说明（重要，请务必读完再上真金白银）

这类"暴力配对挖掘"策略最大的风险不是代码 bug，而是**看起来很显著、实际是巧
合**。本项目在设计上做了这几层防御，但都不是万能的：

1. **多重检验校正（FDR）**：全市场几千只股票两两配对，就是几百万到千万级别的
   假设检验。哪怕真实世界完全没有任何领先-滞后关系，用 p<0.05 这种平铺阈值也
   会筛出成千上万个"显著"配对，全是巧合。`leadlag/factor.py` 用
   Benjamini-Hochberg 过程控制错误发现率，但这只降低巧合的比例，不能降到零。
2. **样本外验证（更重要）**：即使通过了 FDR 校正，单个配对仍然可能只是某一段
   历史窗口的偶然产物（比如两只票恰好在某个题材周期里一起被资金炒作）。
   `research/run_mining.py` 强制切出训练/测试两段时间，只把训练窗口挖出来的
   候选配对拿到测试窗口上重新算一遍——lift 符号翻转或样本量不够的直接淘汰。
   即便如此，测试窗口通过了，也不代表未来会继续成立（体制变化、题材退潮都可能
   让历史相关性消失）。**建议上实盘前，再单独留一段"策略上线后才发生"的时间做
   前瞻验证，而不是只信历史回测。**
3. **剔除大盘共振（`--mode excess`，默认开启）**：大盘普涨的日子里，几乎所有
   股票都会一起涨，这会让任意两只股票看起来像"领先-滞后"，其实只是共同暴露在
   同一个市场因子上。`compute_up_indicator(mode='excess')` 用当天收益减去当天
   横截面中位数，只保留相对强弱信号，能过滤掉大部分这种伪相关，但不能保证完全
   剔除行业/板块层面的共同因子（比如两只票恰好同属一个板块，板块内資金轮动也会
   造成类似的"领先-滞后"假象）。如果你有行业分类数据，建议在拿到 `pairs.csv`
   后自行剔除同行业/同板块配对。
4. **执行假设是简化的**：回测假设信号触发后能在下一交易日开盘按当时价格、扣除
   佣金/印花税/滑点后完整成交；真实市场里涨停无法买入、跌停无法卖出、大单会有
   冲击成本，`strategy/leadlag_strategy.py` 做了涨停价规避，但没有模拟盘口深度、
   没有做跌停保护、也没有止损逻辑——这些都是可以在此基础上继续加的风控层，
   本项目为了不过度设计先没有做。

**这不是投资建议，历史统计显著不代表未来盈利，实盘前务必用小资金做模拟盘验证。**

## 性能与规模

全市场（~5000+ 只 A 股）挖掘的核心计算量是若干个 `(N x T) @ (T x N)` 矩阵乘法
（`leadlag/factor.py` 的 `mine_lead_lag_pairs`），量级是 `O(N^2 * T)`。N=5000、
T=1000 个交易日（约 4 年）大约是 `2.5*10^10` 次乘加运算 × 4 个矩阵，纯 numpy/BLAS
通常几十秒到几分钟能跑完，内存峰值主要是几个 `N x N` 的 float64 矩阵
（5000x5000 float64 约 200MB/个）。如果机器吃紧或股票池想再大一些，建议先按
成交额/换手率过滤掉流动性太差的股票（比如只留成交额排名前 2000-3000 的），
既降低计算量也避免挖到根本没法建仓的票。

## 依赖

`leadlag/` 只依赖 `pandas` + `numpy`（`stats.py` 自己实现了正态分布 p 值和
BH-FDR，不需要 scipy，这样 `strategy/leadlag_strategy.py` 里如果想直接复用这些
函数，也能在 QMT 内置 Python 环境（不带 scipy）里跑）。`--source xtdata` 需要
`xtquant`，随 QMT/MiniQMT 客户端安装，装法参考迅投官方文档。
