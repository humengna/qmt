# 盘中触发 - 同日板块跟涨策略

**两种 leader 触发定义**（`run_intraday_mining.py --trigger`）：

| `--trigger` | 定义 | 幅度可调？ |
|---|---|---|
| `limitup`（默认） | 当日首次触及涨停价 | ❌ 由板制决定 |
| `surge`（急拉） | 最近 `--surge-window` 根K线内首次涨到 `--leader-threshold` | ✅ **幅度和时间窗口都是参数** |

`surge` 和"从开盘累计涨X%"是不同的东西：一只票4小时磨上去3%不算急拉，10分钟
拉上去3%才算，而板块跟风恰恰是被**陡峭度**引发的。回看窗口不跨交易日，所以隔夜
跳空永远不算急拉，每天每票最多触发一次。

> **⚠️ T+1：个股当天买入不能卖出**，所以本目录回测建模的"同日平仓"对A股个股
> **不可执行**。挖掘本身仍是有效的研究（而且盘中触发是比任何日线信号都更精确的
> **入场**时点），但要落地必须改成持有到次日再卖。T+0 品种是另一回事，见
> `research/run_etf_mining.py`——不过那条假设已因其它原因被否定。

`limitup/`（次日开盘版）的盘中版本："某只龙头股当日某根 K 线首次触及涨停价后，
同板块内某只跟随股在之后 `lag_bars` 根 K 线内（**同一交易日**）上涨的概率，
显著高于它自身的基准概率"。这是"龙头封板带动板块跟风股补涨"这个民间说法里，
真正对应盘中动作的版本——`limitup/` 验证的是"隔夜"传导，这里验证的是"当天
下午/几十分钟内"的传导。

## 为什么这次要重写挖掘/回测的对齐方式，不能直接套 leadlag

`leadlag`/`limitup` 的核心技巧是：把"leader 第 t 天触发"和"follower 第 t+lag
天的结果"做矩阵位移（`.iloc[:-lag]` / `.iloc[lag:]`）后一次性做矩阵乘法。这依赖
一个前提：follower 的"结果"指标是一个**独立于 t 的、只挂在 t+lag 这一天本身**
的量（"t+lag 这天涨不涨"），位移操作负责把它和"t 天的 leader 状态"对齐。

盘中同日传导天然不满足这个前提：follower 的"结果"是"**从触发的那根 K 线开始
算，往后数 lag_bars 根，价格有没有涨**"——这个量本身就是以触发时刻 t 为起点算
出来的前瞻收益，t 和结果在同一行，不需要（也不能）再额外做一次位移
（`leadlag.factor` 里专门写了一条注释：`lag=0` 时 `arr[:-0]` 在 numpy/pandas 里
等价于 `arr[:0]`，即空数组——不是"不切片"，是一个容易踩到的坑）。

`leadlag/factor.py` 为此新增了"同行对齐"版本
（`compute_pairwise_stats_same_row` / `validate_out_of_sample_same_row`），跟
"位移对齐"版本共用同一段两比例 z 检验核心代码（`_two_proportion_matrix_stats`
/ `_two_proportion_pair_stat`），行为上互不影响——`leadlag`/`limitup` 的全部测试
在这次重构后原样通过。

## 目录结构

```
intraday/
  data.py      拉分钟线+日线、涨停价广播、首次触板检测(按天去重)、
                日内涨幅阈值首次触发检测(按天去重，T0 ETF 用)、
                跨日边界感知的"从 t 到 t+lag_bars 是否上涨"前瞻指标构造、
                分钟级合成数据生成器（涨停版 + 阈值版）
  event.py     把上面这些拼成 build_leader_frames（涨停版）/
                build_leader_frames_threshold（阈值版）/ build_follower_frames
  backtest.py  盘中回测引擎：触发即建仓、持有恰好 lag_bars 根K线、
                当天最后一根K线强制清仓（不留隔夜仓位）
research/
  run_intraday_mining.py    离线挖掘 CLI（涨停触发，全市场/按行业配对）
  run_intraday_backtest.py  回测 CLI（涨停触发）
  run_etf_mining.py         离线挖掘 CLI（T0 ETF，日内涨幅阈值触发，不分行业）
  run_etf_backtest.py       回测 CLI（T0 ETF）
strategy/
  intraday_strategy.py      QMT ContextInfo 策略脚本，运行在分钟周期上
tests/
  test_intraday.py          合成数据测试：涨停价公式精确匹配、阈值触发正确性、
                            每天每票最多触发一次、前瞻窗口不跨日、挖掘找回
                            注入信号、纯噪声样本外FDR拒绝、回测端到端跑通+
                            验证退出调度不跨日（涨停版 + 阈值版都覆盖）
```

## T0 ETF 版本：为什么不能直接用涨停触发

`run_intraday_mining.py`/`build_leader_frames` 的触发定义是"当日某根 K 线首次
**触及涨停价**"——这对个股成立，但这套项目最初面向个股设计时忽略了一个前提：
**A 股普通股票是 T+1**，当天买入当天不能卖出，"同日盘中触发买入 → 同日盘中
卖出"这个回测/实盘设计对个股根本不可执行。能在 A 股市场里做真正日内回转交易
的，只有部分 **T0 可交易品种**（跨境 QDII/港股通 ETF、商品 ETF、部分债券
ETF 等）——这些品种：

1. 绝大多数不会真的封涨停（10%/20% 的价格限制对它们要么不适用要么极少触发），
   用涨停公式当触发条件基本挖不到东西；
2. T0 资格本身**没法在这套代码里用编程方式验证**（`docs/QMT_API_NOTES.md`
   翻遍了 QMT 官方 PDF 也没找到 T0/T+1 规则的接口说明）——`run_etf_mining.py`
   里的 `SYMBOL_LIST` 是用户自己确认过的 T0 品种清单，不是本项目推断出来的。

`run_etf_mining.py`/`build_leader_frames_threshold` 把触发条件换成了更朴素的
"**从当天第一根 K 线到当前 K 线，累计涨幅首次达到 `--leader-threshold`
（默认 1%）**"（`compute_first_threshold_cross_indicator`），不再依赖涨停价
公式、不需要日线收盘价做基准；同时**不做行业/板块限定**，直接在用户提供的
固定 ETF 全集内两两配对挖掘（`sector_map=None`）——这些 ETF 之间没有申万一级
行业结构，"同板块"这个概念对它们不适用。挖掘/样本外验证/FDR/回测引擎全部
复用 `leadlag.factor` 的同行对齐版本和 `intraday/backtest.py`，跟涨停版本
完全一致，唯一区别就是这一个"触发定义"。

**注意合成数据里的噪声尺度**：`make_synthetic_threshold_market` 的背景波动率
比涨停版 `make_synthetic_intraday_market` 小了一个数量级——涨停版是为了让
个股级别的波动率能通过复利滚到 10%~20% 的涨停价而调的，同样的波动率下，
一个"1% 阈值"在 47 根 K 线内几乎必然被随机噪声触发（已用真实合成数据验证过：
提高波动率后，跟任何 leader 都无关的股票也会大量"触发"）。ETF 本身是分散化
持仓，日内波动天然比个股小得多，1% 在真实 ETF 上本来就应该是一个有信息量的
稀有事件，合成数据的噪声尺度是照这个直觉调的，不是随便选的。

### 【最终结论】这个假设已经被完整否定 —— 工具保留，策略不要上实盘

> **一句话：这个信号的毛边际是 1.4~3bp，比中国市场上最便宜的佣金还小。**
>
> 不是"滑点太高"、不是"执行不够快"、不是"参数没调对"。在**零滑点、零价差、
> 零印花税**的理想世界里，光是万分之一（1bp/边）的佣金就已经把它吃穿了。
> 没有任何执行层面的改进能救它，因为要救它需要的是负成本。

在真实 xtdata 上一共跑了七轮，每一轮都是否定的：

| # | 做了什么 | 结果 |
|---|---|---|
| 1 | 5m，85只，老窗口（测试窗口起点 2023-09） | 挖到 10 个样本外FDR显著配对，回测 **-27%**，8个follower无一盈利 |
| 2 | 同上，但零成本重跑 | **+19.1%**（Sharpe 4.47）→ 毛利方向是对的，是成本吃掉的 |
| 3 | 5m，85只，窗口拉长到 2022-01→2026-09 | 389个候选，225个保持正号（58%，接近扔硬币），**0个**通过样本外FDR |
| 4 | 5m 参数网格（12组），2020-01→2024-06，top-liquid 40 | hub污染 **74~92%**；阈值≥2% 全部0候选（功效不足） |
| 5 | 第4轮的干净配对前推到 2024-06→2026-09（完全未见过） | 零滑点 **+1.44% / -3.95%**；5bp滑点 -10.6% / -24.2%；**盈亏平衡滑点 ≈0.6bp/边** |
| 6 | 换 1m 粒度重做网格（6组），全85只 | hub污染升到 **93~97%**；标的×2.1、样本外显著×7，**可部署数纹丝不动（10→11）** |
| 7 | 第6轮配对前推，按真实费率 | **零滑点也是负的：-0.75% / -0.67%**；盈亏平衡总往返成本 1.37~1.77bp **＜ 佣金本身的 2bp** |

第7轮是决定性的。用两个成本点反解（总收益对成本近似线性）：

| 组合（1m，前推2.2年） | 零滑点(1bp佣金) | 5bp滑点 | 盈亏平衡所需总往返成本 |
|---|---|---|---|
| 1.5% / 10分钟 | **-0.75%** | -33.16% | **1.77bp** |
| 1.5% / 30分钟 | **-0.67%** | -11.30% | **1.37bp** |

而 1bp/边的佣金一项就是 **2bp 往返**。换个说法：1块钱的跨境ETF，最小变动价位
0.001元 = 10bp，**这个信号每笔往返的全部毛利只有 0.15~0.3 个最小变动价位**，
而买入吃卖一、卖出砸买一，一个往返就要付掉约一整个最小变动价位。
**利润比市场的价格颗粒度还小，物理上就不可能被市价单捕获。**

#### 三个机制（为什么会这样）

1. **共同因子污染**。少数几个 follower 反复出现、同时对应十几个不同的 leader
   ——不是"A带动B"，而是这几个 follower 对全池共享因子（隔夜海外市场/商品
   价格）beta 特别高，只要池子里随便哪个成分先冲了阈值（往往意味着共同因子
   已经在动），它们随后大概率也涨。85只标的高度同质（全是跨境/商品/债券ETF），
   `mode='excess'` 用这85只自己的截面中位数去剔共同因子，对高beta名字剔不干净。
   扩大池子只会放大污染：从40只加到85只，多出来的45只正是低流动性的薄跟踪
   基金，最容易跟着大盘飘——样本外显著数涨了7倍，**可部署数一个没多**。
2. **粒度变细反而更差**。1分钟线能缩短下单延迟（`quickTrade=0` 是"下一根K线
   首个tick成交"），本以为能改善滑点；但它同时捕捉到大量"冲上1%又马上回落"
   的尖峰——这些在5分钟收盘价上根本不会被记为触发。多出来的样本不是信号，
   是最不该追的那类。净效果：毛边际从 ~3bp 降到 ~1.4bp。
3. **tick 地板**。0.001元的最小变动价位与K线周期、参数、池子大小全都无关。
   这是价格离散化的硬约束，不是能靠优化解决的问题。

#### 留下来的东西

**方法论是通用的**，跟这个具体假设无关，下一个因子想法可以直接复用：
FDR多重检验控制 → 样本外重新验证（必须是真FDR，不能只看符号）→ hub-follower
过滤 → 参数网格 → **完全未见过的前推窗口** → 按真实费率反解盈亏平衡成本。
这六道关每一道都在该拦的时候拦住过一次错误结论：按第1轮那10个"样本外FDR显著"
的配对直接上实盘，这2.2年会亏掉三成。

**下次要先做的一步筛选**（这次是倒着学会的）：在挖信号之前，先算
「最小变动价位 ÷ 价格」，确认标的物理上支不支持你要的边际量级。

| 品种 | 大致价格 | 0.001元相当于 |
|---|---|---|
| 债券ETF（511010等） | ~100元 | ~0.1bp |
| 黄金ETF（518880） | ~7-9元 | ~1.3bp |
| 跨境ETF（513050等） | ~1元 | **~10bp** ← 本次全军覆没的那批 |

1块钱的跨境ETF配上3bp的目标边际，这一步在最开始做就能一眼看出不成立。

---

以下是这条链路上做出来的工具和踩过的坑，**工具本身是好的，只是这个假设不成立**。

#### ETF 的成本结构跟股票不一样（务必看清楚）

- **印花税：ETF 免征**。印花税只对**个股**的卖出方征收，交易所交易的基金
  （ETF/LOF）是免税的。`intraday/backtest.py` 里 `stamp_tax_bps=5.0` 的默认值
  是给股票管道用的；`run_etf_backtest.py` 的 `--stamp-tax-bps` **默认已改为 0**。
  （这里踩过坑：最早几次 ETF 回测沿用了股票的 5bp 印花税，等于每个往返多收
  5bp——对一个毛利只有十几bp的信号来说，这个误差不小。）
- **佣金**：按你券商实际费率填。万分之一（1bp）这种折扣价很常见。每笔最低
  5 元这类门槛在本回测的仓位规模下**不会触发**：仓位是 `权益/--top-k`，就算
  权益跌到 50 万、top_k=5，单仓也有 10 万，1bp 就是 10 元，已经高过 5 元下限。
- **滑点不是手续费，不能因为"没有其他费用"就填 0**。回测的成交价用的是 K 线
  **收盘价**，而实盘市价单买要吃卖一、卖要砸买一。国内 ETF 报价最小变动是
  0.001 元，一只 1 块钱左右的跨境 ETF，一个最小变动价位就是约 10bp 的价差，
  也就是**光穿价差每边就要付约 5bp**。`--slippage-bps 0` 量出来的是**毛边际**，
  不是一个真能成交的价格。

**成本敏感性**：`run_etf_backtest.py` 的 `--commission-bps`/`--slippage-bps`
/`--stamp-tax-bps` 可以覆盖 `IntradayBacktestConfig` 的默认值，用来分离
"信号本身"和"成本假设"这两件事，而不用改代码。**跑两个成本点就能反解出盈亏
平衡所需的成本**（总收益对成本近似线性），这是上面第5、7轮那个决定性数字的
来源，比争论"滑点到底该填几"有用得多：

```bash
# 点1：零滑点 = 毛边际上限（不是一个真能成交的价格）
python research/run_etf_backtest.py --pairs research/output/etf_pairs.csv --source xtdata \
    --start 20240601 --eval-start none --commission-bps 1 --slippage-bps 0
# 点2：加上现实的半个价差
python research/run_etf_backtest.py --pairs research/output/etf_pairs.csv --source xtdata \
    --start 20240601 --eval-start none --commission-bps 1 --slippage-bps 5
```

**前推检验要加 `--download`**：`--source xtdata` 默认 `download=False`（常规
用法是回测刚挖过的那段，本地已有）。但前推检验的 `--start` 在挖掘窗口之后，
那段历史从没下载过，不加 `--download` 会读到空数据。同时 `--eval-start none`
是必须的，否则它会去读 meta.json 里记的老测试窗口起点，又绕回挖掘用过的数据上。

**hub follower 排除**：`leadlag.factor.exclude_hub_followers(pairs, max_leaders_per_follower)`
——在样本外 FDR 通过之后、进实盘符号预算裁剪之前，把"对应了超过
`max_leaders_per_follower` 个不同 leader"的 follower 整个剔除（`=0` 关闭这个
过滤）。`run_etf_mining.py` 默认 `--max-leaders-per-follower=3`。

这只是一个粗粒度的、单变量的安全阀，不会真的把共同因子从数据里剔除干净——
真要根治，需要在算 follower 的 forward outcome 时先对某个基准做 beta 回归、
只保留残差（**未实现**：第7轮证明了即使剔干净也没用，剩下的边际本来就低于
佣金，所以没有继续投入）。但它作为**诊断工具**非常有价值：被它过滤掉的比例
（本次 74%→97%）直接量化了"共同因子污染"有多严重。

#### 做对照实验时的两个坑

**务必用固定、明确的 `--start`/`--end`**：`run_etf_mining.py` 和
`run_etf_param_sweep.py` 的 `--start`/`--end` 默认都是空字符串（等价于"能拉
多少拉多少"），两次不传日期的运行拉到的历史范围可能不一样（"能拉多少"还会
随时间推移变化）。本次就踩过：一次漏传日期导致窗口从 2020-2024 变成
2022-2026，结果"0个通过FDR"到底是参数生效还是窗口变了，一时分不清。

**`--output-dir` 要分开**：网格搜索的输出文件名只带阈值和lag（`pairs_thr0.015_lag10.csv`），
**不带周期和池子大小**。1分钟/40只 和 1分钟/85只 两次扫描会互相覆盖。做对照
实验时给每组加 `--output-dir research/output/etf_sweep_1m_full` 之类的独立目录。

#### 参数网格搜索：`run_etf_param_sweep.py`

系统扫一遍 `--leader-threshold` 和 `--lag-bars` 的组合，看有没有哪一组配置的
关系不只是某个特定参数点的巧合。只拉一次分钟线数据，在内存里对网格里的每一组
参数分别重新做触发检测 + 挖掘 + 样本外FDR + hub-follower过滤，比对每个组
合分别跑一次 `run_etf_mining.py`（重复下载/解析同一份数据）快得多：

```bash
python research/run_etf_param_sweep.py --source xtdata --start 20200101 --end 20240601 \
    --top-liquid 40 --thresholds 0.01 0.015 0.02 0.03 --lag-bars-list 3 6 12
```

`--top-liquid`（也可以单独在 `run_etf_mining.py` 上使用）把 85 只 ETF 全集缩小
到成交额最高的 N 只——这既是在缓解 hub follower 那类"薄流动性的孪生跟踪基金"
风险，也让回测里那个"每边固定10bp滑点"的假设对剩下的名字更贴近现实（真实
交易量小的名字，滑点大概率比这个假设更差，不是更好）。

**排序也是分钟级的**：`intraday.data.rank_by_intraday_turnover_xtdata` 把每
个交易日内所有分钟K线的 `amount`（成交额）加总成当日成交额、再取跨日中位数
来排名，读的是这条链路本来就要挖的那份分钟数据，**整条 ETF 链路不读任何日线**
（`fetch_intraday_close_panels_xtdata` 只拉分钟 close/suspendFlag；只有涨停
触发版才需要日线收盘价去算涨停价，走的是 `fetch_intraday_panels_xtdata`）。

这个区别不是洁癖：xtdata 的日线和分钟线是**分开缓存**的，本地有 5m 不代表本
地有 1d。第一版 `--top-liquid` 复用了 `leadlag` 那个按日线成交额排序的函数，
结果在只下载过 5m 的 ETF 池上静默地排出 0 只（`kept top 0 of 85`），股票池被
清空后才在后面炸出一个看不出所以然的 IndexError。现在排序读分钟线、排不出来
时直接报错说明原因，`load_panels` 也会在面板为空时直接报错而不是继续往下跑。

跑完会在 `research/output/etf_sweep/`（可用 `--output-dir` 改）下写一份
`sweep_summary.csv`（按 `n_deployable`、`mean_oos_z` 排序）和每个"有样本外
显著 pair 存活"的组合各自的 `pairs_thr<T>_lag<L>.csv`/`.meta.json`。**这一步
本身不能替代回测**：网格搜索复用的是同一套 in-sample/OOS-FDR/hub-filter 门
槛，挑出的组合仍然必须再单独用 `run_etf_backtest.py` 跑一遍（尤其要看零成本
和默认成本两种情况），confirm 毛利方向和是否覆盖得了成本，跟单组参数跑挖掘
之后的流程完全一样——网格搜索只是帮你更快地跑遍参数空间，不改变"统计显著
不等于能赚钱"这条准则。

## 数据量级警告（务必先读）

分钟线数据量比日线大得多：一个完整交易日约 240 根 1 分钟 K 线 / 48 根 5 分钟
K 线。在把这套东西指向全市场、跑好几年历史之前：

- **先缩小股票池**：`--top-liquid 200`~`500`，或只挖同板块龙头股活跃的几个
  行业/概念（`--sector-names`）。
- **先缩短时间范围**：`--start`/`--end` 建议是几个月到一年级别，不是像
  `leadlag`/`limitup` 那样动辄跑 5-8 年。
- `research/run_intraday_mining.py --source xtdata` 的 `--start`/`--end` 默认
  是空字符串（等价于"能下载多少下载多少"），**这是故意的，逼你自己想清楚范围**，
  不是遗漏。

## 关于 K 线索引格式的一个未验证假设

`leadlag.data.panel_from_field_dict`（本模块直接复用）用 `pd.to_datetime` 解析
`get_market_data_ex` 返回的索引，这在真实环境的**日线**数据上已经验证没问题
（"YYYYMMDD"格式）。分钟线的索引大概率是"YYYYMMDDHHMMSS"格式，pandas 的日期
解析器通常也能自动识别，但**这个假设没有在真实 xtdata 分钟线返回值上验证过**。
如果 `fetch_intraday_panels_xtdata` 报解析错误，或者索引长得不对，先打印
`xtdata.get_market_data_ex` 的原始返回值看看，而不是假设是本模块其它地方的问题。

## 快速开始

```bash
# 涨停触发版：合成数据端到端跑通（内置了几组"真实"的盘中触发关系）
python research/run_intraday_mining.py --source synthetic --min-obs 15 \
    --candidate-mode top-n --top-n 40 --output research/output/intraday_pairs.csv
python research/run_intraday_backtest.py --pairs research/output/intraday_pairs.csv --source synthetic

# 急拉触发版：同上，触发换成"5根K线内涨2%"
python research/run_intraday_mining.py --source synthetic --trigger surge \
    --leader-threshold 0.02 --surge-window 5 --min-obs 15 \
    --candidate-mode top-n --top-n 40 --output research/output/surge_pairs.csv
python research/run_intraday_backtest.py --pairs research/output/surge_pairs.csv --source synthetic

# T0 ETF 阈值触发版：合成数据端到端跑通
python research/run_etf_mining.py --source synthetic --min-obs 15 \
    --candidate-mode top-n --top-n 40 --output research/output/etf_pairs.csv
python research/run_etf_backtest.py --pairs research/output/etf_pairs.csv --source synthetic

# 单元测试（两个版本都在这一个文件里）
python -m unittest tests.test_intraday -v
```

接入真实数据跑 T0 ETF 版本（`SYMBOL_LIST` 已经内置在 `run_etf_mining.py` 里，
不需要传股票池参数，也不需要 `--sectors`/`--top-liquid`）：

```bash
python research/run_etf_mining.py --source xtdata --start 20220101 --end 20240601 \
    --output research/output/etf_pairs.csv
python research/run_etf_backtest.py --pairs research/output/etf_pairs.csv --source xtdata \
    --start 20220101 --end 20240601
```

**尚未实现**：`strategy/intraday_strategy.py` 里的实盘触发逻辑是照涨停公式
写死的（`_limit_pct_for_code` + `prev_close`），还不能直接拿去跑 T0 ETF 版本
挖出来的 pairs——真要上实盘，需要照它的结构另写一份用"日内涨幅阈值"当触发
条件的 ContextInfo 脚本。目前 `run_etf_mining.py`/`run_etf_backtest.py` 只
覆盖研究/回测阶段。

## 接入真实数据 / 部署到 QMT（涨停触发版）

```bash
python research/run_intraday_mining.py --source xtdata --top-liquid 300 \
    --start 20240101 --end 20240601 --output research/output/intraday_pairs.csv
```

跑完把 `intraday_pairs.csv` / `intraday_pairs.meta.json` 拷到
`strategy/intraday_strategy.py` 同目录，改好 `ACCOUNT_ID`，**把 QMT 策略编辑器
的主图周期设成和挖掘时一致的分钟周期**（默认 5 分钟），先跑回测再切换模拟盘/
实盘。策略脚本默认关闭了自动下载历史数据（`AUTO_DOWNLOAD_HISTORY = False`），
建议先用 QMT 客户端"数据管理"界面手动下载好标的池的分钟线，原因见上面的数据
量级警告。

### 真要覆盖全市场？按行业分批跑

全市场 + 多年分钟线一次性跑，大概率装不进内存（见上面的数据量级警告）。因为
`--same-sector-only` 默认就是同板块内部配对，按申万一级行业拆成 31 次小任务
跑，不会损失任何统计效力——`research/run_intraday_mining_by_sector.py` 就是
干这个的：

```bash
python research/run_intraday_mining_by_sector.py --start 20200101 \
    --candidate-mode top-n --top-n 30
```

每个行业单独下载、单独挖掘、单独存日志（`research/output/by_sector/logs/`），
某个行业跑失败或超时不影响其它行业继续跑；跑完把所有行业的配对合并成一份，并在
合并后**重新做一次全局的实盘订阅数量上限裁剪**（每个行业单独跑时都各自封顶
500 只，但 31 个行业合起来很容易超过 500，最终喂给 QMT 实盘脚本的应该是合并后
再裁剪过的那一份，不是简单拼接）。中途中断了，加 `--skip-existing` 重跑，
已经跑完的行业不会重新下载。

统计陷阱（多重检验、样本外验证要用真正的 FDR、大盘/板块共振、涨跌停无法成交
等）跟主 README、`limitup/README.md` 完全一样，这里不重复。
