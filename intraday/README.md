# 盘中涨停触发 - 同日板块跟涨策略

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

### 真实数据跑出来的教训：样本外显著 ≠ 扣完成本能赚钱（这次证据更直接）

用户在真实 xtdata 上跑通了一次全流程（85 只 ETF，样本外测试窗口约 8 个月，
5 分钟 K 线），挖出 10 组样本外 FDR 显著的 pairs，回测结果却是全面负收益
（total_return -27%，全部 8 个 follower 无一例外都亏钱）。逐笔拆解
（`etf_trades.csv` 的 `pnl` 列是扣成本后的净值，用买卖价差反推毛收益）发现
两件事，都不是"随机噪声"或"统计方法有bug"：

1. **毛利（不含成本）其实是正的**，方向和挖掘阶段找到的完全一致——不是伪发现，
   也不是方向反了。但毛利太薄：换算成基点，每笔交易平均毛收益只有十几个bp，
   而回测默认成本假设（佣金3bp+滑点10bp每边、卖出再加印花税5bp）一来一回
   要吃掉约 31bp，边际比成本还小，怎么调都补不回来。
2. **少数几个 follower（比如某几只港股/纳指相关的 ETF）反复出现，同时对应
   十几个不同的 leader**——这不像是"A带动B"式的一对一关系，更像是这几个
   follower 本身对全池子共享的因子（隔夜海外市场/商品价格波动）beta 特别高，
   只要池子里随便哪个成分先冲了阈值（往往意味着共同因子已经在动），这几个
   高beta名字随后大概率也会涨。85 只标的本身高度同质（都是跨境/商品/债券
   ETF），`mode='excess'` 用这 85 只自己的截面中位数去剔除共同因子，对这种
   高beta名字剔不干净——被误判成了"对每个 leader 的独立响应"。

针对这两点分别加了诊断/防护手段：

**成本敏感性**：`run_etf_backtest.py` 新增 `--commission-bps`/`--slippage-bps`
/`--stamp-tax-bps`，可以覆盖 `IntradayBacktestConfig` 的默认值，用来验证
"扣掉成本才由正转负"这个判断，而不用改代码：

```bash
python research/run_etf_backtest.py --pairs research/output/etf_pairs.csv --source xtdata \
    --start 20220101 --end 20240601 --commission-bps 0 --slippage-bps 0 --stamp-tax-bps 0
```

**hub follower 排除**：`leadlag.factor.exclude_hub_followers(pairs, max_leaders_per_follower)`
——在样本外 FDR 通过之后、进实盘符号预算裁剪之前，把"对应了超过
`max_leaders_per_follower` 个不同 leader"的 follower 整个剔除（`=0` 关闭这个
过滤）。`run_etf_mining.py` 默认 `--max-leaders-per-follower=3`：

```bash
python research/run_etf_mining.py --source xtdata --start 20220101 --end 20240601 \
    --max-leaders-per-follower 3 --output research/output/etf_pairs.csv
```

这只是一个粗粒度的、单变量的安全阀，不会真的把共同因子从数据里剔除干净——
真要根治，需要在算 follower 的 forward outcome 时先对某个基准做 beta 回归、
只保留残差（比现在改动大得多，暂未实现）。当前这一步只是防止继续把"共同
因子驱动的伪配对"当成真实的逐对领先滞后关系送进实盘。

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
