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
                跨日边界感知的"从 t 到 t+lag_bars 是否上涨"前瞻指标构造、
                分钟级合成数据生成器
  event.py     把上面这些拼成 build_leader_frames / build_follower_frames
  backtest.py  盘中回测引擎：触发即建仓、持有恰好 lag_bars 根K线、
                当天最后一根K线强制清仓（不留隔夜仓位）
research/
  run_intraday_mining.py    离线挖掘 CLI
  run_intraday_backtest.py  回测 CLI
strategy/
  intraday_strategy.py      QMT ContextInfo 策略脚本，运行在分钟周期上
tests/
  test_intraday.py          合成数据测试：涨停价公式精确匹配、每天每票最多
                            触发一次、前瞻窗口不跨日、挖掘找回注入信号、
                            纯噪声样本外FDR拒绝、回测端到端跑通+验证退出
                            调度不跨日
```

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
# 合成数据端到端跑通（内置了几组"真实"的盘中触发关系）
python research/run_intraday_mining.py --source synthetic --min-obs 15 \
    --candidate-mode top-n --top-n 40 --output research/output/intraday_pairs.csv
python research/run_intraday_backtest.py --pairs research/output/intraday_pairs.csv --source synthetic

# 单元测试
python -m unittest tests.test_intraday -v
```

## 接入真实数据 / 部署到 QMT

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

统计陷阱（多重检验、样本外验证要用真正的 FDR、大盘/板块共振、涨跌停无法成交
等）跟主 README、`limitup/README.md` 完全一样，这里不重复。
