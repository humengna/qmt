# 涨停触发 - 板块跟涨策略

验证并交易另一个版本的领先-滞后假设："某只龙头股当日收盘封涨停后，同板块内
另一只跟随股在之后第 `lag` 个交易日上涨的概率，显著高于它自身的基准概率"。
这是 `../leadlag/` 的一个变体：把"leader 触发"的定义从"今天上涨"换成了
"今天收盘封涨停"，其余的挖掘/统计/回测机制完全复用 `leadlag` 已经调试好的
核心（矩阵化两比例 z 检验、Benjamini-Hochberg FDR、样本外重新验证、组合回测）。

## 为什么单独建一个项目，而不是改 leadlag 的参数

`leadlag` 那边试过全市场、流动性前 500、同板块限定等好几种配置，"某票涨后
另一票次日大概率涨"这个最朴素的日线版本，全部无法通过真正的样本外 FDR 检验
（详见主 README 的"统计陷阱说明"）。"涨停"是一个信息量远比"上涨"更大、也更
稀有的事件（龙头封板往往伴随题材发酵、资金关注度骤增），是一个方向性不同、
更贴近"游资/题材炒作"直觉的假设，值得独立验证，而不是在同一套参数上继续微调。

## 核心差异：涨停判定不需要分钟线

涨停价 = 前收盘价 × (1 + 涨跌幅限制)，涨跌幅限制按股票代码前缀推断：

| 板块 | 代码前缀 | 涨跌幅限制 |
|---|---|---|
| 主板/中小板 | 其他 | 10%（默认） |
| 创业板 | 300/301 | 20% |
| 科创板 | 688 | 20% |
| 北交所 | 8/4/92 开头（近似） | 30% |

这个规则**忽略了 ST（5%）和新股上市首日（不限）两种情况**——两者都只会让算
出来的涨停价偏高，导致极少数真实涨停被漏判（假阴性），不会把没涨停的日子误
判成涨停（假阳性）。挖掘阶段默认会排除当前处于 ST 状态的股票（见
`limitup/data.py` 的 `build_st_exclusion_set_xtdata`），缓解这个问题，但那也
只用了股票的**当前**名称判断，不追溯历史 ST 状态。

## 目录结构

```
limitup/
  data.py      涨跌幅规则、涨停判定、ST 过滤、xtdata 数据获取、合成数据生成器
  event.py     把 close/preClose/停牌 面板转成"当日是否封涨停"布尔面板
                （唯一真正新增的逻辑，其余全部复用 leadlag.factor 的非对称版本）
research/
  run_limitup_mining.py    离线挖掘 CLI，对照 leadlag 的 run_mining.py
  run_limitup_backtest.py  回测 CLI，对照 leadlag 的 run_backtest.py
strategy/
  limitup_strategy.py      QMT ContextInfo 策略脚本，对照 leadlag_strategy.py
tests/
  test_limitup.py          合成数据测试：涨停判定精确性、挖掘找回注入信号、
                            纯噪声下样本外 FDR 正确拒绝、回测能跑通
```

`leadlag/factor.py` 为了支持这个项目新增了非对称版本
（`compute_pairwise_stats_asymmetric` / `validate_out_of_sample_asymmetric`），
允许 leader 的"触发"信号和 follower 的"结果"信号是两种不同的定义（这里是
"封涨停" vs "普通上涨"），原有的对称版本改成了调用非对称版本的薄封装，行为
不变（`tests/test_factor.py` 的全部用例仍然通过）。

## 快速开始

```bash
# 1. 在合成市场上跑通流程（内置了几组"真实"的涨停触发关系）
python research/run_limitup_mining.py --source synthetic --min-obs 20 \
    --candidate-mode top-n --top-n 30 --output research/output/limitup_pairs.csv

# 2. 回测
python research/run_limitup_backtest.py --pairs research/output/limitup_pairs.csv --source synthetic

# 3. 单元测试
python -m unittest tests.test_limitup -v
```

## 接入真实数据 / 部署到 QMT

跟 `leadlag` 完全一样的流程（同一份 `docs/QMT_API_NOTES.md`、同一套
`--same-sector-only` / `--sector-names` / `--list-sectors` 板块过滤逻辑，
`limitup` 默认就是同板块限定，除非传 `--all-sectors`）：

```bash
python research/run_limitup_mining.py --source xtdata --start 20180101 \
    --output research/output/limitup_pairs.csv
```

跑完把 `limitup_pairs.csv` / `limitup_pairs.meta.json` 拷到
`strategy/limitup_strategy.py` 同目录，改好 `ACCOUNT_ID`，粘贴进 QMT 策略
编辑器跑回测/模拟盘/实盘。

**统计陷阱和执行假设的说明跟主 README 完全一样**（多重检验、样本外验证要用
真正的 FDR 而不是只看符号、大盘/板块共振、涨跌停无法成交等），这里不重复，
务必读完主 README 再决定要不要上真金白银。
