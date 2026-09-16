# QMT 内置 Python API 要点摘录

本文档摘录自仓库自带的 `QMT内置Python_API完整文档.zip`（300 页，"内置Python"
指策略编辑器里 `ContextInfo` 驱动的沙盒环境），只记录 `strategy/leadlag_strategy.py`
实际用到、且容易踩坑的部分，附页码方便回查原文。如果你本地客户端版本的接口
签名跟这里记的不一样，以官方文档/`QMT内置Python_API完整文档.pdf`原文为准。

## 1. 两套相似但不同的 API surface

- **ContextInfo（内置 Python，本仓库文档覆盖的范围）**：只能在 QMT 策略编辑器里跑，
  通过 `init(ContextInfo)` / `handlebar(ContextInfo)` 回调驱动，多数函数是
  `ContextInfo.xxx()` 方法调用。
- **xtquant.xtdata（独立 SDK）**：一个普通的 pip 包/模块，装在有 QMT/MiniQMT
  客户端运行的机器上，可以从任意 Python 进程（终端、venv、Jupyter）里
  `from xtquant import xtdata` 直接用，函数名基本和 ContextInfo 版一一对应
  （去掉 `ContextInfo.` 前缀），但**没有** `subscribe=True` 时 500 只的实时订阅
  上限（那个上限是 ContextInfo 沙盒里实时行情推送机制的限制）。

本项目的取舍：**全市场因子挖掘用 xtquant.xtdata**（`leadlag/data.py` 里的
`fetch_price_panels_xtdata` / `get_full_market_stock_list_xtdata`），因为挖掘阶段
需要遍历几千只股票的历史数据，不应该受 500 只的限制；**实盘/模拟盘策略用
ContextInfo**（`strategy/leadlag_strategy.py`），因为那是策略编辑器唯一支持的
运行方式。挖掘阶段产出的 `pairs.csv` 会被裁剪到 500 只以内的股票池，正好衔接
ContextInfo 侧的限制（见下面第 3 条）。

## 2. `get_market_data_ex` 返回结构（第 106-109 页）

```python
ContextInfo.get_market_data_ex(
    fields=[], stock_code=[], period='follow', start_time='', end_time='',
    count=-1, dividend_type='follow', fill_data=True, subscribe=True)
```

返回 `dict{stock_code: pd.DataFrame}`，每个 DataFrame 的 index 是时间、columns
是你传入的 `fields`（各标的的 DataFrame 维度、索引相同）。`leadlag/data.py` 的
`panel_from_field_dict` 就是把这个结构转成本项目统一用的宽表
（行=交易日，列=股票代码）。

关键参数：
- `subscribe=False`：只读本地已下载的数据，不做实时订阅，**不受数量上限限制**，
  但要求提前 `download_history_data` 过（第 4 页、第 259 页示例）。回测/离线研究
  应该用这个。
- `subscribe=True`（默认）：会做实时订阅，**上限 500 只**（第 260 页），可以拿到
  动态行情。实盘/模拟盘策略应该用这个。

## 3. 500 只订阅上限（第 260 页）

> get_market_data_ex(subscribe=True) 有订阅股票数量限制，即 stock_list 参数的
> 数量不能超过 500

`strategy/leadlag_strategy.py` 的 `handlebar` 用 `subscribe=True` 拉当日行情，
所以它能处理的股票池（leader ∪ follower 去重后）不能超过 500 只。
`leadlag/factor.py` 的 `select_for_deployment(max_unique_symbols=500)` 就是在
挖掘产出阶段把这个约束前移，保证生成的 `pairs.csv` 一定能直接喂给实盘脚本。

## 4. `download_history_data`（第 91 页、第 259 页）

```python
download_history_data(stockcode, period, startTime, endTime)
download_history_data(stockcode, period, startTime, endTime, incrementally=True)
```

模块级函数（不带 `ContextInfo.` 前缀），单只标的、单次调用。批量下载就是循环调用
（官方示例本身也是这么写的，见第 259 页 `my_download` 函数）。

## 5. `ContextInfo` 属性的跨 K 线回滚问题（第 14、92 页）——本项目踩过的坑

> 由于底层机制的限制，ContextInfo 中存储的变量值将会回滚，即在对 ContextInfo
> 中的变量进行修改之后，在下一次 handlebar 调用时，这些修改将不会保留。

意思是：**在 `init()` 里赋值一次、之后在 `handlebar()` 里只读不改**是安全的
（本仓库到处这么用：`ContextInfo.pairs`、`ContextInfo.leader_map` 等）；但如果
在 `handlebar()` 里 `.append()` / 重新赋值某个 ContextInfo 属性，指望它在下一根
K 线还保留，就会踩坑（第 15 页官方原文举的反例正是
`ContextInfo.stock_list.append(...)`）。

`strategy/leadlag_strategy.py` 需要在每天收盘后记录"这只票是哪天建仓的"，属于
会在 `handlebar` 里逐日修改的可变状态，所以改用了**模块级普通全局变量**
`_HELD_SINCE`（第 408-409 页官方原文对"立即生效"类模型状态保存的推荐做法，
本项目认为对默认的延迟 K 线模式一样适用，且更不容易踩上面的回滚坑）。

## 6. `passorder`（综合下单函数，第 167-169 页）

```python
passorder(opType, orderType, accountid, orderCode, prType, price, volume,
          strategyName, quickTrade, userOrderId, ContextInfo)
```

本项目固定用官方示例里买卖股票的组合（第 9-10、268 页多处示例一致）：

- `opType=23` 买入，`opType=24` 卖出
- `orderType=1101`（按股数下单）
- `prType=5`（最新价；`price` 参数此时不生效，填 `-1` 即可）
- `quickTrade=0`（默认的"逐 K 线"延迟模式：收盘生成的信号在下一根 K 线第一个
  tick 到来时才真正下单，第 81 页原文——这正好符合本策略"今天收盘触发信号、
  明天开盘成交"的设计意图，不需要额外处理）

`volume` 的单位由 `orderType` 最后一位决定，`1101` 是按股数，因此要素成 A 股一手
=100 股的整数倍（策略脚本里自己 round 到 100）。

还有一组只能在**回测**里用的简化下单函数
（`order_value` / `order_percent` / `order_target_value`，第 198-201 页，原文
明确标注"以下函数仅回测生效，实盘和模拟盘交易均不可用"）。为了让回测和实盘走
同一套下单逻辑（避免"回测好看、实盘用不了"的落差），本项目**没有用**这组函数，
统一用 `passorder`。

## 7. 持仓/账户查询（`get_trade_detail_data`，第 181-183 页）

```python
get_trade_detail_data(accountID, strAccountType, strDatatype)
```

- `get_trade_detail_data(acc, 'stock', 'position')` → 持仓明细列表，用到的字段：
  `m_strInstrumentID` + `.` + `m_strExchangeID` 拼出标准代码、`m_nVolume`（持仓量）、
  `m_nCanUseVolume`（可用/可卖数量）。
- `get_trade_detail_data(acc, 'stock', 'account')` → 账户对象列表（通常取
  `[0]`），用到 `m_dBalance`（总资产）。

## 8. 合约详情 / 涨跌停价（`get_instrument_detail`，第 148-150 页）

```python
ContextInfo.get_instrument_detail(stockcode, iscomplete=False)
```

返回 dict，含 `UpStopPrice`（当日涨停价）、`DownStopPrice`（跌停价）。策略脚本
用它跳过"现价已经顶到涨停附近，大概率买不进"的标的。旧版客户端函数名是
`get_instrumentdetail`（第 148 页原文提示），脚本里做了兼容判断。

## 9. 全市场股票池（`get_stock_list_in_sector`，第 148 页附近多处示例）

```python
ContextInfo.get_stock_list_in_sector('沪深A股')   # 或 xtdata.get_stock_list_in_sector(...)
```

`research/run_mining.py --source xtdata` 默认板块是 `沪深A股`；北交所股票可以
加 `京市A股`（第 258、268 页示例里出现过 `沪深京A股` 的写法）。
