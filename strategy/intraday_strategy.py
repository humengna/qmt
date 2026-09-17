# coding:utf-8
"""
盘中涨停触发-同日板块跟涨策略（QMT ContextInfo 策略脚本，5 分钟周期）
====================================================================

思路：某只龙头股当日盘中某一根 K 线首次触及涨停价后，同板块内某只跟随股在之后
第 `lag_bars` 根 K 线（同一交易日内）上涨的概率显著高于其自身基准概率——这是
`limitup_strategy.py`（次日开盘版）的盘中版本，验证的是"同日下午/短时"传导而
不是"隔日"传导。领先-跟随关系表由 research/run_intraday_mining.py 离线挖掘产生。

和 leadlag_strategy.py / limitup_strategy.py 的核心区别
-------------------------------------------------------
1. **本策略必须运行在分钟周期上**（跟挖掘时用的 `--period` 一致，默认 5 分钟），
   而不是日线——`handlebar` 每根分钟 K 线都会检查一次"触发"，而不是每天收盘检查
   一次。
2. **"首次触板"要按天去重**：同一只龙头股同一天只能触发一次（哪怕它整个上午
   都封在涨停板上），用模块级全局变量 `_TRIGGERED_TODAY` 记录每只票"今天是否已
   经触发过"，每根 K 线开始时先检查日期是否变化，变了就清空记录。
3. **持有期是"根数"不是"天数"**：跟随股买入后持有 `lag_bars` 根 K 线就卖出
   （对应挖掘/回测时验证的同一个窗口），不会跨日持仓——如果 `lag_bars` 根之后
   已经是收盘，会在当天最后一根 K 线强制平仓，不留隔夜仓位（这个假设本身就是
   "盘中传导"，没有验证隔夜持有的效果，不应该顺带隔夜暴露风险）。

使用步骤
--------
1. 在装有 QMT/MiniQMT 的机器上运行
   `python research/run_intraday_mining.py --source xtdata --period 5m ...`，
   产出 `intraday_pairs.csv` 和 `intraday_pairs.meta.json`。
2. 拷贝到本文件同一目录，改好 `ACCOUNT_ID`。
3. 整个文件粘贴进 QMT 策略编辑器，**主图周期选成和挖掘时一致的分钟周期**
   （默认 5 分钟），先跑回测，再切换到模拟盘/实盘。

重要限制
--------
- 涨跌幅限制的推断规则、ST 排除的近似之处，跟 limitup_strategy.py 完全一样，
  见该文件和 docs/QMT_API_NOTES.md、intraday/README.md。
- 数据量：分钟线本身数据量远大于日线，实盘运行时的历史数据下载/订阅耗时会明显
  更长，标的池务必控制在合理范围内（配合挖掘阶段的 `--top-liquid`）。
"""

import json
import os

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV_PATH = os.path.join(_THIS_DIR, "intraday_pairs.csv")
PAIRS_META_PATH = os.path.join(_THIS_DIR, "intraday_pairs.meta.json")

ACCOUNT_ID = "your_account_id_here"  # 改成你的资金账号
MAX_POSITIONS = 10                    # 同时持有的最大只数（等权分配总资产）
# 分钟线数据量远大于日线，默认关闭自动下载：建议先用 QMT 客户端的"数据管理"
# 界面，对 pairs 涉及到的标的池手动下载一段有限时间范围的分钟线，再打开这里。
# 开着的话下面会用空字符串区间调用 download_history_data，等价于"下载全部历史
# 分钟线"，标的一多可能非常慢。
AUTO_DOWNLOAD_HISTORY = False

_HELD_SINCE: dict[str, int] = {}          # code -> 建仓时的 ContextInfo.barpos
_TRIGGERED_TODAY: dict[str, str] = {}     # code -> 今天已经触发过的日期字符串(YYYYMMDD)


def _limit_pct_for_code(code: str) -> float:
    stock = code.split(".")[0]
    if stock.startswith(("300", "301", "688")):
        return 0.20
    if stock.startswith(("8", "4", "92")):
        return 0.30
    return 0.10


def _load_pairs():
    pairs = pd.read_csv(PAIRS_CSV_PATH)
    with open(PAIRS_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    weight_col = "oos_z" if "oos_z" in pairs.columns else "z"
    return pairs, meta, weight_col


def _get_instrument_detail(ContextInfo, code):
    if hasattr(ContextInfo, "get_instrument_detail"):
        return ContextInfo.get_instrument_detail(code)
    return ContextInfo.get_instrumentdetail(code)


def init(ContextInfo):
    pairs, meta, weight_col = _load_pairs()
    ContextInfo.pairs = pairs
    ContextInfo.lag_bars = int(meta["lag_bars"])
    ContextInfo.tolerance = float(meta.get("tolerance", 0.003))
    ContextInfo.bar_period = meta.get("period", "5m")

    leaders = sorted(pairs["leader"].unique().tolist())
    followers = sorted(pairs["follower"].unique().tolist())
    ContextInfo.leaders = leaders
    ContextInfo.universe = sorted(set(leaders) | set(followers))

    leader_map: dict[str, list[tuple[str, float]]] = {}
    for row in pairs.itertuples(index=False):
        leader_map.setdefault(row.leader, []).append((row.follower, float(getattr(row, weight_col))))
    ContextInfo.leader_map = leader_map
    ContextInfo.accountid = ACCOUNT_ID

    print(f"[intraday] loaded {len(pairs)} pairs | {len(leaders)} leaders | "
          f"{len(followers)} followers | universe={len(ContextInfo.universe)} | "
          f"period={ContextInfo.bar_period} | lag_bars={ContextInfo.lag_bars}")

    if AUTO_DOWNLOAD_HISTORY:
        for i, code in enumerate(ContextInfo.universe, 1):
            try:
                download_history_data(code, ContextInfo.bar_period, "", "")
            except Exception as exc:  # noqa: BLE001
                print(f"[intraday] download_history_data failed for {code}: {exc}")
            if i % 100 == 0 or i == len(ContextInfo.universe):
                print(f"[intraday] history download {i}/{len(ContextInfo.universe)}")

    _HELD_SINCE.clear()
    _TRIGGERED_TODAY.clear()


def handlebar(ContextInfo):
    if not ContextInfo.is_last_bar():
        return

    bar_datetime = timetag_to_datetime(ContextInfo.get_bar_timetag(ContextInfo.barpos), "%Y-%m-%d %H:%M:%S")
    today_str = bar_datetime[:10].replace("-", "")
    is_last_bar_of_day = _is_last_bar_of_trading_day(ContextInfo, bar_datetime)

    intraday_data = ContextInfo.get_market_data_ex(
        ["close", "suspendFlag"], ContextInfo.universe,
        period=ContextInfo.bar_period, count=1, subscribe=True,
    )
    daily_data = ContextInfo.get_market_data_ex(
        ["close"], ContextInfo.leaders, period="1d", count=1, subscribe=True,
    )

    # ---- 检测今天第一次触板的 leader ----
    triggered_leaders = []
    for code in ContextInfo.leaders:
        if _TRIGGERED_TODAY.get(code) == today_str:
            continue  # 今天已经触发过，跳过（首次触板去重）
        bar_df, daily_df = intraday_data.get(code), daily_data.get(code)
        if bar_df is None or bar_df.empty or daily_df is None or daily_df.empty:
            continue
        row = bar_df.iloc[-1]
        if row.get("suspendFlag", 0) == 1:
            continue
        prev_close = daily_df.iloc[-1]["close"]
        if not prev_close or pd.isna(prev_close):
            continue
        limit_price = round(prev_close * (1 + _limit_pct_for_code(code)), 2)
        if row["close"] >= limit_price - ContextInfo.tolerance:
            triggered_leaders.append(code)
            _TRIGGERED_TODAY[code] = today_str

    scores: dict[str, float] = {}
    for leader in triggered_leaders:
        for follower, weight in ContextInfo.leader_map.get(leader, []):
            scores[follower] = scores.get(follower, 0.0) + weight

    positions = get_trade_detail_data(ContextInfo.accountid, "stock", "position")
    held = {f"{p.m_strInstrumentID}.{p.m_strExchangeID}": p for p in positions if p.m_nVolume > 0}

    # ---- 卖出：持有期已满，或今天最后一根K线强制清仓（不留隔夜仓位） ----
    for code in list(_HELD_SINCE.keys()):
        entry_bar = _HELD_SINCE[code]
        due = (ContextInfo.barpos - entry_bar >= ContextInfo.lag_bars) or is_last_bar_of_day
        if not due:
            continue
        pos = held.get(code)
        if pos and pos.m_nCanUseVolume > 0:
            passorder(24, 1101, ContextInfo.accountid, code, 5, -1, pos.m_nCanUseVolume,
                      "intraday_exit", 0, "intraday_exit", ContextInfo)
        _HELD_SINCE.pop(code, None)

    if not scores:
        return

    # ---- 买入：当日得分最高、当前未持有、未停牌未涨停的跟随股 ----
    # 只有当天剩余K线数还能撑满 lag_bars 时才建仓，跟回测的窗口口径保持一致，
    # 不引入没被验证过的"临收盘缩短持有期"变体。
    if is_last_bar_of_day:
        return

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    free_slots = MAX_POSITIONS - len(held)
    if free_slots <= 0:
        return

    accounts = get_trade_detail_data(ContextInfo.accountid, "stock", "account")
    total_asset = accounts[0].m_dBalance if accounts else 0.0
    budget_per_name = total_asset / MAX_POSITIONS

    picked = 0
    for follower, score in ranked:
        if picked >= free_slots:
            break
        if follower in held or score <= 0:
            continue
        bar_df = intraday_data.get(follower)
        if bar_df is None or bar_df.empty:
            continue
        row = bar_df.iloc[-1]
        if row.get("suspendFlag", 0) == 1:
            continue
        price = row["close"]
        if not price or price <= 0:
            continue
        detail = _get_instrument_detail(ContextInfo, follower) or {}
        up_stop = detail.get("UpStopPrice")
        if up_stop and price >= up_stop * 0.999:
            continue  # 跟随股自己也已经顶到涨停附近，大概率买不进

        vol = int(budget_per_name // (price * 100)) * 100
        if vol <= 0:
            continue
        passorder(23, 1101, ContextInfo.accountid, follower, 5, -1, vol,
                  "intraday_entry", 0, "intraday_entry", ContextInfo)
        _HELD_SINCE[follower] = ContextInfo.barpos
        picked += 1


def _is_last_bar_of_trading_day(ContextInfo, bar_datetime: str) -> bool:
    """当前 K 线是否是当天最后一根（用 A 股 15:00 收盘时间近似判断）。"""
    return bar_datetime[11:16] >= "14:55"
