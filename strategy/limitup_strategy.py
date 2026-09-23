# coding:utf-8
"""
涨停触发-板块跟涨策略（QMT ContextInfo 策略脚本）
================================================

思路：某只"龙头股"当日收盘封涨停后，同板块内某只"跟随股"在 `lag` 个交易日后
上涨的概率显著高于其自身基准概率（题材/资金的板块内传导效应）。涨停判定只用
日线数据：涨停价 = 前收盘价 × (1 + 涨跌幅限制)，涨跌幅限制按股票代码前缀推断
（主板/中小板 10%，创业板 300/301 和科创板 688 是 20%，北交所约 30%），不需要
分钟线。领先-跟随关系表由 research/run_limitup_mining.py 离线挖掘产生。

本文件和 strategy/leadlag_strategy.py 结构几乎一样，唯一的区别是"领先股触发"
的判定条件从"当日上涨"换成了"当日封涨停"，其余（打分、建仓、下单、涨停规避、
持仓管理）完全一致，两份策略可以对照着看。

使用步骤
--------
1. 在装有 QMT/MiniQMT 的机器上运行
   `python research/run_limitup_mining.py --source xtdata ...`，产出
   `limitup_pairs.csv` 和 `limitup_pairs.meta.json`。
2. 把这两个文件拷贝到本文件同一目录下（或修改下面的路径常量）。
3. 把 ACCOUNT_ID 改成你自己的资金账号。
4. 整个文件粘贴进 QMT 策略编辑器，选日线周期，先跑回测验证，再切换到模拟盘/实盘。

重要限制（详见 docs/QMT_API_NOTES.md 和 README.md）
--------------------------------------------------
- 涨跌幅限制的推断是按代码前缀的近似规则，忽略了 ST（5%）和新股上市首日（不限）
  两种情况——这两种情况只会让计算出的涨停价偏高，导致个别真实涨停被漏判（假
  阴性），不会把没涨停的日子误判成涨停（假阳性），所以对策略是"漏报"而不是
  "误报"，方向上是安全的，但如果你的候选池里有大量 ST 股，建议在挖掘阶段就用
  `--include-st` 的反义（默认已排除 ST）处理掉。
- `ContextInfo.get_market_data_ex(subscribe=True)` 实时订阅上限 500 只，参见
  leadlag_strategy.py 同样的说明。
- 同样用模块级全局变量 `_HELD_SINCE` 保存建仓日，避免 ContextInfo 属性跨 K 线
  被修改后不保留的问题。
"""

import json
import os

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV_PATH = os.path.join(_THIS_DIR, "limitup_pairs.csv")
PAIRS_META_PATH = os.path.join(_THIS_DIR, "limitup_pairs.meta.json")

ACCOUNT_ID = "your_account_id_here"  # 改成你的资金账号
MAX_POSITIONS = 10                    # 同时持有的最大只数（等权分配总资产）
HOLDING_PERIOD = 1                    # 持有交易日数；1 = 次日开盘买入、下一交易日开盘卖出
AUTO_DOWNLOAD_HISTORY = True          # 首次运行自动补齐历史数据；本地数据已是最新时可关闭

_HELD_SINCE: dict[str, int] = {}


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
    ContextInfo.lag = int(meta["lag"])
    ContextInfo.tolerance = float(meta.get("tolerance", 0.003))

    leaders = sorted(pairs["leader"].unique().tolist())
    followers = sorted(pairs["follower"].unique().tolist())
    ContextInfo.leaders = leaders
    ContextInfo.universe = sorted(set(leaders) | set(followers))

    leader_map: dict[str, list[tuple[str, float]]] = {}
    for row in pairs.itertuples(index=False):
        leader_map.setdefault(row.leader, []).append((row.follower, float(getattr(row, weight_col))))
    ContextInfo.leader_map = leader_map
    ContextInfo.accountid = ACCOUNT_ID

    print(f"[limitup] loaded {len(pairs)} pairs | {len(leaders)} leaders | "
          f"{len(followers)} followers | universe={len(ContextInfo.universe)}")

    if AUTO_DOWNLOAD_HISTORY:
        for i, code in enumerate(ContextInfo.universe, 1):
            try:
                download_history_data(code, "1d", "", "")
            except Exception as exc:  # noqa: BLE001 - keep init resilient to a single bad code
                print(f"[limitup] download_history_data failed for {code}: {exc}")
            if i % 100 == 0 or i == len(ContextInfo.universe):
                print(f"[limitup] history download {i}/{len(ContextInfo.universe)}")

    _HELD_SINCE.clear()


def handlebar(ContextInfo):
    if not ContextInfo.is_last_bar():
        return

    data = ContextInfo.get_market_data_ex(
        ["close", "preClose", "suspendFlag"], ContextInfo.universe,
        period="1d", count=1, subscribe=True,
    )

    triggered_leaders = []
    for code in ContextInfo.leaders:
        df = data.get(code)
        if df is None or df.empty:
            continue
        row = df.iloc[-1]
        pre_close = row.get("preClose")
        if row.get("suspendFlag", 0) == 1 or not pre_close or pd.isna(pre_close):
            continue
        limit_price = round(pre_close * (1 + _limit_pct_for_code(code)), 2)
        if row["close"] >= limit_price - ContextInfo.tolerance:
            triggered_leaders.append(code)

    if not triggered_leaders:
        return

    scores: dict[str, float] = {}
    for leader in triggered_leaders:
        for follower, weight in ContextInfo.leader_map.get(leader, []):
            scores[follower] = scores.get(follower, 0.0) + weight

    positions = get_trade_detail_data(ContextInfo.accountid, "stock", "position")
    held = {f"{p.m_strInstrumentID}.{p.m_strExchangeID}": p for p in positions if p.m_nVolume > 0}

    # ---- 卖出：持有期已满的仓位，按当日最新价卖出 ----
    for code in list(_HELD_SINCE.keys()):
        entry_bar = _HELD_SINCE[code]
        if ContextInfo.barpos - entry_bar < HOLDING_PERIOD:
            continue
        pos = held.get(code)
        if pos and pos.m_nCanUseVolume > 0:
            passorder(24, 1101, ContextInfo.accountid, code, 5, -1, pos.m_nCanUseVolume,
                      "limitup_exit", 0, "limitup_exit", ContextInfo)
        _HELD_SINCE.pop(code, None)

    # ---- 买入：当日得分最高、当前未持有、未停牌未涨停的跟随股 ----
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    free_slots = MAX_POSITIONS - len(held)
    if free_slots <= 0 or not ranked:
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
        df = data.get(follower)
        if df is None or df.empty:
            continue
        row = df.iloc[-1]
        if row.get("suspendFlag", 0) == 1:
            continue
        price = row["close"]
        if not price or price <= 0:
            continue
        pre_close = row.get("preClose")
        if pre_close and not pd.isna(pre_close):
            follower_limit = round(pre_close * (1 + _limit_pct_for_code(follower)), 2)
            if price >= follower_limit - ContextInfo.tolerance:
                continue  # 跟随股自己也已经封涨停了，大概率买不进，跳过
        detail = _get_instrument_detail(ContextInfo, follower) or {}
        up_stop = detail.get("UpStopPrice")
        if up_stop and price >= up_stop * 0.999:
            continue

        vol = int(budget_per_name // (price * 100)) * 100
        if vol <= 0:
            continue
        passorder(23, 1101, ContextInfo.accountid, follower, 5, -1, vol,
                  "limitup_entry", 0, "limitup_entry", ContextInfo)
        _HELD_SINCE[follower] = ContextInfo.barpos
        picked += 1
