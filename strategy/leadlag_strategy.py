# coding:utf-8
"""
领先-滞后相关性因子策略（QMT ContextInfo 策略脚本）
================================================

思路：某只"领先股"当日收盘相对市场走强（或绝对上涨）后，某只"跟随股"在
`lag` 个交易日后上涨的概率显著高于其自身基准概率。这个领先-跟随关系表
（leader, follower, 权重等）由 research/run_mining.py 离线挖掘产生，
本文件只负责在每个交易日收盘后读取当日谁是"领先股触发"，聚合出跟随股
打分，并对得分最高的若干只跟随股做等权买入、持有期满后卖出。

使用步骤
--------
1. 在一台装有 QMT/MiniQMT 且有本地历史行情缓存的机器上，运行
   `python research/run_mining.py --source xtdata ...` 挖掘出
   `pairs.csv` 和 `pairs.meta.json`（全市场挖掘可能要几分钟到几十分钟，
   具体取决于股票池大小，参见 README 的性能说明）。
2. 把这两个文件拷贝到本文件同一目录下（或修改下面 PAIRS_CSV_PATH /
   PAIRS_META_PATH 指向绝对路径）。
3. 把 ACCOUNT_ID 改成你自己的资金账号。
4. 将本文件整个粘贴进 QMT 策略编辑器，选择"日线"周期，先跑"回测"验证
   效果（回测参数里设置好起止时间、初始资金、手续费），确认没问题后再
   切换到模拟盘/实盘运行。

重要限制（详见 docs/QMT_API_NOTES.md）
--------------------------------------
- `ContextInfo.get_market_data_ex(subscribe=True)` 的实时订阅标的数量上限为
  500 只，所以 pairs.csv 去重后的股票代码总数不应超过 500
  （离线挖掘脚本的 select_for_deployment 已经按这个上限做了裁剪）。
- ContextInfo 的属性只要不在 handlebar 里被"修改"就能跨 K 线安全保存
  （本文件在 init 中一次性写入 pairs/leader_map 等静态配置，之后只读不改，
  是官方文档推荐的用法）；但持仓的"建仓日"这类需要在 handlebar 里逐日
  更新的可变状态，改用模块级全局变量 `_HELD_SINCE` 保存，这也是官方文档
  对"立即生效"类状态的推荐做法，避免 ContextInfo 属性的跨 K 线回滚问题。
- 若在 QMT 编辑器中出现中文注释乱码，改用 `# coding:gbk` 并将本文件另存为
  GBK 编码即可，不影响逻辑。
"""

import json
import os

import pandas as pd

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV_PATH = os.path.join(_THIS_DIR, "pairs.csv")
PAIRS_META_PATH = os.path.join(_THIS_DIR, "pairs.meta.json")

ACCOUNT_ID = "your_account_id_here"  # 改成你的资金账号
MAX_POSITIONS = 10                    # 同时持有的最大只数（等权分配总资产）
HOLDING_PERIOD = 1                    # 持有交易日数；1 = 次日开盘买入、下一交易日开盘卖出
AUTO_DOWNLOAD_HISTORY = True          # 首次运行自动补齐历史数据；本地数据已是最新时可关闭以加快启动

# 建仓日记录用普通全局变量保存（不用 ContextInfo 属性），避免 ContextInfo
# 属性在 handlebar 内被修改后不能跨 K 线保留的问题。
_HELD_SINCE: dict[str, int] = {}


def _load_pairs():
    pairs = pd.read_csv(PAIRS_CSV_PATH)
    with open(PAIRS_META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    weight_col = "oos_z" if "oos_z" in pairs.columns else "z"
    return pairs, meta, weight_col


def _get_instrument_detail(ContextInfo, code):
    # 新版客户端函数名为 get_instrument_detail；旧版为 get_instrumentdetail。
    if hasattr(ContextInfo, "get_instrument_detail"):
        return ContextInfo.get_instrument_detail(code)
    return ContextInfo.get_instrumentdetail(code)


def init(ContextInfo):
    pairs, meta, weight_col = _load_pairs()
    ContextInfo.pairs = pairs
    ContextInfo.mode = meta["mode"]          # 'absolute' 或 'excess'
    ContextInfo.threshold = float(meta["threshold"])
    ContextInfo.lag = int(meta["lag"])

    leaders = sorted(pairs["leader"].unique().tolist())
    followers = sorted(pairs["follower"].unique().tolist())
    ContextInfo.leaders = leaders
    ContextInfo.universe = sorted(set(leaders) | set(followers))

    leader_map: dict[str, list[tuple[str, float]]] = {}
    for row in pairs.itertuples(index=False):
        leader_map.setdefault(row.leader, []).append((row.follower, float(getattr(row, weight_col))))
    ContextInfo.leader_map = leader_map
    ContextInfo.accountid = ACCOUNT_ID

    print(f"[leadlag] loaded {len(pairs)} pairs | {len(leaders)} leaders | "
          f"{len(followers)} followers | universe={len(ContextInfo.universe)} | mode={ContextInfo.mode}")

    if AUTO_DOWNLOAD_HISTORY:
        for i, code in enumerate(ContextInfo.universe, 1):
            try:
                download_history_data(code, "1d", "", "")
            except Exception as exc:  # noqa: BLE001 - keep init resilient to a single bad code
                print(f"[leadlag] download_history_data failed for {code}: {exc}")
            if i % 100 == 0 or i == len(ContextInfo.universe):
                print(f"[leadlag] history download {i}/{len(ContextInfo.universe)}")

    _HELD_SINCE.clear()


def handlebar(ContextInfo):
    if not ContextInfo.is_last_bar():
        return

    data = ContextInfo.get_market_data_ex(
        ["close", "preClose", "suspendFlag"], ContextInfo.universe,
        period="1d", count=1, subscribe=True,
    )

    rets = {}
    for code in ContextInfo.leaders:
        df = data.get(code)
        if df is None or df.empty:
            continue
        row = df.iloc[-1]
        pre_close = row.get("preClose")
        if row.get("suspendFlag", 0) == 1 or not pre_close or pd.isna(pre_close):
            continue
        rets[code] = row["close"] / pre_close - 1.0

    if not rets:
        return

    if ContextInfo.mode == "excess":
        market_ret = pd.Series(rets).median()
        signal = {code: r - market_ret for code, r in rets.items()}
    else:
        signal = rets

    triggered_leaders = [code for code, s in signal.items() if s > ContextInfo.threshold]

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
                      "leadlag_exit", 0, "leadlag_exit", ContextInfo)
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
        detail = _get_instrument_detail(ContextInfo, follower) or {}
        up_stop = detail.get("UpStopPrice")
        if up_stop and price >= up_stop * 0.999:
            continue  # 涨停附近大概率买不进，跳过

        vol = int(budget_per_name // (price * 100)) * 100
        if vol <= 0:
            continue
        passorder(23, 1101, ContextInfo.accountid, follower, 5, -1, vol,
                  "leadlag_entry", 0, "leadlag_entry", ContextInfo)
        _HELD_SINCE[follower] = ContextInfo.barpos
        picked += 1
