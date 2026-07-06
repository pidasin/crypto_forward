# -*- coding: utf-8 -*-
"""
每日前向測試訊號腳本 (策略A / CB溢價 x FNG)  — 雲端版
=====================================================
用法: python forward_daily.py
它會:
 1. 抓最新資料, 算出今天8幣的倉位 (多空版 + 現貨版)
 2. 算「模擬金累積損益」(從 FORWARD_START 起, 本金 $10,000)
 3. 算健康度紅綠燈 (90日命中率)
 4. 把今天決策+損益寫進 forward_log.csv (用當根日K日期去重, 一天只記一筆)

一天跑幾次都沒關係 — 去重機制保證同一根日K只記一筆, 多跑純為容錯。
前向測試起始日 FORWARD_START 之後才是真驗證。
"""
import warnings, csv, os, numpy as np, pandas as pd, requests, time
warnings.filterwarnings('ignore')

FORWARD_START = '2026-07-05'
STRATEGY = 'A'                       # 用策略A: (CB多FNG貪+1, CB空FNG貪-0.5, CB多FNG平+0.5, CB空FNG平-1)
COINS = ['BTC','ETH','SOL','LTC','LINK','ADA','DOGE','XLM']
CAPITAL = 10000
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'forward_log.csv')

def fetch_bn(sym, days=400):
    end = int(time.time()*1000); start = end - days*86400*1000
    rows=[]; cur=start
    while cur < end:
        try:
            r = requests.get('https://api.binance.com/api/v3/klines',
                params=dict(symbol=sym, interval='1d', startTime=cur, limit=1000), timeout=20).json()
        except: break
        if not isinstance(r,list) or not r: break
        rows += r; cur = r[-1][0]+1
        if len(r) < 1000: break
    if not rows: return None
    df = pd.DataFrame(rows, columns=list('tohlcv')+['ct','qv','n','tb','tq','ig'])
    df['t'] = pd.to_datetime(df['t'], unit='ms'); df['c'] = df['c'].astype(float)
    return df.drop_duplicates('t').set_index('t')['c']

def fetch_cb(prod, days=400):
    end = int(time.time()); start = end - days*86400
    rows=[]; cur=start
    while cur < end:
        seg = min(cur + 250*86400, end)
        try:
            r = requests.get(f'https://api.exchange.coinbase.com/products/{prod}/candles',
                params=dict(granularity=86400, start=pd.to_datetime(cur,unit='s').isoformat(),
                            end=pd.to_datetime(seg,unit='s').isoformat()), timeout=20).json()
        except: r=None
        if isinstance(r,list): rows += r
        cur = seg; time.sleep(0.1)
    if not rows: return None
    df = pd.DataFrame(rows, columns=['t','l','h','o','c','v']); df['t']=pd.to_datetime(df['t'],unit='s')
    return df.drop_duplicates('t').sort_values('t').set_index('t')['c'].astype(float)

def pos_A(prem, fng):
    """策略A四象限, 回傳當日『目標倉位』序列"""
    cbL = prem > 0; greed = fng > 60
    p = pd.Series(0.0, index=prem.index)
    p[cbL & greed]        = 1.0
    p[(~cbL) & greed]     = -0.5
    p[cbL & (~greed)]     = 0.5
    p[(~cbL) & (~greed)]  = -1.0
    return p

# --- FNG ---
r = requests.get('https://api.alternative.me/fng/?limit=0&format=json', timeout=20).json()
fng = pd.DataFrame(r['data']); fng['value']=fng['value'].astype(int)
fng['date']=pd.to_datetime(fng['timestamp'].astype(int),unit='s').dt.normalize()
fng = fng.sort_values('date').set_index('date')['value']

rets_ls={}; rets_sp={}; signals=[]; hitrates=[]; candle_date=None
for c in COINS:
    bn = fetch_bn(c+'USDT'); cb = fetch_cb(c+'-USD')
    if bn is None or cb is None:
        print(f"  {c}: 資料失敗跳過"); continue
    d = pd.DataFrame({'p':bn}).dropna()
    d['prem'] = ((cb.reindex(d.index, method='ffill')/d['p']-1)*10000).rolling(7).mean()
    d['fng'] = fng.reindex(d.index.normalize()).fillna(50)
    d = d.dropna()
    ret = d['p'].pct_change().fillna(0)
    pos = pos_A(d['prem'], d['fng'])
    pl = pos.shift(1).fillna(0)                       # 昨訊號今執行
    rets_ls[c] = ret*pl - pl.diff().abs().fillna(0)*0.001
    rets_sp[c] = ret*pl.clip(lower=0) - pl.clip(lower=0).diff().abs().fillna(0)*0.001
    latest = d.iloc[-1]
    ls, sp = latest_pos = (pl.iloc[-1] if len(pl) else 0), max(pl.iloc[-1],0)
    # 今天『要設定』的目標倉位 = 最新一根訊號
    tgt = pos.iloc[-1]
    signals.append((c, latest['prem'], latest['fng'], tgt, max(tgt,0)))
    candle_date = d.index[-1]
    # 90日命中率
    hit = (np.sign(pos.shift(1))==np.sign(ret)).astype(float)
    hitrates.append(hit.tail(90).mean()*100)

if not signals:
    print("全部資料抓取失敗, 結束"); raise SystemExit

# --- 模擬金累積損益 (從 FORWARD_START) ---
Rls = pd.DataFrame(rets_ls); Rsp = pd.DataFrame(rets_sp)
nav = (~Rls.isna()).sum(axis=1)
b_ls = Rls.div(nav,axis=0).fillna(0).sum(axis=1)
b_sp = Rsp.div(nav,axis=0).fillna(0).sum(axis=1)
fstart = pd.Timestamp(FORWARD_START)
eq_ls = (1+b_ls[b_ls.index>=fstart]).prod()
eq_sp = (1+b_sp[b_sp.index>=fstart]).prod()
days_fwd = (b_ls.index[-1]-fstart).days if b_ls.index[-1]>=fstart else 0

hr = float(np.mean(hitrates)) if hitrates else float('nan')

# --- 印報告 ---
print('='*60)
print(f"  策略{STRATEGY} 每日訊號  |  日K基準: {candle_date.date()}")
print('='*60)
print(f"\n{'幣':5}{'溢價7d':>9}{'FNG':>5}{'多空倉位':>9}{'現貨倉位':>9}")
print('-'*40)
for c,prem,fv,ls,sp in signals:
    lsd = '滿多' if ls==1 else '半多' if ls==0.5 else '半空' if ls==-0.5 else '滿空'
    spd = '滿多' if sp==1 else '半多' if sp==0.5 else '現金'
    print(f"{c:5}{prem:>8.1f}bp{fv:>5.0f}{ls:>+6.1f} {lsd}{sp:>+6.1f} {spd}")
n=len(signals)
print('-'*40)
print(f"\n【多空版】平均淨曝險 {sum(s[3] for s in signals)/n*100:+.0f}%")
print(f"【現貨版】平均投入 {sum(s[4] for s in signals)/n*100:.0f}% | 現金 {100-sum(s[4] for s in signals)/n*100:.0f}%")

print(f"\n--- 模擬金損益 (本金 ${CAPITAL:,}, 起始 {FORWARD_START}, 已 {days_fwd} 天) ---")
print(f"  多空版: ${CAPITAL*eq_ls:,.0f}  ({(eq_ls-1)*100:+.1f}%)")
print(f"  現貨版: ${CAPITAL*eq_sp:,.0f}  ({(eq_sp-1)*100:+.1f}%)")

light = '🟢 綠燈' if hr>=52 else '🟡 黃燈' if hr>=48 else '🔴 紅燈'
print(f"\n健康度 90日命中率: {hr:.1f}%  {light}  (連續兩月<48%=減碼)")

# --- 寫 log (用日K日期去重) ---
key = str(candle_date.date())
existing=set()
if os.path.exists(LOG):
    with open(LOG, encoding='utf-8-sig') as f:
        for row in csv.reader(f):
            if len(row)>1: existing.add(row[1])
if key in existing:
    print(f"\n(日K {key} 已記錄過, 略過寫入 — 這是正常的去重)")
else:
    newfile = not os.path.exists(LOG)
    with open(LOG,'a',newline='',encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if newfile:
            w.writerow(['執行時間UTC','日K基準','90日命中率','多空版$','現貨版$','幣','溢價7d','FNG','多空倉位','現貨倉位'])
        now = pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M')
        for c,prem,fv,ls,sp in signals:
            w.writerow([now, key, round(hr,1), round(CAPITAL*eq_ls), round(CAPITAL*eq_sp), c, round(prem,1), int(fv), ls, sp])
    print(f"\n✅ 已記錄日K {key} (8幣) 到 forward_log.csv")
