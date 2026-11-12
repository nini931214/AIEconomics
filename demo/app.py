import os
import json
import sqlite3
import tempfile
import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from email.utils import parsedate_to_datetime
from pathlib import Path
from datetime import datetime, timedelta, timezone
import feedparser
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from fredapi import Fred
from pydantic import BaseModel, Field
from typing import Literal

st.set_page_config(page_title='全球總體經濟與匯率儀表板', page_icon='🌐', layout='wide')
st.title('🌐 全球總體經濟、債券與匯率分析儀表板')
st.caption('FRED + Yahoo Finance + Google News RSS + Gemini｜資料頻率、日期與延遲請依來源確認；非投資建議')


def secret(name):
    try:
        return st.secrets.get(name, os.getenv(name, ''))
    except (FileNotFoundError, KeyError):
        return os.getenv(name, '')

st.sidebar.header('⚙️ 設定')
fred_key = secret('FRED_API_KEY') or st.sidebar.text_input('FRED API Key', type='password')
gemini_key = secret('GEMINI_API_KEY') or st.sidebar.text_input('Gemini API Key（摘要功能）', type='password')
# 快速模式適合 200 字左右的短評；原本的 3.8 Flash 仍可選用。
MODEL_CHOICES = ['gemini-3.5-flash-lite', 'gemini-3.5-flash', 'gemini-3.8-flash']
configured_model = secret('GEMINI_MODEL')
if configured_model and configured_model not in MODEL_CHOICES:
    MODEL_CHOICES.append(configured_model)
gemini_model = st.sidebar.selectbox(
    'Gemini 摘要模型',
    MODEL_CHOICES,
    index=MODEL_CHOICES.index(configured_model) if configured_model else 0,
    format_func=lambda name: {
        'gemini-3.5-flash-lite': '⚡ 快速｜Gemini 3.5 Flash-Lite',
        'gemini-3.5-flash': '🧠 課堂指定｜Gemini 3.5 Flash',
        'gemini-3.8-flash': '🧠 進階｜Gemini 3.8 Flash',
    }.get(name, name),
    help='快速模式較適合短評；模型仍依你的 Gemini API 使用額度及可用性運作。',
)
if st.sidebar.button('🔄 清除快取並重新抓取'):
    st.cache_data.clear()
    st.rerun()

FRED_SERIES = {
    '美國 CPI 指數': 'CPIAUCSL',
    '美國 10 年期公債殖利率': 'DGS10',
    '美國 2 年期公債殖利率': 'DGS2',
    '美國失業率': 'UNRATE',
    '聯邦基金有效利率': 'FEDFUNDS',
    '日本 10 年期公債殖利率': 'IRLTLT01JPM156N',
}
MARKET_TICKERS = {
    'S&P 500': '^GSPC',
    '台灣加權指數': '^TWII',
    '布蘭特原油期貨 (USD/桶)': 'BZ=F',
    '黃金期貨 (USD/盎司)': 'GC=F',
    'USD/TWD（1 美元＝台幣）': 'TWD=X',
    'GBP/USD（1 英鎊＝美元）': 'GBPUSD=X',
    'GBP/TWD（1 英鎊＝台幣）': 'GBPTWD=X',
}

@st.cache_data(ttl=3600, show_spinner=False)
def get_fred_data(api_key):
    fred = Fred(api_key=api_key)
    series = {}
    failures = {}
    for label, code in FRED_SERIES.items():
        try:
            s = fred.get_series(code, observation_start=(datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d'))
            s.index = pd.to_datetime(s.index)
            series[label] = pd.to_numeric(s, errors='coerce')
        except Exception as exc:
            failures[label] = str(exc)
    if not series:
        raise RuntimeError(f'所有 FRED 指標均無法取得：{failures}')
    df = pd.DataFrame(series).sort_index()
    if {'美國 10 年期公債殖利率', '美國 2 年期公債殖利率'} <= set(df.columns):
        # 先以原始觀測日對齊，再計算利差；避免用補值製造假觀測
        df['10Y−2Y 利差（百分點）'] = df['美國 10 年期公債殖利率'] - df['美國 2 年期公債殖利率']
    return df, failures

@st.cache_data(ttl=600, show_spinner=False)
def get_market_data():
    result = {}
    failures = {}
    for label, ticker in MARKET_TICKERS.items():
        try:
            raw = yf.download(ticker, period='2y', interval='1d', auto_adjust=True, progress=False, threads=False)
            if raw.empty:
                raise ValueError('無行情')
            close = raw['Close']
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close.index = pd.to_datetime(close.index).tz_localize(None) if close.index.tz is not None else pd.to_datetime(close.index)
            result[label] = pd.to_numeric(close, errors='coerce')
        except Exception as exc:
            failures[label] = str(exc)
    if not result:
        raise RuntimeError(f'所有市場行情均無法取得：{failures}')
    df = pd.DataFrame(result).sort_index()
    # 若直盤不可用，可由兩個可取得的報價交叉推導；保留資料來源註記
    cross = 'GBP/TWD（1 英鎊＝台幣）'
    usd = 'USD/TWD（1 美元＝台幣）'
    gbpusd = 'GBP/USD（1 英鎊＝美元）'
    if usd in df and gbpusd in df:
        derived = df[usd].ffill(limit=3) * df[gbpusd].ffill(limit=3)
        if cross not in df or df[cross].dropna().empty:
            df[cross] = derived
            failures[cross] = '使用 GBP/USD × USD/TWD 交叉推算（非銀行成交價）'
    return df, failures


# 新聞來源：將原本過長的 AND 查詢，拆成易找到最新報導的主題。
# Google News RSS 非推播；when:7d 是搜尋範圍，不代表每篇文章都會即時收錄。
NEWS_SOURCES = (
    ('台灣匯率', '新台幣 匯率 when:7d', 'zh-TW', 'TW', 'TW:zh-Hant'),
    ('美國央行', '聯準會 利率 when:7d', 'zh-TW', 'TW', 'TW:zh-Hant'),
    ('美國通膨', '美國 通膨 when:7d', 'zh-TW', 'TW', 'TW:zh-Hant'),
    ('英國央行', '英國央行 英鎊 when:7d', 'zh-TW', 'TW', 'TW:zh-Hant'),
    ('Fed News', 'Federal Reserve interest rates when:7d', 'en-US', 'US', 'US:en'),
    ('BoE News', 'Bank of England pound when:7d', 'en-GB', 'GB', 'GB:en'),
)
custom_news_query = st.sidebar.text_input(
    '📰 自訂新聞關鍵字', value='',
    placeholder='例如：台灣 CPI 或央行 利率',
    help='額外追蹤指定關鍵字；會和其他主題一起檢查 Google News RSS。',
).strip()
active_news_sources = list(NEWS_SOURCES)
if custom_news_query:
    query = custom_news_query[:100]
    if 'when:' not in query.lower():
        query += ' when:7d'
    active_news_sources.append(('自訂主題', query, 'zh-TW', 'TW', 'TW:zh-Hant'))
active_news_sources = tuple(active_news_sources)
TW = timezone(timedelta(hours=8))


def parse_news_time(raw):
    try:
        dt = parsedate_to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(TW)
    except (TypeError, ValueError, IndexError):
        return None


def fetch_one_feed(source):
    """逐一查詢 Google News RSS，回傳主題名稱和可供去重排序的文章。"""
    label, query, hl, gl, ceid = source
    url = (
        f'https://news.google.com/rss/search?q={quote(query, safe="")}'
        f'&hl={hl}&gl={gl}&ceid={ceid}'
    )
    req = Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (compatible; AIEconomics/3.1)',
        'Accept': 'application/rss+xml, application/xml;q=0.9, */*;q=0.7',
    })
    with urlopen(req, timeout=8) as response:
        feed = feedparser.parse(response.read())
    if feed.bozo and not feed.entries:
        raise ValueError('RSS 解析失敗')
    results = []
    for entry in feed.entries[:14]:
        title = str(entry.get('title', '')).strip()
        link = str(entry.get('link', '')).strip()
        raw_date = str(entry.get('published', '')).strip()
        dt = parse_news_time(raw_date)
        if not dt and entry.get('published_parsed'):
            try:
                dt = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc).astimezone(TW)
            except (TypeError, ValueError, OverflowError):
                dt = None
        if title and link.startswith(('http://', 'https://')):
            results.append({
                'title': title,
                'url': link,
                'published': dt.strftime('%Y-%m-%d %H:%M') if dt else (raw_date or '未提供日期'),
                'published_ts': dt.timestamp() if dt else 0,
                'topic': label,
            })
    return results


def news_fingerprint(item):
    """避免 Google News 同篇不同 URL 重複出現。"""
    return re.sub(r'\s+-\s+[^-]+$', '', item['title']).strip().casefold() or item['url']


@st.cache_data(ttl=100, show_spinner=False, max_entries=8)
def get_news(sources):
    """實際向 RSS 請求，回傳新聞、檢查時刻及連線狀態。"""
    checked_at = datetime.now(TW).strftime('%Y-%m-%d %H:%M:%S')
    items = []
    successes = 0
    failed_labels = []
    empty_labels = []
    with ThreadPoolExecutor(max_workers=min(len(sources), 6)) as pool:
        futures = [pool.submit(fetch_one_feed, source) for source in sources]
        for source, future in zip(sources, futures):
            try:
                batch = future.result()
                items.extend(batch)
                successes += 1
                if not batch:
                    empty_labels.append(source[0])
            except Exception as exc:
                failed_labels.append(source[0])
                logging.warning('RSS 來源 %s 讀取失敗：%s', source[0], type(exc).__name__)
    unique = []
    seen = set()
    for item in sorted(items, key=lambda x: x['published_ts'], reverse=True):
        key = news_fingerprint(item)
        if key not in seen:
            seen.add(key)
            unique.append(item)
    diagnostics = {
        'success': successes, 'total': len(sources),
        'failed': failed_labels, 'empty': empty_labels,
        'fetched_count': len(unique),
    }
    return unique[:18], checked_at, diagnostics


def last_obs(series):
    s = series.dropna()
    if s.empty:
        return None, None, None
    return float(s.iloc[-1]), pd.Timestamp(s.index[-1]), float(s.iloc[-2]) if len(s) >= 2 else None


def metric(column, label, df, key, suffix='', digits=2):
    if key not in df:
        column.metric(label, '無資料')
        return
    value, date, previous = last_obs(df[key])
    if value is None:
        column.metric(label, '無資料')
        return
    delta = f'{value-previous:+.{digits}f}{suffix}' if previous is not None else None
    column.metric(label, f'{value:,.{digits}f}{suffix}', delta, help=f'最近觀測日：{date.date()}；delta 為前一筆有效觀測差')


def scenario(series, horizon=30):
    # 歷史波動率推算的情境區間，不是具預測能力的信賴區間
    s = series.dropna()
    r = np.log(s / s.shift(1)).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 40:
        return None
    spot = float(s.iloc[-1])
    sigma = float(r.tail(90).std())
    scale = sigma * np.sqrt(horizon)
    return {'spot': spot, 'low': spot * np.exp(-1.96 * scale), 'high': spot * np.exp(1.96 * scale), 'sigma': sigma}

with st.spinner('正在取得最新可用資料…'):
    macro_df, macro_errors = (get_fred_data(fred_key) if fred_key else (pd.DataFrame(), {}))
    try:
        market_df, market_errors = get_market_data()
    except Exception as exc:
        market_df, market_errors = pd.DataFrame(), {'市場行情': str(exc)}

if not fred_key:
    st.info('輸入 FRED API Key 可開啟總經數據與殖利率曲線；市場資料與新聞仍可使用。')

st.subheader('📊 核心經濟指標')
if not macro_df.empty:
    cols = st.columns(4)
    metric(cols[0], '美國 CPI（指數，非通膨率）', macro_df, '美國 CPI 指數', digits=1)
    metric(cols[1], '美國 10 年債殖利率', macro_df, '美國 10 年期公債殖利率', '%')
    metric(cols[2], '美國失業率', macro_df, '美國失業率', '%', 1)
    metric(cols[3], '聯邦基金有效利率', macro_df, '聯邦基金有效利率', '%')
    cols = st.columns(2)
    metric(cols[0], '日本 10 年債殖利率（月資料）', macro_df, '日本 10 年期公債殖利率', '%')
    metric(cols[1], '美債 10Y−2Y 利差', macro_df, '10Y−2Y 利差（百分點）', ' 個百分點')

st.subheader('🚦 美國公債殖利率曲線警示')
spread_key = '10Y−2Y 利差（百分點）'
if spread_key in macro_df and not macro_df[spread_key].dropna().empty:
    spread = macro_df[spread_key].dropna()
    val, obs_date, _ = last_obs(spread)
    if val < 0:
        st.error(f'🔴 Warning：美債殖利率曲線倒掛！10Y−2Y = {val:.2f} 個百分點（{obs_date.date()}）。倒掛是風險訊號，並非衰退保證。')
    else:
        st.success(f'🟢 最近觀測利差為 {val:.2f} 個百分點（{obs_date.date()}），未倒掛。')
    fig = px.line(spread.rename('利差'), title='美國 10 年期 − 2 年期公債殖利率')
    fig.add_hline(y=0, line_dash='dash', line_color='red')
    fig.update_layout(yaxis_title='百分點', xaxis_title='觀測日期', hovermode='x unified')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info('取得 DGS10 與 DGS2 後會顯示倒掛燈號與歷史圖。')

st.subheader('📈 金融市場與總經歷史趨勢')
tab_market, tab_macro = st.tabs(['金融市場（Yahoo Finance）', '總體經濟（FRED）'])
with tab_market:
    if not market_df.empty:
        choice = st.selectbox('市場標的', list(market_df.columns))
        fig = px.line(market_df, y=choice, title=f'{choice}｜最近兩年可用行情')
        fig.update_layout(xaxis_title='日期', yaxis_title='行情數值')
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f'最新觀測日：{market_df[choice].dropna().index[-1].date()}；Yahoo Finance 行情可能延遲。')
    else:
        st.warning('目前無法取得市場行情。')
with tab_macro:
    if not macro_df.empty:
        choice2 = st.selectbox('總體指標', list(macro_df.columns))
        fig = px.line(macro_df[choice2].dropna(), title=f'{choice2}｜歷史觀測值')
        fig.update_layout(xaxis_title='日期', yaxis_title='原始單位')
        st.plotly_chart(fig, use_container_width=True)
        st.caption('各指標有不同發布頻率；未對缺值做跨頻率填補。')
    else:
        st.info('請提供 FRED API Key。')

st.subheader('💱 美元、台幣與英鎊的連動')
USD, GBPUSD, GBPTWD = 'USD/TWD（1 美元＝台幣）', 'GBP/USD（1 英鎊＝美元）', 'GBP/TWD（1 英鎊＝台幣）'
if not market_df.empty:
    cols = st.columns(3)
    metric(cols[0], '1 美元＝新台幣', market_df, USD, ' TWD', 3)
    metric(cols[1], '1 英鎊＝美元', market_df, GBPUSD, ' USD', 4)
    metric(cols[2], '1 英鎊＝新台幣', market_df, GBPTWD, ' TWD', 3)
    if all(x in market_df for x in (USD, GBPUSD, GBPTWD)):
        pair = market_df[[USD, GBPUSD, GBPTWD]].ffill(limit=3).dropna()
        if len(pair) >= 20:
            normalized = pair / pair.iloc[0] * 100
            fig = px.line(normalized, title='三組匯率走勢比較（共同起點＝100）')
            fig.update_layout(yaxis_title='指數（起點＝100）', xaxis_title='日期')
            st.plotly_chart(fig, use_container_width=True)
            daily_changes = pair.pct_change(fill_method=None).dropna() * 100
            if len(daily_changes) >= 20:
                corr = daily_changes.corr()
                st.plotly_chart(px.imshow(corr, text_auto='.2f', zmin=-1, zmax=1, color_continuous_scale='RdBu_r', title='每日匯率變動相關係數'), use_container_width=True)
            st.markdown('**匯率恆等式：GBP/TWD ≈ GBP/USD × USD/TWD。** 美元對台幣升值可能推高英鎊換匯成本，但若英鎊同時對美元貶值，兩者可能抵銷。相關性不是因果關係。')
            st.caption('這是市場參考行情，不是台新銀行即期賣出價；實際購匯成本需另查銀行牌告。')
            sc = scenario(pair[GBPTWD])
            if sc:
                st.markdown(f"**30 個交易日歷史波動情境（非價格預測）：** 假設價格不變且日報酬波動延續，約為 **{sc['low']:.2f}–{sc['high']:.2f} TWD/GBP**；基準報價 {sc['spot']:.2f}。")
                st.caption('採最近 90 筆日對數報酬標準差 × √30 與 ±1.96 倍波動幅度；未納入央行決策、突發新聞、趨勢漂移及極端風險，不能當作 95% 實際預測保證。')


@st.fragment(run_every='120s')
def show_live_news():
    st.subheader('📰 最新財經時事與央行動向')
    st.caption('網站開啟期間每 2 分鐘檢查 RSS；可按按鈕強制重新查詢，不用重新整理整個頁面。')
    manual_check = st.button('🔄 立即檢查新聞', key='refresh_news', use_container_width=False)
    if manual_check:
        get_news.clear()  # 強制跳過 100 秒新聞快取，保留 FRED / Yahoo 的快取。
    if manual_check:
        with st.spinner('正在向新聞來源重新查詢，並比對是否有新標題…'):
            fetched, checked_at, diag = get_news(active_news_sources)
    else:
        fetched, checked_at, diag = get_news(active_news_sources)

    old_news = st.session_state.get('latest_news', [])
    old_keys = {news_fingerprint(item) for item in old_news}
    news = fetched
    if fetched:
        st.session_state['latest_news'] = fetched
        st.session_state['last_news_success_time'] = checked_at
    elif old_news:
        news = old_news  # 短暫斷線顯示上次成功取得的快照。
    new_count = len([item for item in fetched if news_fingerprint(item) not in old_keys]) if old_news else 0

    if manual_check:
        if diag['success'] == 0:
            st.error(f'檢查失敗：{diag["total"]} 個來源均無法連線。顯示的是上次成功抓取的新聞，請查看下方來源狀態。')
        elif new_count:
            st.success(f'已完成即時檢查！找到 {new_count} 則與上次顯示不同的新聞。')
        else:
            st.info('已重新連線檢查完成，但目前 RSS 提供的文章與上次顯示相同，並非按鈕沒有反應。')

    st.caption(
        f'最近實際檢查：{checked_at}（台灣時間）｜'
        f'成功來源 {diag["success"]}/{diag["total"]}｜'
        f'取得 {diag["fetched_count"]} 則不重複文章（最多顯示 18 則）'
    )
    if diag['failed']:
        st.warning('本次無法連線的來源：' + '、'.join(diag['failed']))
    if diag['success'] == 0 and news:
        st.warning('以下是暫存的舊新聞；最近一次成功檢查：' + st.session_state.get('last_news_success_time', '不明'))

    if news:
        newest = next((x for x in news if x.get('published_ts')), None)
        if newest:
            st.caption(f'目前最新文章發布時間：{newest["published"]}（台灣時間）；這與「檢查時間」不同。')
        for item in news:
            label = '台灣時間' if item.get('published_ts') else '來源日期未確認'
            st.markdown(
                f'- **[{item["title"]}]({item["url"]})**  ·  '
                f'{item["published"]}（{label}） · {item.get("topic", "新聞來源")}'
            )
    else:
        st.warning('目前沒有抓到新聞。可能是來源未回應、近期沒有符合條件的文章，或 Google News RSS 尚未收錄。')
    st.caption('Google News RSS 並非真正的即時新聞推播；新文章是否出現，取決於新聞發布及 Google 收錄時間。')


show_live_news()

# SQLite 僅保存在 Streamlit 目前執行環境；重新部署可能遺失。
DB_PATH = Path(tempfile.gettempdir()) / 'aieconomics_gemini_history.sqlite3'


def init_history():
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            summary TEXT NOT NULL
        )''')


def save_history(model, summary):
    created_at = datetime.now(timezone(timedelta(hours=8))).strftime('%Y-%m-%d %H:%M:%S %z')
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute('INSERT INTO summaries (created_at, model, summary) VALUES (?, ?, ?)',
                     (created_at, model, summary))
    return created_at


def load_history():
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        return conn.execute(
            'SELECT id, created_at, model, summary FROM summaries ORDER BY id DESC'
        ).fetchall()


def delete_history(record_id):
    with sqlite3.connect(DB_PATH, timeout=10) as conn:
        conn.execute('DELETE FROM summaries WHERE id = ?', (record_id,))


class NewsImpact(BaseModel):
    id: int = Field(description='對應所提供新聞的編號，從 1 開始')
    stance: Literal['鷹派', '鴿派', '中性', '無法判定'] = Field(description='從標題對貨幣政策立場的判讀')
    impact: str = Field(description='根據標題做的 30~60 字可能影響分析，需清楚表達不確定性')
    currency_note: str = Field(description='對美元/台幣或英鎊/台幣的可能方向；不能判斷則明說')


class MarketAnalysis(BaseModel):
    sentiment_score: int = Field(ge=0, le=100, description='市場風險情緒，0 代表風險趨避，50 中性，100 風險偏好')
    hawkish_score: int = Field(ge=0, le=100, description='政策鷹派程度，0 極鴿派，50 中性，100 極鷹派')
    sentiment_reason: str = Field(description='解釋兩項分數評分根據與主觀性，40~80 字')
    overall_summary: str = Field(description='150~220 字的總體經濟短評，依日期與指標做謹慎推論')
    news_impacts: list[NewsImpact] = Field(description='對應提供之每一則新聞的標題分析')


@st.cache_data(ttl=900, max_entries=64, show_spinner=False)
def generate_summary_cached(prompt, model, key_fingerprint, _api_key):
    """固定結構化回傳、有限次 SDK retry；相同輸入 15 分鐘重用。"""
    from google import genai
    from google.genai import errors, types

    client = genai.Client(
        api_key=_api_key,
        http_options=types.HttpOptions(
            timeout=20000,
            retry_options=types.HttpRetryOptions(
                attempts=2, initial_delay=1.0, max_delay=2.0, jitter=0.5,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )

    def invoke(which_model):
        options = {
            'response_mime_type': 'application/json',
            'response_schema': MarketAnalysis,
            'max_output_tokens': 2200,
            'temperature': 0.2,
        }
        if which_model.startswith('gemini-3.'):
            options['thinking_config'] = types.ThinkingConfig(
                thinking_level='minimal' if 'flash-lite' in which_model else 'low'
            )
        response = client.models.generate_content(
            model=which_model, contents=prompt,
            config=types.GenerateContentConfig(**options),
        )
        if not (response.text or '').strip():
            raise ValueError('Gemini 未回傳可解析的內容。')
        # 透過 schema + Pydantic 第二次驗證，避免沒有評分值的結果進入 UI。
        return MarketAnalysis.model_validate_json(response.text).model_dump()

    chosen = model
    try:
        try:
            analysis = invoke(chosen)
        except errors.APIError as exc:
            # 配額 429 不用另一個模型再浪費請求；可用性故障才切換備援。
            if chosen == 'gemini-3.5-flash-lite' and getattr(exc, 'code', None) in (404, 500, 502, 503, 504):
                chosen = 'gemini-3.5-flash'
                analysis = invoke(chosen)
            else:
                raise
        generated_at = datetime.now(TW).strftime('%Y-%m-%d %H:%M:%S')
        return analysis, chosen, generated_at
    finally:
        client.close()


def make_summary_prompt(macro_df, market_df, news):
    context = []
    for key in ['美國 CPI 指數', '美國失業率', '美國 10 年期公債殖利率',
                '美國 2 年期公債殖利率', '聯邦基金有效利率',
                '日本 10 年期公債殖利率', spread_key]:
        if key in macro_df:
            value, dt, _ = last_obs(macro_df[key])
            if value is not None:
                context.append(f'{key}: {value:.3f}（觀測日 {dt.date()}）')
    for key in [USD, GBPUSD, GBPTWD]:
        if key in market_df:
            value, dt, _ = last_obs(market_df[key])
            if value is not None:
                context.append(f'{key}: {value:.4f}（行情日 {dt.date()}）')
    for idx, item in enumerate(news[:5], start=1):
        context.append(f'新聞 #{idx}（標題，未查證全文）：{item["title"]}；發布日期：{item["published"]}')
    return (
        '你是嚴謹的總體經濟學新聞分析員。請只根據下方提供的數值、觀測日期和新聞標題，'
        '以繁體中文回傳符合要求的 JSON 欄位。'
        'sentiment_score 是市場風險情緒（0 風險趨避／50 中性／100 風險偏好）；'
        'hawkish_score 是貨幣政策立場（0 鴿派／50 中性／100 鷹派）；'
        '兩者絕對不可混為一談，且分數是模型對有限標題的主觀估計、不是統計市場指數。'
        '每則新聞依序標示 id，根據標題分析 stance、impact 與 currency_note。'
        '非央行或政策消息請標示中性或無法判定，不能亂判成鷹派、鴿派。'
        '如有 5 則標題，news_impacts 應列出 5 則，不能新增不存在的新聞。'
        '不得杜撰新聞正文、政策決策、數據或因果；標題不等於已驗證新聞內容。'
        'CPI 指數不是年通膨率；日本殖利率是月資料，不是即時觀測。'
        '不必對匯率給出必然漲跌結論，若資訊不足，應明確說明。'
        'overall_summary 請說明利率、殖利率、通膨、就業、以及 USD/TWD 和 GBP/TWD 可能傳導，'
        '以用於經濟學課程報告。請勿輸出 JSON 之外的文字。'
        '\n\n資料：\n' + '\n'.join(context)
    )


def analysis_as_markdown(result, news):
    """保存到原有 SQLite 的完整文字；保留新聞標題來源和 AI 判讀。"""
    lines = [
        f'**市場風險情緒：{result["sentiment_score"]}/100**（0 風險趨避；100 風險偏好）',
        f'**政策鷹派程度：{result["hawkish_score"]}/100**（0 鴿派；100 鷹派）',
        result['sentiment_reason'], '', result['overall_summary'], '',
        '### 逐條新聞影響判讀',
    ]
    for impact in result['news_impacts']:
        idx = impact['id'] - 1
        if not (0 <= idx < min(len(news), 5)):
            continue
        item = news[idx]
        lines += [
            f'**新聞 {idx+1}｜[{item["title"]}]({item["url"]})**｜{impact["stance"]}',
            impact['impact'],
            '匯率觀察：' + impact['currency_note'], '',
        ]
    lines.append('註：依新聞標題與已知數據分析，非實際全文查證、實測情緒指數或投資建議。')
    return '\n\n'.join(lines)


@st.fragment
def show_ai_panel():
    st.subheader('🤖 Gemini AI 市場情緒與新聞衝擊分析')
    st.caption('新聞每 2 分鐘更新；AI 僅按鈕觸發。對最新 5 則新聞進行鷹鴿派標籤化、0~100 雙指標及逐則分析。')
    st.caption('情緒分數是以目前所選標題與可用經濟數據作出的模型估計，並非真實交易量或市場調查。')

    try:
        init_history()
        history_available = True
    except (OSError, sqlite3.Error) as exc:
        history_available = False
        st.warning(f'目前無法使用歷史紀錄儲存：{exc}')

    if st.button('✨ 產生 AI 市場情緒分析', type='primary', key='generate_summary'):
        if not gemini_key:
            st.warning('請先在 Streamlit Secrets 設定 GEMINI_API_KEY。')
        else:
            latest_news = st.session_state.get('latest_news', [])
            if not latest_news:
                latest_news, _, _ = get_news(active_news_sources)
            if not latest_news and macro_df.empty:
                st.warning('新聞與 FRED 總經資料目前都不可用，請稍後重試。')
            else:
                sampled_news = latest_news[:5]
                prompt = make_summary_prompt(macro_df, market_df, sampled_news)
                signature = hashlib.sha256((gemini_model + '\n' + prompt).encode('utf-8')).hexdigest()
                try:
                    with st.spinner('Gemini 正在分析新聞標題、政策立場與匯率影響…'):
                        fingerprint = hashlib.sha256(gemini_key.encode('utf-8')).hexdigest()
                        result, used_model, generated_at = generate_summary_cached(
                            prompt, gemini_model, fingerprint, gemini_key
                        )
                    output = analysis_as_markdown(result, sampled_news)
                    st.session_state['recent_analysis'] = (result, sampled_news, used_model, generated_at)
                    st.session_state['recent_summary'] = (output, used_model, generated_at)
                    st.success('分析完成！')
                    if history_available and st.session_state.get('last_saved_signature') != signature:
                        try:
                            save_history(used_model, output)
                            st.session_state['last_saved_signature'] = signature
                        except (OSError, sqlite3.Error) as exc:
                            st.warning(f'分析完成，但無法寫入歷史紀錄：{exc}')
                except Exception as exc:
                    code = getattr(exc, 'code', None)
                    if code == 429:
                        st.error('Gemini API 已達使用額度／速率限制；請確認 Google AI Studio 的配額。')
                    elif code in (500, 502, 503, 504):
                        st.error('Gemini 服務暫時繁忙；請稍後再試。')
                    elif code in (401, 403):
                        st.error('Gemini API Key 無效或沒有權限使用選定模型。')
                    elif code == 404:
                        st.error('選定的 Gemini 模型不支援或未對這組 API Key 開放。')
                    else:
                        logging.warning('Gemini 分析失敗：%s', type(exc).__name__)
                        st.error(f'分析失敗（{type(exc).__name__}）。請檢查 SDK、回傳 JSON 及部署日誌。')
                    if 'recent_analysis' in st.session_state:
                        st.info('保留上一次成功的分析結果。')

    if 'recent_analysis' in st.session_state:
        analysis, articles, used_model, generated_at = st.session_state['recent_analysis']
        col1, col2 = st.columns(2)
        col1.metric('市場風險情緒', f'{analysis["sentiment_score"]} / 100')
        col2.metric('政策鷹派程度', f'{analysis["hawkish_score"]} / 100')
        st.progress(analysis['sentiment_score'], text='風險情緒：0 風險趨避 → 50 中性 → 100 風險偏好')
        st.progress(analysis['hawkish_score'], text='政策立場：0 鴿派 → 50 中性 → 100 鷹派')
        st.caption(analysis['sentiment_reason'])
        st.markdown('#### 總體經濟短評')
        st.markdown(analysis['overall_summary'])
        st.markdown('#### 逐條新聞衝擊剖析')
        shown = set()
        for item in analysis['news_impacts']:
            idx = item['id'] - 1
            if idx in shown or not (0 <= idx < len(articles)):
                continue
            shown.add(idx)
            article = articles[idx]
            with st.expander(f'#{idx+1} {article["title"]}｜{item["stance"]}', expanded=(idx == 0)):
                st.markdown(f'**[查看新聞原始連結]({article["url"]})**｜{article["published"]}')
                st.write(item['impact'])
                st.caption('匯率可能影響：' + item['currency_note'])
        if len(shown) < len(articles):
            st.warning('這次模型未對全部標題完成逐條分析，請重新產生一次。')
        st.caption(f'分析生成時間：{generated_at}（台灣時間）｜模型：{used_model}。')
        st.caption('分數與標籤皆屬 AI 主觀研判，不代表新聞全文已核實、央行立場認證或可交易的價格預測。')
        st.download_button('⬇️ 下載本次結構化結果（JSON）',
                           data=json.dumps(analysis, ensure_ascii=False, indent=2),
                           file_name='economic_sentiment_analysis.json',
                           mime='application/json', key='download_analysis_json')

    st.subheader('🗂️ AI 分析歷史紀錄')
    st.caption('沿用原本 SQLite，執行環境重建時紀錄可能遺失，建議下載備份。')
    if history_available:
        try:
            records = load_history()
            if not records:
                st.info('目前尚無分析紀錄。')
            else:
                st.write(f'目前有 {len(records)} 筆紀錄')
                export = [dict(id=r[0], created_at=r[1], model=r[2], summary=r[3]) for r in records]
                st.download_button('⬇️ 下載所有紀錄（JSON）',
                                   data=json.dumps(export, ensure_ascii=False, indent=2).encode('utf-8'),
                                   file_name='gemini_history.json', mime='application/json')
                for record_id, created_at, model, summary in records:
                    with st.expander(f'📅 {created_at}｜{model}｜紀錄 #{record_id}'):
                        st.markdown(summary)
                        st.download_button('下載此紀錄（TXT）', data=summary,
                                           file_name=f'gemini_summary_{record_id}.txt',
                                           mime='text/plain', key=f'download_{record_id}')
                        if st.button('🗑️ 刪除此筆紀錄', key=f'delete_{record_id}'):
                            delete_history(record_id)
                            st.rerun(scope='fragment')
        except (OSError, sqlite3.Error) as exc:
            st.warning(f'讀取歷史紀錄失敗：{exc}')


show_ai_panel()

with st.expander('資料來源、缺漏與更新頻率'):
    st.markdown('FRED：美國 CPI／失業率／有效聯邦基金利率通常為月資料；DGS10、DGS2 通常為日資料；日本 10 年公債系列 IRLTLT01JPM156N 為月資料。Yahoo Finance 為市場參考行情，可能延遲或缺漏。新聞採用 feedparser 讀取 Google News RSS，開啟頁面時每 2 分鐘檢查一次，手動按鈕可強制跳過快取。搜尋條件為近 7 天，並非即時推播。')
    if macro_errors:
        st.warning('FRED 部分系列失敗：' + '；'.join(macro_errors.keys()))
    if market_errors:
        st.warning('市場資料備註：' + '；'.join(f'{k}: {v}' for k, v in market_errors.items()))
    st.markdown('[FRED](https://fred.stlouisfed.org/) · [Yahoo Finance](https://finance.yahoo.com/) · [台新銀行](https://www.taishinbank.com.tw/)')
