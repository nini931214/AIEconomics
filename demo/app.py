import os
import json
import sqlite3
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta, timezone
import xml.etree.ElementTree as ET
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from fredapi import Fred

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
gemini_model = secret('GEMINI_MODEL') or 'gemini-3.8-flash'
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

@st.cache_data(ttl=1800, show_spinner=False)
def get_news():
    # Google News RSS 為公開新聞標題；無法保證涵蓋所有重大事件
    queries = ['英鎊 台幣 美元 匯率 央行', 'Bank of England Federal Reserve Taiwan dollar exchange rate']
    items = []
    for query in queries:
        try:
            url = f'https://news.google.com/rss/search?q={quote(query)}&hl=zh-TW&gl=TW&ceid=TW:zh-Hant'
            req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urlopen(req, timeout=8) as response:
                root = ET.fromstring(response.read())
            for item in root.findall('./channel/item')[:6]:
                title = item.findtext('title', '').strip()
                link = item.findtext('link', '').strip()
                date = item.findtext('pubDate', '').strip()
                if title and link:
                    items.append({'title': title, 'url': link, 'published': date})
        except Exception:
            continue
    seen = set()
    unique = []
    for item in items:
        if item['url'] not in seen:
            unique.append(item)
            seen.add(item['url'])
    return unique[:10]


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

st.subheader('📰 每日財經時事與央行動向')
news = get_news()
if news:
    for item in news:
        st.markdown(f"- [{item['title']}]({item['url']})  ·  {item['published']}")
    st.caption('Google News RSS 聚合標題；新聞日期與來源請點開核實。新聞不代表已確認的市場因果解釋。')
else:
    st.warning('目前無法讀取新聞 RSS；請稍後刷新。')

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


def call_gemini_with_retry(api_key, model, prompt, max_attempts=3):
    from google import genai
    from google.genai import errors
    client = genai.Client(api_key=api_key)
    status = st.empty()
    try:
        for attempt in range(max_attempts):
            try:
                status.info(f'Gemini 正在生成摘要（第 {attempt + 1}/{max_attempts} 次）…')
                response = client.models.generate_content(model=model, contents=prompt)
                content = (response.text or '').strip()
                if not content:
                    raise ValueError('Gemini 未回傳文字，可能受到內容安全限制。')
                return content
            except (errors.ServerError, errors.ClientError) as exc:
                code = getattr(exc, 'code', None)
                if code is None:
                    code = getattr(exc, 'status_code', None)
                try:
                    code = int(code)
                except (TypeError, ValueError):
                    code = None
                # 429、500、502、503、504 才重試；400、401、403、404 直接回報
                if code not in (429, 500, 502, 503, 504) or attempt == max_attempts - 1:
                    raise
                wait_seconds = 2 ** (attempt + 1)
                status.warning(f'Gemini 暫時無法服務（HTTP {code}），{wait_seconds} 秒後重試…')
                time.sleep(wait_seconds)
    finally:
        status.empty()
        client.close()


st.subheader('🤖 Gemini AI 總體經濟摘要')
st.caption('使用最新可用數據與新聞標題生成約 200 字短評；新聞標題未核實全文，不能視為市場因果證據。')

try:
    init_history()
    history_available = True
except (OSError, sqlite3.Error) as exc:
    history_available = False
    st.warning(f'目前無法使用歷史紀錄儲存：{exc}')

if st.button('✨ 產生 AI 總體經濟摘要', type='primary'):
    if not gemini_key:
        st.warning('請先在 Streamlit Secrets 設定 GEMINI_API_KEY。')
    elif macro_df.empty:
        st.warning('需要有效的 FRED API Key 與總經資料才能產生摘要。')
    else:
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
        context.extend('新聞標題（未核實全文）：' + x['title'] + ' / ' + x['published']
                       for x in news[:5])
        prompt = (
            '請以繁體中文撰寫約 200 個中文字的總體經濟短評，重點是美國通膨、就業、利率、'
            '10Y−2Y 利差，以及美元/台幣與英鎊/台幣的連動，兼顧英國留學換匯者。'
            '務必區分各指標觀測日與新聞日期；CPI 指數不是通膨率，日本殖利率為月頻；'
            '新聞標題未驗證全文，不能把標題直接當因果證據；'
            '不得杜撰政策、數字、即時事件或確定的匯率預測。資料不足請明確指出。'
            '\n\n資料：\n' + '\n'.join(context)
        )
        try:
            summary = call_gemini_with_retry(gemini_key, gemini_model, prompt)
            st.success('摘要產生成功！')
            st.markdown(summary)
            if history_available:
                try:
                    created_at = save_history(gemini_model, summary)
                    st.caption(f'已加入歷史紀錄｜{created_at}')
                except (OSError, sqlite3.Error) as exc:
                    st.warning(f'摘要成功，但無法寫入歷史紀錄：{exc}')
        except Exception as exc:
            st.error(f'Gemini 呼叫失敗：{type(exc).__name__}: {exc}')
            st.info('若為 503，代表模型服務暫時繁忙；若為 404，請確認目前帳號可用的模型名稱。')

st.subheader('🗂️ AI 摘要歷史紀錄')
st.caption('紀錄儲存在此 Streamlit 執行環境的暫存資料庫；重新部署、容器重建或休眠恢復可能遺失。請下載備份。')
if history_available:
    try:
        records = load_history()
        if not records:
            st.info('目前還沒有摘要紀錄。成功產生摘要後會自動新增。')
        else:
            st.write(f'目前有 {len(records)} 筆摘要紀錄')
            export = [dict(id=r[0], created_at=r[1], model=r[2], summary=r[3]) for r in records]
            st.download_button('⬇️ 下載全部歷史紀錄（JSON）',
                               data=json.dumps(export, ensure_ascii=False, indent=2).encode('utf-8'),
                               file_name='gemini_summary_history.json', mime='application/json')
            for record_id, created_at, model, summary in records:
                with st.expander(f'📅 {created_at}｜{model}｜紀錄 #{record_id}'):
                    st.markdown(summary)
                    st.download_button('下載此摘要（TXT）', data=summary,
                                       file_name=f'gemini_summary_{record_id}.txt',
                                       mime='text/plain', key=f'download_{record_id}')
                    if st.button('🗑️ 刪除此筆紀錄', key=f'delete_{record_id}'):
                        delete_history(record_id)
                        st.rerun()
    except (OSError, sqlite3.Error) as exc:
        st.warning(f'讀取歷史紀錄失敗：{exc}')

with st.expander('資料來源、缺漏與更新頻率'):
    st.markdown('FRED：美國 CPI／失業率／有效聯邦基金利率通常為月資料；DGS10、DGS2 通常為日資料；日本 10 年公債系列 IRLTLT01JPM156N 為月資料。Yahoo Finance 為市場參考行情，可能延遲或缺漏。新聞為 Google News RSS。')
    if macro_errors:
        st.warning('FRED 部分系列失敗：' + '；'.join(macro_errors.keys()))
    if market_errors:
        st.warning('市場資料備註：' + '；'.join(f'{k}: {v}' for k, v in market_errors.items()))
    st.markdown('[FRED](https://fred.stlouisfed.org/) · [Yahoo Finance](https://finance.yahoo.com/) · [台新銀行](https://www.taishinbank.com.tw/)')
