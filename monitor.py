"""Opt-in daily monitor for GitHub Actions; does not run inside Streamlit.

Requires GitHub repository Secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID.
FRED_API_KEY is optional but enables 10Y-2Y and CPI alert checks.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demo'))
import pandas as pd
import requests
import yfinance as yf
import core


def fetch_macro(key):
    if not key:
        return pd.DataFrame()
    from fredapi import Fred
    fred = Fred(api_key=key)
    data = {}
    for name in (core.CPI, '美國 10 年期公債殖利率',
                 '美國 2 年期公債殖利率'):
        try:
            data[name] = fred.get_series(core.FRED_SERIES[name], observation_start='2024-01-01')
        except Exception as exc:
            print('FRED unavailable:', name, type(exc).__name__)
    return core.enrich_macro(pd.DataFrame(data)) if data else pd.DataFrame()


def fetch_market():
    data = {}
    for label in (core.USD, core.GBPUSD, core.GBPTWD, '布蘭特原油期貨 (USD/桶)'):
        try:
            raw = yf.download(core.MARKET_TICKERS[label], period='7d', interval='1d',
                              auto_adjust=True, threads=False, progress=False, timeout=8)
            if not raw.empty:
                s = raw['Close']
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s.index = pd.to_datetime(s.index).tz_localize(None)
                data[label] = s
        except Exception as exc:
            print('Yahoo unavailable:', label, type(exc).__name__)
    if not data:
        return pd.DataFrame()
    out, _ = core.derive_cross(pd.DataFrame(data))
    return out


def main():
    token = os.getenv('TELEGRAM_BOT_TOKEN', '')
    chat = os.getenv('TELEGRAM_CHAT_ID', '')
    if not token or not chat:
        print('Telegram GitHub Secrets not configured; no notification sent.')
        return
    macro = fetch_macro(os.getenv('FRED_API_KEY', ''))
    market = fetch_market()
    if macro.empty and market.empty:
        print('No live data; skip notification instead of sending speculative alert.')
        return
    fx_threshold = float(os.getenv('ALERT_FX_THRESHOLD', '1.0'))
    oil_threshold = float(os.getenv('ALERT_OIL_THRESHOLD', '3.0'))
    cpi_threshold = float(os.getenv('ALERT_CPI_THRESHOLD', '4.0'))
    alerts = core.detect_alerts(macro, market, fx_threshold, oil_threshold, cpi_threshold)
    if not alerts:
        print('No threshold breach; no message sent.')
        return
    msg = '🚨 AI Economics V4 每日監測（非投資建議）\n' + '\n'.join(
        f'• {a["name"]}：{a["detail"]}（觀測日 {a["date"]}）' for a in alerts)
    resp = requests.post(f'https://api.telegram.org/bot{token}/sendMessage',
                         json={'chat_id': chat, 'text': msg}, timeout=10)
    resp.raise_for_status()
    if not resp.json().get('ok'):
        raise RuntimeError('Telegram did not confirm sendMessage')
    print('Alert accepted by Telegram.')


if __name__ == '__main__':
    main()
