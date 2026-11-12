"""Optional Supabase REST persistence with local SQLite fallback. Never expose service key."""
from __future__ import annotations
import json
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

import requests

DB = Path(tempfile.gettempdir()) / 'ai_economics_v4.sqlite3'


def init_local() -> None:
    with sqlite3.connect(DB, timeout=10) as con:
        con.execute('''CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            model TEXT NOT NULL,
            signature TEXT NOT NULL UNIQUE,
            sentiment_score INTEGER NOT NULL,
            hawkish_score INTEGER NOT NULL,
            payload TEXT NOT NULL
        )''')


def save_local(record: dict) -> None:
    init_local()
    with sqlite3.connect(DB, timeout=10) as con:
        con.execute('''INSERT OR IGNORE INTO analyses
          (created_at, model, signature, sentiment_score, hawkish_score, payload)
          VALUES (?, ?, ?, ?, ?, ?)''',
          (record['created_at'], record['model'], record['signature'],
           record['sentiment_score'], record['hawkish_score'],
           json.dumps(record['payload'], ensure_ascii=False)))


def list_local(limit: int = 150) -> list[dict]:
    init_local()
    with sqlite3.connect(DB, timeout=10) as con:
        rows = con.execute('''SELECT id, created_at, model, signature,
                            sentiment_score, hawkish_score, payload
                            FROM analyses ORDER BY created_at DESC LIMIT ?''', (limit,)).fetchall()
    return [dict(id=r[0], created_at=r[1], model=r[2], signature=r[3],
                 sentiment_score=r[4], hawkish_score=r[5], payload=json.loads(r[6])) for r in rows]


def delete_local(row_id: int) -> None:
    with sqlite3.connect(DB, timeout=10) as con:
        con.execute('DELETE FROM analyses WHERE id=?', (row_id,))


def enabled(url: str, key: str) -> bool:
    return bool(url and key)


def _endpoint(url: str) -> str:
    if not url.startswith('https://'):
        raise ValueError('Supabase URL 必須是 HTTPS')
    return url.rstrip('/') + '/rest/v1/ai_analyses'


def _headers(key: str) -> dict:
    return {'apikey': key, 'Authorization': 'Bearer ' + key,
            'Content-Type': 'application/json', 'Prefer': 'return=minimal,resolution=ignore-duplicates'}


def save_supabase(url: str, key: str, record: dict) -> None:
    response = requests.post(_endpoint(url), headers=_headers(key), json=record, timeout=10)
    response.raise_for_status()


def list_supabase(url: str, key: str, limit: int = 150) -> list[dict]:
    response = requests.get(_endpoint(url), headers=_headers(key),
                            params={'select': 'id,created_at,model,signature,sentiment_score,hawkish_score,payload',
                                    'order': 'created_at.desc', 'limit': limit}, timeout=10)
    response.raise_for_status()
    return response.json()


def delete_supabase(url: str, key: str, row_id: int) -> None:
    response = requests.delete(_endpoint(url), headers=_headers(key),
                               params={'id': f'eq.{row_id}'}, timeout=10)
    response.raise_for_status()


def save(record: dict, url: str = '', key: str = '') -> str:
    if enabled(url, key):
        save_supabase(url, key, record)
        return 'Supabase'
    save_local(record)
    return '本機暫存 SQLite'


def load(url: str = '', key: str = '', limit: int = 150) -> list[dict]:
    if enabled(url, key):
        return list_supabase(url, key, limit)
    return list_local(limit)


def delete(row_id: int, url: str = '', key: str = '') -> None:
    if enabled(url, key):
        delete_supabase(url, key, row_id)
    else:
        delete_local(row_id)
