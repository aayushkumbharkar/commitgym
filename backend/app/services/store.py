"""SQLite document store. Each operation uses a short transaction; no secrets are stored."""
import json
import os
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone

class Store:
    def __init__(self, path=None):
        self.path = path or os.getenv('COMMITGYM_DB','data/commitgym.sqlite3')
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as c:
            c.execute('CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, kind TEXT, created TEXT, body TEXT)')
            c.execute('CREATE TABLE IF NOT EXISTS spending (id TEXT PRIMARY KEY, reserved REAL, actual REAL, tokens INTEGER, status TEXT)')
    def connect(self):
        c=sqlite3.connect(self.path, timeout=15)
        c.execute('PRAGMA journal_mode=WAL')
        return c
    def save(self, kind, body, id=None):
        id=id or str(uuid.uuid4())
        body={**body,'id':id}
        with self.connect() as c:
            c.execute('INSERT INTO records VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET body=excluded.body',
                      (id,kind,datetime.now(timezone.utc).isoformat(),json.dumps(body)))
        return body
    def get(self, id, kind):
        with self.connect() as c: row=c.execute('SELECT body FROM records WHERE id=? AND kind=?',(id,kind)).fetchone()
        return json.loads(row[0]) if row else None
    def list(self, kind, limit=100):
        with self.connect() as c: rows=c.execute('SELECT body FROM records WHERE kind=? ORDER BY created DESC LIMIT ?',(kind,limit)).fetchall()
        return [json.loads(r[0]) for r in rows]
    def budget(self):
        with self.connect() as c:
            row=c.execute('SELECT count(*), coalesce(sum(coalesce(actual,reserved)),0), coalesce(sum(tokens),0) FROM spending').fetchone()
        return {'requests':row[0],'estimated_spend_usd':row[1],'tokens':row[2]}
    def reserve(self, amount, cap, max_requests):
        with self.connect() as c:
            c.execute('BEGIN IMMEDIATE')
            count, spent=c.execute('SELECT count(*),coalesce(sum(coalesce(actual,reserved)),0) FROM spending').fetchone()
            if count >= max_requests or spent+amount > cap: raise ValueError('Request or spending budget exhausted')
            id=str(uuid.uuid4()); c.execute('INSERT INTO spending VALUES (?,?,NULL,0,?)',(id,amount,'reserved'))
        return id
    def settle(self, id, actual, tokens, status):
        with self.connect() as c: c.execute('UPDATE spending SET actual=?,tokens=?,status=? WHERE id=?',(actual,tokens,status,id))
