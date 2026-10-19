"""Archivio personale locale: esecuzioni, diario, calendario e profilo. Nessun ordine."""
from datetime import datetime, timezone
from contextlib import contextmanager
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import threading

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / '06_dati/quantix_iq.sqlite'
GUARD = threading.RLock()
DEFAULT_PROFILE = {'name': 'Marco', 'email': '', 'account_alias': 'Il mio conto',
                   'notify_open': True, 'notify_close': True, 'notify_stops': True,
                   'notify_risk': True, 'notify_connection': True, 'notify_amounts': False,
                   'risk_warning': 50}
CONFIG = json.loads((ROOT / '07_collegamento/config.json').read_text(encoding='utf-8'))
SCOPE = hashlib.sha256((str(CONFIG['expected_login'])+'|'+CONFIG['expected_server']).encode()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def cash(values):
    return float(sum((Decimal(str(v or 0)) for v in values), Decimal('0')).quantize(Decimal('.01')))


@contextmanager
def connect():
    DB.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB, timeout=15)
    db.row_factory = sqlite3.Row
    try:
        db.execute('PRAGMA journal_mode=WAL')
        with db:
            yield db
    finally:
        db.close()


def initialize():
    with GUARD, connect() as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS deals(id TEXT PRIMARY KEY,day TEXT NOT NULL,position_id TEXT NOT NULL,data TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS deals_day ON deals(day);
        CREATE TABLE IF NOT EXISTS contexts(position_id TEXT PRIMARY KEY,observed TEXT NOT NULL,data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,position_id TEXT NOT NULL,stamp TEXT NOT NULL,kind TEXT NOT NULL,data TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS notes(position_id TEXT PRIMARY KEY,note TEXT NOT NULL,updated TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS notices(id TEXT PRIMARY KEY,created TEXT NOT NULL,kind TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS subscriptions(id TEXT PRIMARY KEY,data TEXT NOT NULL,created TEXT NOT NULL,enabled INTEGER NOT NULL DEFAULT 1,last_status TEXT);
        CREATE TABLE IF NOT EXISTS deliveries(notice_id TEXT NOT NULL,subscription_id TEXT NOT NULL,attempts INTEGER NOT NULL DEFAULT 0,next_try REAL NOT NULL DEFAULT 0,status TEXT NOT NULL DEFAULT 'pending',PRIMARY KEY(notice_id,subscription_id));
        ''')


def meta(db, key, default=None):
    row = db.execute('SELECT value FROM meta WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else default


def put(db, key, value):
    db.execute('INSERT OR REPLACE INTO meta VALUES (?,?)', (key, dumps(value)))


def profile():
    with GUARD, connect() as db:
        return {**DEFAULT_PROFILE, **meta(db, 'profile', {})}


def save_profile(value):
    if not isinstance(value, dict) or set(value) - set(DEFAULT_PROFILE):
        raise ValueError('Campi del profilo non validi.')
    current = profile()
    for key, limit in [('name', 60), ('email', 160), ('account_alias', 80)]:
        if key in value:
            if not isinstance(value[key], str) or len(value[key].strip()) > limit:
                raise ValueError('Profilo troppo lungo o non valido.')
            current[key] = value[key].strip()
    if not current['name'] or not current['account_alias']:
        raise ValueError('Inserisci nome e nome del conto.')
    if current['email'] and not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', current['email']):
        raise ValueError('Indirizzo email non valido.')
    for key in DEFAULT_PROFILE:
        if key.startswith('notify_') and key in value:
            if not isinstance(value[key], bool):
                raise ValueError('Preferenza notifiche non valida.')
            current[key] = value[key]
    if 'risk_warning' in value:
        threshold = value['risk_warning']
        if isinstance(threshold, bool) or not isinstance(threshold, (float, int)) or not 1 <= threshold <= 10000:
            raise ValueError('Soglia avviso compresa tra 1 e 10000 nella valuta del conto.')
        current['risk_warning'] = round(threshold, 2)
    with GUARD, connect() as db:
        put(db, 'profile', current)
    return current


def notice(db, key, kind, body, position_id=None, day=None, amount=None, currency=None):
    payload = {'title': 'QUANTIX IQ', 'body': body, 'kind': kind,
               'position_id': position_id, 'day': day, 'amount': amount, 'currency': currency}
    db.execute('INSERT OR IGNORE INTO notices VALUES (?,?,?,?)', (key, now(), kind, dumps(payload)))


def ingest(snapshot):
    """Importazione idempotente; al primo avvio lo storico non genera notifiche."""
    if not snapshot.get('ok'):
        with GUARD, connect() as db:
            before = meta(db, 'connection_ok')
            if before is True:
                notice(db, 'offline:' + now()[:16], 'connection', 'Il collegamento a MT5 non risponde. Apri l’app per verificare.')
            put(db, 'connection_ok', False)
        return
    stamp = snapshot['observed_italy']
    observed = snapshot['observed_utc']
    positions = {str(p['position_id']): p for p in snapshot.get('positions', [])}
    with GUARD, connect() as db:
        if meta(db, 'account_scope', SCOPE) != SCOPE:
            raise ValueError('Il conto configurato è diverso da quello dell’archivio. Creare un archivio separato.')
        put(db, 'account_scope', SCOPE)
        initialized = meta(db, 'initialized', False)
        previous = meta(db, 'positions', {})
        currency = snapshot['account']['currency']
        magic = snapshot['bot']['magic']
        for deal in snapshot.get('journal_deals', snapshot.get('deals', [])):
            key = str(deal.get('ticket') or hashlib.sha256(dumps(deal).encode()).hexdigest())
            entry = int(deal['entry'])
            position_id = str(deal['position_id'])
            row = db.execute('INSERT OR IGNORE INTO deals VALUES (?,?,?,?)',
                             (key, deal['day'], position_id, dumps(deal)))
            recent = abs((datetime.fromisoformat(observed) - datetime.fromisoformat(deal['time_italy'])).total_seconds()) < 180
            if initialized and row.rowcount and recent:
                kind = 'open' if entry == 0 else 'close'
                text = ('Nuova entrata' if entry == 0 else 'Esecuzione di chiusura') + f" {deal['side']} {deal['symbol']} · {deal['volume']:g} lotti."
                notice(db, 'deal:' + key, kind, text, position_id, deal['day'], deal['net'], currency)
        for key, position in positions.items():
            same_symbol = position['symbol'] == snapshot['bot']['symbol']
            context = {'position': position, 'account': snapshot['account'], 'quote': snapshot['quote'] if same_symbol else {},
                       'analysis': snapshot['analysis'] if same_symbol else {'title': 'Asset diverso dal bot', 'reason': 'I filtri del bot e il suo grafico non sono attribuiti a questa operazione.'},
                       'bot': snapshot['bot'], 'captured_at': stamp,
                       'candles': snapshot.get('candles', [])[-48:] if same_symbol else [],
                       'source': 'Prima rilevazione disponibile nell’app; non certifica le condizioni al momento dell’ingresso.'}
            db.execute('INSERT OR IGNORE INTO contexts VALUES (?,?,?)', (key, stamp, dumps(context)))
            old = previous.get(key)
            if old and (old.get('sl') != position.get('sl') or old.get('tp') != position.get('tp')):
                eid = 'stops:' + key + ':' + stamp
                change = {'before': {'sl': old.get('sl'), 'tp': old.get('tp')},
                          'after': {'sl': position.get('sl'), 'tp': position.get('tp')}}
                db.execute('INSERT OR IGNORE INTO events VALUES (?,?,?,?,?)', (eid, key, stamp, 'stops', dumps(change)))
                notice(db, eid, 'stops', f"SL/TP aggiornati · {position['symbol']} {position['side']}.", key, stamp[:10])
        connected = bool(snapshot['connection']['connected'])
        if initialized and connected and meta(db, 'connection_ok') is False:
            notice(db, 'online:' + observed[:16], 'connection', 'Collegamento a MT5 ripristinato.')
        elif initialized and not connected and meta(db, 'connection_ok') is True:
            notice(db, 'offline:' + observed[:16], 'connection', 'MT5 risulta disconnesso dal broker.')
        p = {**DEFAULT_PROFILE, **meta(db, 'profile', {})}
        remaining = snapshot['account']['protection_distance_estimate']
        day = stamp[:10]
        if initialized and remaining <= p['risk_warning']:
            notice(db, 'risk:' + day, 'risk', 'Margine stimato alla protezione ridotto. Controlla il conto.',
                   day=day, amount=remaining, currency=currency)
        put(db, 'positions', positions)
        put(db, 'connection_ok', bool(snapshot['connection']['connected']))
        put(db, 'account', {'currency': currency, 'label': snapshot['account']['label'],
                            'broker': CONFIG['expected_server'],
                            'bot': snapshot['bot']['name'], 'magic': magic, 'symbol': snapshot['bot']['symbol'],
                            'observed_italy': stamp, 'history_from': snapshot.get('history_from'),
                            'history_complete_from': snapshot.get('history_from')})
        put(db, 'initialized', True)


def all_trades(db, origin='all'):
    groups = {}
    account = meta(db, 'account', {})
    live = meta(db, 'positions', {})
    for row in db.execute('SELECT data FROM deals ORDER BY day,id'):
        deal = json.loads(row[0])
        groups.setdefault(str(deal['position_id']), []).append(deal)
    result = []
    for pid, deals in groups.items():
        deals.sort(key=lambda d: (d['time_italy'], int(d.get('ticket') or 0)))
        entries = [d for d in deals if d['entry'] == 0]
        exits = [d for d in deals if d['entry'] in (1, 2)]
        reversal = any(d['entry'] == 3 for d in deals)
        first = entries[0] if entries else deals[0]
        own = bool(entries and first.get('magic') == account.get('magic') and first['symbol'] == account.get('symbol'))
        if origin == 'bot' and not own or origin == 'other' and own:
            continue
        vin = sum(Decimal(str(d['volume'])) for d in entries)
        vout = sum(Decimal(str(d['volume'])) for d in exits)
        closed = bool(entries and exits and vin == vout and pid not in live and not reversal)
        state = 'closed' if closed else 'open' if pid in live else 'partial' if entries and exits else 'unknown'
        comment = str(first.get('comment', ''))
        tag = comment.split('|', 1)[0]
        session = tag if tag in ('ASIA', 'ASIA_FADE', 'LONDON', 'NY', 'NY_FADE', 'NY_REVERSAL', 'PYRAMID', 'EXTRA') else None
        result.append({'id': pid, 'symbol': first['symbol'], 'side': first['side'] if entries else 'Da verificare',
                       'own_bot': own, 'session': session, 'state': state, 'opened': entries[0]['time_italy'] if entries else None,
                       'closed': exits[-1]['time_italy'] if closed else None,
                       'entry_price': first['price'] if entries else None,
                       'exit_price': exits[-1]['price'] if closed else None,
                       'volume': float(vin), 'net': cash(d['net'] for d in deals),
                       'costs': cash(d.get('costs', 0) for d in deals), 'executions': len(deals),
                       'days': sorted(set(d['day'] for d in deals)), 'floating': live.get(pid, {}).get('profit'),
                       'current': live.get(pid), 'deals': deals})
    return sorted(result, key=lambda t: t['opened'] or t['deals'][0]['time_italy'], reverse=True)


def summary(trades, deals):
    closed = [t for t in trades if t['state'] == 'closed']
    gains = cash(t['net'] for t in closed if t['net'] > 0)
    losses = cash(t['net'] for t in closed if t['net'] < 0)
    return {'net': cash(d['net'] for d in deals), 'trades': len(trades), 'closed': len(closed),
            'executions': len(deals), 'win_rate': round(sum(t['net'] > 0 for t in closed) / len(closed) * 100, 2) if closed else None,
            'profit_factor': round(gains / abs(losses), 2) if losses else None,
            'average_closed': round(cash(t['net'] for t in closed) / len(closed), 2) if closed else None,
            'best': max((t['net'] for t in closed), default=None),
            'worst': min((t['net'] for t in closed), default=None),
            'costs': cash(d.get('costs', 0) for d in deals)}


def calendar_data(month, origin='all'):
    if not re.fullmatch(r'20\d\d-(0[1-9]|1[0-2])', month):
        raise ValueError('Mese non valido.')
    with GUARD, connect() as db:
        account = meta(db, 'account', {})
        trades = all_trades(db, origin)
        deals = [d for t in trades for d in t['deals'] if d['day'].startswith(month)]
        days = {}
        for d in deals:
            days.setdefault(d['day'], []).append(d)
        daily = [{'date': day, 'net': cash(d['net'] for d in ds),
                  'trades': len(set(str(d['position_id']) for d in ds)), 'executions': len(ds)}
                 for day, ds in sorted(days.items())]
        completed = [t for t in trades if t['closed'] and t['closed'].startswith(month)]
        totals = summary(completed, deals)
        totals['trades'] = len(set(str(d['position_id']) for d in deals))
        totals['active_days'] = len(days)
        months = [r[0] for r in db.execute('SELECT DISTINCT substr(day,1,7) FROM deals ORDER BY day DESC')]
        return {'ok': True, 'month': month, 'currency': account.get('currency', 'USD'), 'daily': daily,
                'totals': totals, 'months': months, 'coverage': account, 'origin': origin,
                'note': 'P&L realizzato delle esecuzioni, con costi registrati nel giorno italiano. Win rate e profit factor sulle posizioni interamente chiuse nel periodo. Flottante e versamenti esclusi.'}


def day_data(day, origin='all'):
    try:
        parsed = datetime.strptime(day, '%Y-%m-%d')
        if parsed.year < 2000 or parsed.year > 2100:
            raise ValueError()
    except (ValueError, TypeError):
        raise ValueError('Giorno non valido.')
    with GUARD, connect() as db:
        trades = [t for t in all_trades(db, origin) if day in t['days']]
        deals = [d for t in trades for d in t['deals'] if d['day'] == day]
        completed = [t for t in trades if t['closed'] and t['closed'][:10] == day]
        totals = summary(completed, deals)
        totals['trades'] = len(trades)
        for t in trades:
            t['day_net'] = cash(d['net'] for d in t['deals'] if d['day'] == day)
            t.pop('deals')
        return {'ok': True, 'day': day, 'totals': totals, 'trades': trades,
                'deals': sorted(deals, key=lambda d: d['time_italy'], reverse=True),
                'currency': meta(db, 'account', {}).get('currency', 'USD')}


def journal_data(origin='all', limit=250):
    with GUARD, connect() as db:
        trades = all_trades(db, origin)
        notes = {r[0]: r[1] for r in db.execute('SELECT position_id,note FROM notes')}
        for t in trades:
            t.pop('deals')
            t['has_note'] = bool(notes.get(t['id']))
        return {'ok': True, 'trades': trades[:limit], 'total': len(trades),
                'coverage': meta(db, 'account', {}), 'currency': meta(db, 'account', {}).get('currency', 'USD')}


def trade_data(pid):
    with GUARD, connect() as db:
        trade = next((t for t in all_trades(db) if t['id'] == pid), None)
        if not trade:
            raise ValueError('Operazione non presente nel diario.')
        ctx = db.execute('SELECT observed,data FROM contexts WHERE position_id=?', (pid,)).fetchone()
        note = db.execute('SELECT note,updated FROM notes WHERE position_id=?', (pid,)).fetchone()
        events = [dict(r) for r in db.execute('SELECT stamp,kind,data FROM events WHERE position_id=? ORDER BY stamp', (pid,))]
        for event in events:
            event['data'] = json.loads(event['data'])
        return {'ok': True, 'trade': trade, 'context': json.loads(ctx['data']) if ctx else None,
                'chart_symbol': meta(db, 'account', {}).get('symbol'),
                'events': events, 'note': note['note'] if note else '',
                'currency': meta(db, 'account', {}).get('currency', 'USD')}


def save_note(pid, text):
    if not isinstance(text, str) or len(text) > 4000:
        raise ValueError('La nota può contenere al massimo 4000 caratteri.')
    with GUARD, connect() as db:
        if not db.execute('SELECT 1 FROM deals WHERE position_id=? LIMIT 1', (pid,)).fetchone():
            raise ValueError('Operazione non presente nel diario.')
        db.execute('INSERT OR REPLACE INTO notes VALUES (?,?,?)', (pid, text.strip(), now()))
    return {'ok': True}


def account_data():
    with GUARD, connect() as db:
        return {'ok': True, 'profile': {**DEFAULT_PROFILE, **meta(db, 'profile', {})},
                'account': meta(db, 'account', {}),
                'storage': {'executions': db.execute('SELECT count(*) FROM deals').fetchone()[0],
                            'notes': db.execute("SELECT count(*) FROM notes WHERE note<>''").fetchone()[0]},
                'scope': 'Profilo personale condiviso sui tuoi dispositivi autorizzati tramite Tailscale. Dati conservati sul PC; nessuna password di trading salvata nel profilo.'}


initialize()
