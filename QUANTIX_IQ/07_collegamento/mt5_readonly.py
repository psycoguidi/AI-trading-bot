"""Lettura del terminale esistente. Nessun login, ordine o modifica al bot."""
import calendar
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import os
from entry_criteria import build_criteria, risk_observations

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / '07_collegamento/config.json').read_text(encoding='utf-8'))
UTC = timezone.utc


def change_day(year, month, weekday, occurrence):
    first = datetime(year, month, 1, tzinfo=UTC)
    return 1 + (weekday - first.weekday()) % 7 + 7 * (occurrence - 1)


def eu_offset(utc):
    def last_sunday(month):
        last = calendar.monthrange(utc.year, month)[1]
        d = datetime(utc.year, month, last, 1, tzinfo=UTC)
        return d - timedelta(days=(d.weekday() + 1) % 7)
    return 2 if last_sunday(3) <= utc < last_sunday(10) else 1


def server_offset(utc, params):
    winter = int(params.get('InpFTMOServerUTCWinter', 2))
    summer = int(params.get('InpFTMOServerUTCSummer', 3))
    mode = int(params.get('InpFTMOServerDST', 1))
    if mode == 2:
        return winter
    if mode == 0:
        return summer if eu_offset(utc) == 2 else winter
    start = datetime(utc.year, 3, change_day(utc.year, 3, 6, 2), 7, tzinfo=UTC)
    end = datetime(utc.year, 11, change_day(utc.year, 11, 6, 1), 6, tzinfo=UTC)
    return summer if start <= utc < end else winter


def raw_to_utc(raw, params):
    candidate = datetime.fromtimestamp(raw, UTC)
    return candidate - timedelta(hours=server_offset(candidate, params))


def italian(utc):
    return utc.astimezone(timezone(timedelta(hours=eu_offset(utc))))


def params_from_profile():
    file = Path(CONFIG['terminal_data']) / 'MQL5/Profiles/Charts' / CONFIG['profile'] / 'chart01.chr'
    raw = file.read_bytes()
    text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig', errors='replace')
    expert = text.split('<expert>', 1)[1].split('</expert>', 1)[0]
    name = re.search(r'^name=(.+)$', expert, re.M).group(1).strip()
    entries = expert.split('<inputs>', 1)[1].split('</inputs>', 1)[0].splitlines()
    params = dict(line.strip().split('=', 1) for line in entries if line.startswith('Inp') and '=' in line)
    return params, {'name': name, 'profile_source': 'Profilo MT5 salvato', 'profile_saved': file.stat().st_mtime,
                    'ea_permission_from_profile': bool(re.search(r'^expertmode=1\s*$', expert, re.M))}


def terminal_running():
    # Non inizializzare un terminale chiuso: verifica il processo prima dell'API.
    result = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq terminal64.exe', '/FO', 'CSV', '/NH'],
                            capture_output=True, creationflags=0x08000000, timeout=5)
    if b'terminal64.exe' not in result.stdout.lower():
        return False
    # Le copie usate dal tester hanno lo stesso nome: deve essere aperto il terminale esatto.
    command = "Get-CimInstance Win32_Process -Filter \"Name='terminal64.exe'\" | Select-Object -ExpandProperty ExecutablePath | ConvertTo-Json -Compress"
    result = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', command],
                            capture_output=True, creationflags=0x08000000, timeout=6)
    raw = result.stdout.decode('utf-8-sig', errors='replace').strip()
    values = json.loads(raw) if raw else []
    paths = values if isinstance(values, list) else [values]
    expected = CONFIG['terminal_path'].replace('\\', '/').casefold()
    return any(isinstance(value, str) and value.replace('\\', '/').casefold() == expected for value in paths)


def read_observer(now, params):
    """Il lettore MQL5 pubblica solo il commento esposto dal bot sul grafico."""
    file = Path(os.environ.get('APPDATA', str(Path.home() / 'AppData/Roaming'))) / 'MetaQuotes/Terminal/Common/Files/QuantixIQ/observer.json'
    try:
        value = json.loads(file.read_text(encoding='utf-8-sig'))
        age = now.timestamp() - float(value['observed_utc_epoch'])
        if value.get('account') != CONFIG['expected_login'] or value.get('server') != CONFIG['expected_server'] or value.get('bot') != 'prova298' or value.get('symbol') != CONFIG['symbol'] or age < -5 or age > 15 or not value.get('chart_found'):
            return {'available': False, 'reason': 'Lettore dei livelli non aggiornato o grafico non trovato.'}
        comment = str(value.get('comment', ''))
        levels = {}
        for label, name in [('Lon', 'Londra'), ('NY', 'New York'), ('Asia', 'Asia')]:
            match = re.search(r'\b'+label+r' H/L:\s*([0-9.]+)/([0-9.]+)', comment)
            if match:
                high, low = map(float, match.groups())
                if 0 < low < high < 999999:
                    levels[name] = {'high': high, 'low': low}
        trades = re.search(r'\bTrades:\s*(\d+)', comment)
        bias = re.search(r'\bBias:\s*(BULL|BEAR|NEUT)\b', comment)
        return {'available': True, 'age_seconds': round(max(0, age), 1), 'levels': levels,
                'bias': bias.group(1) if bias else None,
                'source': 'Livelli esposti dal bot nel commento MT5', 'chart_id': value.get('chart_id'),
                'trades_today_from_comment': int(trades.group(1)) if trades else None}
    except (OSError, ValueError, KeyError, TypeError):
        return {'available': False, 'reason': 'Lettore dei livelli non disponibile.'}


def read_snapshot():
    now = datetime.now(UTC)
    base = {'observed_utc': now.isoformat(), 'observed_italy': italian(now).isoformat(),
            'mode': 'mt5_read_only', 'orders_enabled': False, 'ok': False}
    if not terminal_running():
        return {**base, 'error': 'MT5 non è aperto. Nessun terminale è stato avviato dall’app.'}
    params, bot = params_from_profile()
    sys.path.insert(0, CONFIG['python_dependencies'])
    import MetaTrader5 as mt5
    if not mt5.initialize(CONFIG['terminal_path'], timeout=5000):
        return {**base, 'error': 'Il terminale MT5 aperto non risponde alla lettura.'}
    try:
        terminal, account = mt5.terminal_info(), mt5.account_info()
        if not terminal or not account:
            return {**base, 'error': 'Dati del terminale non disponibili.'}
        if account.login != CONFIG['expected_login'] or account.server != CONFIG['expected_server']:
            return {**base, 'error': 'Il conto aperto è diverso da quello configurato. Lettura sospesa.'}
        tick = mt5.symbol_info_tick(CONFIG['symbol'])
        positions, orders = mt5.positions_get(), mt5.orders_get()
        history_start = datetime(2025, 1, 1, tzinfo=UTC)
        deals = mt5.history_deals_get(history_start, now + timedelta(days=2))
        if tick is None or positions is None or orders is None or deals is None:
            return {**base, 'error': 'MT5 non ha restituito tutti i dati richiesti. Riprovo al prossimo aggiornamento.'}
        quote_utc = raw_to_utc(tick.time, params)
        age = max(0, (now - quote_utc).total_seconds())
        cash_types = {mt5.DEAL_TYPE_BUY, mt5.DEAL_TYPE_SELL}
        today = italian(now).date().isoformat()
        daily, converted = {}, []
        all_today_cash = 0.0
        for deal in deals:
            stamp = raw_to_utc(deal.time, params)
            day = italian(stamp).date().isoformat()
            cash = deal.profit + deal.commission + deal.swap + deal.fee
            if day == today:
                all_today_cash += cash
            if deal.type not in cash_types:
                continue
            if stamp > now + timedelta(minutes=1):
                continue
            daily[day] = daily.get(day, 0.0) + cash
            converted.append({'time_raw': int(deal.time), 'ticket': str(getattr(deal, 'ticket', '')), 'comment': str(getattr(deal, 'comment', ''))[:256], 'time_italy': italian(stamp).isoformat(), 'day': day,
                              'symbol': deal.symbol, 'side': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                              'entry': deal.entry, 'volume': deal.volume, 'price': deal.price,
                              'net': round(cash, 2), 'profit': round(deal.profit, 2),
                              'costs': round(deal.commission + deal.swap + deal.fee, 2),
                              'magic': deal.magic, 'position_id': str(deal.position_id)})
        midnight_balance = account.balance - all_today_cash
        capital = float(params.get('InpFTMOInitialCapital', 10000))
        daily_limit = float(params.get('InpFTMODailyLossEUR', 450))
        buffer = float(params.get('InpFTMOBufferEUR', 75))
        total_floor = capital - float(params.get('InpFTMOTotalLossEUR', 1000))
        protection = max(midnight_balance - daily_limit, total_floor) + buffer
        magic = int(params.get('InpMagicNumber', 0))
        own = [p for p in positions if p.symbol == CONFIG['symbol'] and p.magic == magic]
        actual_positions = [{'symbol': p.symbol, 'side': 'BUY' if p.type == mt5.POSITION_TYPE_BUY else 'SELL',
                             'volume': p.volume, 'entry_price': p.price_open, 'price': p.price_current,
                             'sl': p.sl, 'tp': p.tp, 'profit': p.profit, 'swap': p.swap,
                             'time_italy': italian(raw_to_utc(p.time, params)).isoformat(),
                             'own_bot': p.magic == magic and p.symbol == CONFIG['symbol'],
                             'entry_kind': 'ATR differito' if '|DATR' in str(getattr(p, 'comment', '')) else 'Ordinario',
                             'session_tag': str(getattr(p, 'comment', '')).split('|', 1)[0] if str(getattr(p, 'comment', '')).split('|', 1)[0] in ('ASIA', 'ASIA_FADE', 'LONDON', 'NY', 'NY_FADE', 'NY_REVERSAL', 'PYRAMID', 'EXTRA') else None,
                             'position_id': str(p.identifier)} for p in positions]
        # Valori osservati e ricostruzioni restano distinti dalle variabili interne dell'EA.
        bars = mt5.copy_rates_from_pos(CONFIG['symbol'], mt5.TIMEFRAME_M5, 0, 100)
        m1 = mt5.copy_rates_from_pos(CONFIG['symbol'], mt5.TIMEFRAME_M1, 0, 1800)
        d1 = mt5.copy_rates_from_pos(CONFIG['symbol'], mt5.TIMEFRAME_D1, 0, 80)
        raw_now = int(now.timestamp() + server_offset(now, params) * 3600)
        server_day = raw_now // 86400 * 86400
        last_closed = None
        if bars is not None:
            closed = [r for r in bars if int(r['time']) + 300 <= raw_now]
            if closed:
                r = closed[-1]
                body = abs(float(r['close'] - r['open']))
                width = float(r['high'] - r['low'])
                last_closed = {'open': float(r['open']), 'close': float(r['close']),
                               'body_pct': round(body / width * 100, 2) if width > 0 else 0,
                               'time_italy': italian(raw_to_utc(int(r['time']), params)).isoformat()}
        ema = previous_daily = None
        if d1 is not None and len(d1) > 21:
            ema = float(d1[0]['close'])
            for row in d1[1:-1]:
                ema += (float(row['close']) - ema) * 2 / 21
            previous_daily = float(d1[-2]['close'])
        observer = read_observer(now, params)
        sessions = []
        for name, prefix, default_hour, default_minute, default_duration in [
            ('Asia', 'Asia', 1, 0, 60), ('Londra', 'Lon', 8, 0, 60), ('New York', 'NY', 15, 30, 30)]:
            start = server_day + int(params.get(f'Inp{prefix}_StartHour', default_hour)) * 3600 + int(params.get(f'Inp{prefix}_StartMin', default_minute)) * 60
            end = start + int(params.get(f'Inp{prefix}_DurationMin', default_duration)) * 60
            deadline = end + int(params.get('InpMaxEntryDelayMin', 360)) * 60
            enabled = prefix != 'Asia' or params.get('InpUseAsia', 'true') == 'true'
            phase = 'disabled' if not enabled else 'waiting' if raw_now < start else 'forming' if raw_now < end else 'searching' if raw_now < deadline else 'ended'
            rows = [r for r in (m1 if m1 is not None else []) if start <= int(r['time']) < min(raw_now, end)]
            high = max((float(r['high']) for r in rows), default=None)
            low = min((float(r['low']) for r in rows), default=None)
            exposed = (observer.get('levels') or {}).get(name) if observer.get('available') else None
            # I livelli della giornata corrente non vengono attribuiti a finestre ancora future.
            if phase in ('waiting', 'disabled'):
                exposed = None
            sessions.append({'name': name, 'phase': phase, 'start_italy': italian(raw_to_utc(start, params)).isoformat(),
                             'end_italy': italian(raw_to_utc(end, params)).isoformat(),
                             'deadline_italy': italian(raw_to_utc(deadline, params)).isoformat(),
                             'high': high, 'low': low, 'bars_available': len(rows),
                             'expected_bars': (end - start) // 60,
                             'bot_high': exposed['high'] if exposed else None,
                             'bot_low': exposed['low'] if exposed else None,
                             'level_source': 'bot_comment' if exposed else 'reconstructed',
                             'source': 'Livelli letti dal commento del bot MT5' if exposed else 'Ricostruzione dalle candele M1; può differire dal range interno del bot'})
        # La finestra extra è visualizzata anche quando non è ancora abilitata dal capitale.
        extra_start=server_day+int(params.get('InpExtraStartHour',16))*3600+int(params.get('InpExtraStartMinute',30))*60
        extra_end=extra_start+int(params.get('InpExtraDurationMin',30))*60
        extra_deadline=extra_end+int(params.get('InpExtraEntryWindowMin',120))*60
        extra_rows=[r for r in (m1 if m1 is not None else []) if extra_start<=int(r['time'])<min(raw_now,extra_end)]
        extra_session={'name':'Extra','phase':'disabled' if params.get('InpExtraORBEnabled','true')!='true' else 'waiting' if raw_now<extra_start else 'forming' if raw_now<extra_end else 'searching' if raw_now<extra_deadline else 'ended',
                       'start_italy':italian(raw_to_utc(extra_start,params)).isoformat(),'end_italy':italian(raw_to_utc(extra_end,params)).isoformat(),
                       'deadline_italy':italian(raw_to_utc(extra_deadline,params)).isoformat(),
                       'high':max((float(r['high']) for r in extra_rows),default=None),'low':min((float(r['low']) for r in extra_rows),default=None),
                       'bars_available':len(extra_rows),'expected_bars':(extra_end-extra_start)//60,'level_source':'reconstructed'}
        if not terminal.connected:
            title, reason = 'MT5 non connesso', 'Il terminale non è collegato al broker.'
        elif not terminal.trade_allowed or not account.trade_allowed or not account.trade_expert:
            title, reason = 'Trading automatico non abilitato', 'Almeno un permesso generale del terminale o del conto non consente il trading automatico.'
        elif age > 180:
            title, reason = 'Quotazioni ferme', 'L’ultimo prezzo non è recente. Non è possibile confermare una nuova analisi del bot.'
        else:
            current = next((s for s in sessions if s['phase'] == 'forming'), None)
            searching = [s for s in sessions if s['phase'] == 'searching']
            if current:
                title = 'Formazione del range ' + current['name']
                reason = f"La finestra iniziale termina alle {current['end_italy'][11:16]} italiane. Dopo questo orario il codice valuta le chiusure M5 oltre il range."
            elif searching:
                title = 'Finestra di ingresso: ' + ', '.join(s['name'] for s in searching)
                reason = 'Il codice può valutare una chiusura M5 oltre il range, con corpo direzionale e filtri soddisfatti. I limiti interni di sessione e il motivo preciso di uno scarto non sono esposti dall’EA.'
            else:
                title = 'Attesa della prossima sessione'
                reason = 'Nessuna delle finestre principali di ingresso è attualmente aperta.'
        if own:
            reason += f' Sono aperte {len(own)} posizioni della prova298.'
        result = {**base, 'ok': True, 'connection': {'connected': terminal.connected, 'algo_enabled': terminal.trade_allowed,
                'account_trade_allowed': account.trade_allowed, 'account_ea_allowed': account.trade_expert},
                'account': {'label': 'Conto demo' if account.trade_mode == mt5.ACCOUNT_TRADE_MODE_DEMO else 'Conto reale',
                            'currency': account.currency, 'balance': account.balance, 'equity': account.equity,
                            'margin_free': account.margin_free, 'floating': account.profit,
                            'day_closed_net': round(daily.get(today, 0), 2),
                            'midnight_balance_reconstructed': round(midnight_balance, 2),
                            'protection_estimate': round(protection, 2),
                            'protection_distance_estimate': round(account.equity - protection, 2),
                            'risk_note': 'Stima dal profilo salvato e dalla cronologia: non è lo stato interno della protezione EA. Importi nella valuta del conto.'},
                'bot': {**bot, 'symbol': CONFIG['symbol'], 'magic': magic,
                        'readiness': 'Profilo e permessi generali osservati; heartbeat interno EA non disponibile',
                        'extra_enabled_after': float(params.get('InpExtraMinCapital', 20000)),
                        'body_min_percent': float(params.get('InpBodyFilterPct', 25)),
                        'rr_target': float(params.get('InpRR_Target', 6)),
                        'entry_rules': {'maximum_positions': int(params.get('InpMaximumPositions', 3)),
                                        'own_positions': len(own),
                                        'free_slots': max(0, int(params.get('InpMaximumPositions', 3)) - len(own)),
                                        'session_max_entries': int(params.get('InpSessionMaxEntries', 2)),
                                        'max_trades_day': int(params.get('InpMaxTradesDay', 12)),
                                        'max_losses_day': int(params.get('InpMaxLossesDay', 4)),
                                        'reentry_cooldown_min': int(params.get('InpReentryCooldownMin', 20)),
                                        'total_open_stop_risk_percent': float(params.get('InpTotalOpenStopRiskPct', 4)),
                                        'deferred_only_when_flat': params.get('InpDeferredOnlyWhenFlat', 'true').lower() == 'true',
                                        'primary_entry_priority': params.get('InpPrimaryEntryPriority', 'true').lower() == 'true',
                                        'normal_session_entry_requires_flat': False,
                                        'source': 'Profilo salvato e codice prova298; disponibilità operativa non garantita.'},
                        'atr_exact': None, 'atr_note': 'Il valore ATR interno del bot non è esposto dall’API Python.'},
                'quote': {'bid': tick.bid, 'ask': tick.ask, 'time_italy': italian(quote_utc).isoformat(), 'age_seconds': round(age, 1)},
                'analysis': {'title': title, 'reason': reason, 'source': 'Ricostruzione dal codice e dai dati osservati',
                             'bot_observer': observer,
                             'sessions': sessions, 'last_closed_m5': last_closed,
                             'previous_daily_close': previous_daily, 'ema20_reconstructed': ema,
                             'short_ema_compatible': previous_daily < ema if ema is not None else None},
                'positions': actual_positions, 'pending_orders': len(orders),
                'candles': [{'time_raw': int(r['time']), 'time_italy': italian(raw_to_utc(int(r['time']), params)).isoformat(),
                             **{key: float(r[key]) for key in ('open', 'high', 'low', 'close')}} for r in (bars if bars is not None else [])],
                'deals': converted[-200:], 'journal_deals': converted, 'history_from': history_start.date().isoformat(), 'history_days': (now-history_start).days,
                'daily': [{'date': day, 'net': round(pnl, 2)} for day, pnl in sorted(daily.items())],
                'limitations': ['Nessun ordine o comando al terminale.',
                               'Stato interno, ATR e motivi di scarto non letti direttamente dall’EA.',
                               'Cronologia del conto richiesta dal 2025; disponibilità dipendente dal broker. Non è il backtest storico.',
                               'I risultati del conto includono eventuali altre strategie.']}
        risk=risk_observations(mt5,account,positions,orders,params,CONFIG['symbol'],own,raw_now)
        result['analysis']['entry_criteria']=build_criteria(result,params,raw_now,bars,d1,risk,mt5,extra_session)
        return result
    finally:
        mt5.shutdown()


if __name__ == '__main__':
    try:
        print(json.dumps(read_snapshot(), ensure_ascii=False, allow_nan=False))
    except Exception as exc:
        print(json.dumps({'ok': False, 'mode': 'mt5_read_only', 'orders_enabled': False,
                          'error': 'Lettura momentaneamente non disponibile (' + type(exc).__name__ + ').'}))
