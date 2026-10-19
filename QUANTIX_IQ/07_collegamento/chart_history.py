"""Candele reali paginate del solo simbolo configurato. Nessun comando di trading."""
import argparse
from datetime import datetime
import json
import sys
from mt5_readonly import CONFIG, UTC, terminal_running, params_from_profile, raw_to_utc, italian

TIMEFRAMES = ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1')


def read_chart(tf='M5', count=300, before=None):
    if tf not in TIMEFRAMES or not 20 <= count <= 500:
        raise ValueError('Parametri grafico non validi.')
    if before is not None and not 946684800 <= before <= 4102444800:
        raise ValueError('Data non valida.')
    if not terminal_running():
        return {'ok': False, 'error': 'Apri MT5 sul PC per leggere lo storico.'}
    params, _ = params_from_profile()
    sys.path.insert(0, CONFIG['python_dependencies'])
    import MetaTrader5 as mt5
    if not mt5.initialize(CONFIG['terminal_path'], timeout=5000):
        return {'ok': False, 'error': 'MT5 non risponde alla lettura dello storico.'}
    try:
        account = mt5.account_info()
        if not account or account.login != CONFIG['expected_login'] or account.server != CONFIG['expected_server']:
            return {'ok': False, 'error': 'Il conto aperto non è quello configurato.'}
        timeframe = getattr(mt5, 'TIMEFRAME_' + tf)
        # Il cursore usa il timestamp grezzo del broker, senza riconversioni ambigue.
        if before is None:
            rates = mt5.copy_rates_from_pos(CONFIG['symbol'], timeframe, 0, count)
        else:
            rates = mt5.copy_rates_from(CONFIG['symbol'], timeframe, datetime.fromtimestamp(before - 1, UTC), count)
        if rates is None:
            return {'ok': False, 'error': 'Storico non ancora disponibile in MT5. Riprova tra poco.'}
        candles = [{'time_raw': int(r['time']),
                    'time_italy': italian(raw_to_utc(int(r['time']), params)).isoformat(),
                    **{key: float(r[key]) for key in ('open', 'high', 'low', 'close')}}
                   for r in rates if before is None or int(r['time']) < before]
        return {'ok': True, 'symbol': CONFIG['symbol'], 'timeframe': tf, 'candles': candles,
                'oldest_cursor': candles[0]['time_raw'] if candles else None,
                'has_more': len(candles) == count, 'orders_enabled': False,
                'source': 'Candele del broker lette da MT5; disponibilità limitata allo storico del terminale.'}
    finally:
        mt5.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tf', choices=TIMEFRAMES, default='M5')
    parser.add_argument('--count', type=int, default=300)
    parser.add_argument('--before', type=int)
    args = parser.parse_args()
    try:
        print(json.dumps(read_chart(args.tf, args.count, args.before), allow_nan=False, ensure_ascii=False))
    except Exception:
        print(json.dumps({'ok': False, 'error': 'Lettura dello storico momentaneamente non disponibile.'}))
