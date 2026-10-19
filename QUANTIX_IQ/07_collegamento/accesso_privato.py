"""Configura esclusivamente QUANTIX IQ in Tailscale Serve; nessun Funnel pubblico."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from avvio import start

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / '06_dati/accesso_privato.json'
PORT = 8444
TARGET = 'http://127.0.0.1:8876'
EXE = Path(os.environ.get('ProgramFiles', r'C:\Program Files')) / 'Tailscale/tailscale.exe'


def command(*args, timeout=30):
    result = subprocess.run([str(EXE), *args], capture_output=True, text=True, encoding='utf-8', errors='replace',
                            timeout=timeout, creationflags=0x08000000)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout).strip()[:1800])
    return result.stdout


def write(value):
    temporary = FILE.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(FILE)


def own(state, host):
    return (state.get('Web') or {}).get(host, {}).get('Handlers', {}).get('/', {}).get('Proxy') == TARGET


def identity():
    state = json.loads(command('status', '--json'))
    if state.get('BackendState') != 'Running':
        raise RuntimeError('Completa l’accesso a Tailscale sul PC, poi riprova.')
    node = state.get('Self') or {}
    host = (node.get('DNSName') or '').rstrip('.').lower()
    user = (state.get('User') or {}).get(str(node.get('UserID')), {})
    owner = user.get('LoginName')
    if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*\.ts\.net', host) or not owner:
        raise RuntimeError('Serve un dispositivo personale con identità Tailscale e MagicDNS.')
    return host + ':' + str(PORT), owner


def compatible(state, host):
    handlers = (state.get('Web') or {}).get(host, {}).get('Handlers', {})
    if ((state.get('TCP') or {}).get(str(PORT)) or handlers) and not own(state, host):
        raise RuntimeError('La porta privata 8444 è occupata da un altro servizio: configurazione conservata.')
    if handlers and set(handlers) != {'/'}:
        raise RuntimeError('La porta contiene altri percorsi: configurazione conservata.')
    if (state.get('AllowFunnel') or {}).get(host):
        raise RuntimeError('Funnel è attivo sulla porta scelta. Il collegamento privato non viene avviato.')


def connect():
    host, owner = identity()
    previous = json.loads(command('serve', 'status', '--json'))
    compatible(previous, host)
    write({'enabled': False, 'host': host, 'owner_login': owner})
    start(False)
    output = command('serve', '--bg', '--https=' + str(PORT), '--yes', TARGET)
    state = json.loads(command('serve', 'status', '--json'))
    compatible(state, host)
    if not own(state, host):
        raise RuntimeError('Il proxy non è stato verificato. Accesso remoto ancora disabilitato.')
    write({'enabled': True, 'host': host, 'owner_login': owner})
    url = 'https://' + host + '/01_app/'
    (ROOT / 'INDIRIZZO_TELEFONO.txt').write_text(url + '\n\nAndroid: Tailscale connesso allo stesso account del PC, poi apri il link in Chrome e scegli Installa app.\nIl PC, MT5 e QUANTIX IQ devono restare avviati per avere dati aggiornati.\n', encoding='utf-8')
    print(json.dumps({'private_url': url, 'verified_proxy': True, 'public_funnel': False}))
    if output.strip():
        print(output.strip())


def disable():
    cfg = json.loads(FILE.read_text(encoding='utf-8'))
    cfg['enabled'] = False
    write(cfg)
    state = json.loads(command('serve', 'status', '--json'))
    compatible(state, cfg['host'])
    if own(state, cfg['host']):
        command('serve', '--https=' + str(PORT), 'off')
    print('Accesso privato QUANTIX IQ disabilitato. Il bot MT5 continua normalmente.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['connect', 'disable', 'status'])
    args = parser.parse_args()
    try:
        if args.action == 'connect':
            connect()
        elif args.action == 'disable':
            disable()
        else:
            host, owner = identity()
            state = json.loads(command('serve', 'status', '--json'))
            print(json.dumps({'tailscale': 'Running', 'proxy_matches': own(state, host),
                              'public_funnel': bool((state.get('AllowFunnel') or {}).get(host))}))
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as exc:
        print('Accesso non completato: ' + str(exc), file=sys.stderr)
        sys.exit(1)
