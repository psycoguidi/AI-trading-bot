"""Server personale su loopback. API MT5 di sola lettura, accesso remoto con identità."""
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import mimetypes
from pathlib import Path
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import unquote, urlsplit, parse_qs
import ai_chat
import journal
import push_notifications

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / '07_collegamento/config.json').read_text(encoding='utf-8'))
DATA = ROOT / '06_dati'
LOCK = threading.Lock()
CHART_LOCK = threading.Lock()
JOBS_LOCK = threading.Lock()
JOBS = {}
SNAPSHOT = {'ok': False, 'mode': 'mt5_read_only', 'orders_enabled': False, 'error': 'Primo collegamento a MT5 in corso.'}


def chat_worker(job_id, question, history, snapshot):
    try:
        def progress(partial):
            with JOBS_LOCK:
                JOBS[job_id]['partial_text'] = partial
        result = ai_chat.answer(question, history, snapshot, on_chunk=progress)
        with JOBS_LOCK:
            JOBS[job_id].update(status='done', result=result)
    except Exception as exc:
        with JOBS_LOCK:
            JOBS[job_id].update(status='error', error='Il modello locale non ha risposto. Verifica che Ollama sia aperto sul PC e riprova.')
        print('Chat AI: ' + type(exc).__name__, flush=True)
    finally:
        ai_chat.GENERATING.release()


def remote_config():
    try:
        cfg = json.loads((DATA / 'accesso_privato.json').read_text(encoding='utf-8'))
        if cfg.get('enabled') is not True:
            return None
        if not re.fullmatch(r'[a-z0-9][a-z0-9.-]*\.ts\.net:8444', cfg.get('host', '')):
            return None
        if not isinstance(cfg.get('owner_login'), str) or not cfg['owner_login'] or len(cfg['owner_login']) > 320:
            return None
        return cfg
    except (OSError, ValueError, TypeError):
        return None


def allowed_request(host, origin, identity, peer, port):
    try:
        if not ipaddress.ip_address(peer).is_loopback:
            return False
        parsed = urlsplit('http://' + host)
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            return False
        local = parsed.hostname in ('127.0.0.1', 'localhost') and parsed.port == port
        cfg = remote_config()
        remote = bool(cfg and host == cfg['host'] and identity == cfg['owner_login'])
        if not local and not remote:
            return False
        if origin and origin != ('https://' if remote else 'http://') + host:
            return False
        return True
    except (ValueError, TypeError):
        return False


def collector():
    global SNAPSHOT
    while True:
        try:
            result = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT / '07_collegamento/mt5_readonly.py')],
                                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=20,
                                    creationflags=0x08000000 if sys.platform == 'win32' else 0)
            data = json.loads(result.stdout)
            data['received_utc'] = datetime.now(timezone.utc).isoformat()
            try:
                journal.ingest(data)
            except Exception as exc:
                print('Archivio: ' + type(exc).__name__, flush=True)
            data.pop('journal_deals', None)
            with LOCK:
                SNAPSHOT = data
        except Exception as exc:
            with LOCK:
                SNAPSHOT = {'ok': False, 'mode': 'mt5_read_only', 'orders_enabled': False,
                            'error': 'Collegamento momentaneamente non disponibile.',
                            'received_utc': datetime.now(timezone.utc).isoformat()}
            print('Lettura MT5: ' + type(exc).__name__, flush=True)
        time.sleep(5)


class Handler(BaseHTTPRequestHandler):
    server_version = 'QuantixIQ/0.4'

    def authorized(self):
        return allowed_request(self.headers.get('Host', ''), self.headers.get('Origin'),
                               self.headers.get('Tailscale-User-Login'), self.client_address[0], self.server.server_port)

    def response(self, status, body, kind='application/json; charset=utf-8', head=False):
        self.send_response(status)
        self.send_header('Content-Type', kind)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store' if self.path.startswith('/api/') else 'no-cache')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'same-origin')
        self.send_header('Permissions-Policy', 'camera=(), microphone=(), geolocation=()')
        self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self' data:; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'")
        self.end_headers()
        if not head:
            self.wfile.write(body)

    def json_response(self, status, data, head=False):
        self.response(status, json.dumps(data, ensure_ascii=False, allow_nan=False).encode('utf-8'), head=head)

    def read(self, head=False):
        if not self.authorized():
            self.json_response(403, {'error': 'Accesso privato non autorizzato.'}, head)
            return
        path = unquote(urlsplit(self.path).path)
        if path.startswith('/api/') and self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.json_response(403, {'error': 'Origine non autorizzata.'}, head)
            return
        if path == '/':
            self.send_response(302)
            self.send_header('Location', '/01_app/')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return
        if path == '/api/health':
            cfg = remote_config()
            self.json_response(200, {'service': 'quantix-iq', 'version': '0.4.0', 'orders_enabled': False,
                                    'private_access_enabled': bool(cfg),
                                    'private_url': 'https://' + cfg['host'] + '/01_app/' if cfg else None}, head)
            return
        if path == '/api/mt5':
            if self.headers.get('Sec-Fetch-Site') == 'cross-site':
                self.json_response(403, {'error': 'Origine non autorizzata.'}, head)
                return
            with LOCK:
                data = dict(SNAPSHOT)
            self.json_response(200, data, head)
            return
        if path in ('/api/profile','/api/calendar','/api/day','/api/journal','/api/journal/trade','/api/push/status','/api/archive/export'):
            query = parse_qs(urlsplit(self.path).query)
            try:
                if any(len(v)!=1 for v in query.values()):
                    raise ValueError('Parametri non validi.')
                origin = query.get('origin',['all'])[0]
                if origin not in ('all','bot','other'):
                    raise ValueError('Filtro non valido.')
                allowed = {'/api/calendar':{'month','origin'},'/api/day':{'day','origin'},'/api/journal':{'origin'},'/api/journal/trade':{'id'}}.get(path,set())
                if set(query)-allowed:
                    raise ValueError('Parametri non validi.')
                if path == '/api/profile': value = journal.account_data()
                elif path == '/api/calendar': value = journal.calendar_data(query.get('month',[''])[0],origin)
                elif path == '/api/day': value = journal.day_data(query.get('day',[''])[0],origin)
                elif path == '/api/journal': value = journal.journal_data(origin)
                elif path == '/api/journal/trade': value = journal.trade_data(query.get('id',[''])[0])
                elif path == '/api/push/status': value = push_notifications.status()
                else:
                    items = journal.journal_data(limit=100000)
                    value = {'exported_utc':datetime.now(timezone.utc).isoformat(),'account':journal.account_data(),
                             'trades':[journal.trade_data(t['id']) for t in items['trades']], 'orders_enabled':False}
                self.json_response(200,value,head)
            except ValueError as exc:
                self.json_response(400,{'error':str(exc)},head)
            except Exception as exc:
                print('Archivio API: '+type(exc).__name__,flush=True)
                self.json_response(503,{'error':'Archivio momentaneamente non disponibile.'},head)
            return
        if path == '/api/ai':
            self.json_response(200, ai_chat.model_status(), head)
            return
        if path.startswith('/api/chat/'):
            job_id = path.removeprefix('/api/chat/')
            with JOBS_LOCK:
                job = dict(JOBS.get(job_id, {}))
            self.json_response(200 if job else 404, job or {'error': 'Conversazione non disponibile.'}, head)
            return
        if path == '/api/chart':
            query = parse_qs(urlsplit(self.path).query)
            try:
                if set(query) - {'tf', 'count', 'before'} or any(len(v) != 1 for v in query.values()):
                    raise ValueError()
                tf = query.get('tf', ['M5'])[0]
                count = int(query.get('count', ['300'])[0])
                before = int(query['before'][0]) if 'before' in query else None
                if tf not in ('M1', 'M5', 'M15', 'M30', 'H1', 'H4', 'D1') or not 20 <= count <= 500 or before is not None and not 946684800 <= before <= 4102444800:
                    raise ValueError()
            except ValueError:
                self.json_response(400, {'error': 'Timeframe o intervallo non valido.'}, head)
                return
            if head:
                self.json_response(200, {'ok': True}, True)
                return
            if not CHART_LOCK.acquire(blocking=False):
                self.json_response(429, {'error': 'Lettura storico in corso. Riprova tra poco.'})
                return
            try:
                command = [sys.executable, '-X', 'utf8', str(ROOT / '07_collegamento/chart_history.py'), '--tf', tf, '--count', str(count)]
                if before is not None:
                    command += ['--before', str(before)]
                result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', timeout=22,
                                        creationflags=0x08000000 if sys.platform == 'win32' else 0)
                data = json.loads(result.stdout)
                self.json_response(200 if data.get('ok') else 503, data)
            except Exception:
                self.json_response(503, {'error': 'Lo storico non risponde ancora. Riprova tra poco.'})
            finally:
                CHART_LOCK.release()
            return
        if path.startswith('/api/'):
            self.json_response(404, {'error': 'Funzione non disponibile.'}, head)
            return
        if path == '/01_app/':
            path += 'index.html'
        permitted = path.startswith('/01_app/') or path == '/02_immagini/logo/logo-app.png'
        file = (ROOT / path.lstrip('/')).resolve()
        if not permitted or not file.is_relative_to(ROOT / '01_app') and file != ROOT / '02_immagini/logo/logo-app.png':
            self.json_response(404, {'error': 'Risorsa non disponibile.'}, head)
            return
        if not file.is_file() or file.suffix.lower() not in ('.html', '.css', '.js', '.png', '.webmanifest', '.txt'):
            self.json_response(404, {'error': 'Risorsa non disponibile.'}, head)
            return
        kind = 'application/manifest+json' if file.suffix == '.webmanifest' else mimetypes.guess_type(file.name)[0] or 'application/octet-stream'
        if kind.startswith('text/') or file.suffix == '.js':
            kind += '; charset=utf-8'
        self.response(200, file.read_bytes(), kind, head)

    def do_GET(self):
        self.read()

    def do_HEAD(self):
        self.read(True)

    def do_POST(self):
        if not self.authorized() or self.headers.get('Sec-Fetch-Site') == 'cross-site':
            self.json_response(403, {'error': 'Accesso privato non autorizzato.'})
            return
        personal_path = urlsplit(self.path).path
        if personal_path in ('/api/profile','/api/journal/note','/api/push/subscribe','/api/push/unsubscribe','/api/push/test'):
            try:
                length=int(self.headers.get('Content-Length','0'))
                if self.headers.get('Transfer-Encoding') or not 0<length<=16384 or self.headers.get('Content-Type','').split(';')[0]!='application/json':
                    raise ValueError('Formato o dimensione della richiesta non validi.')
                self.connection.settimeout(8)
                payload=json.loads(self.rfile.read(length))
                if not isinstance(payload,dict): raise ValueError('Dati non validi.')
                if personal_path=='/api/profile': value={'ok':True,'profile':journal.save_profile(payload)}
                elif personal_path=='/api/journal/note': value=journal.save_note(str(payload.get('id','')),payload.get('note'))
                elif personal_path=='/api/push/subscribe': value=push_notifications.subscribe(payload)
                elif personal_path=='/api/push/unsubscribe': value=push_notifications.unsubscribe(payload.get('endpoint'))
                else: value=push_notifications.test(payload.get('endpoint'))
                self.json_response(200,value)
            except (ValueError,TypeError,OSError) as exc:
                self.json_response(400,{'error':str(exc)[:200]})
            except Exception as exc:
                print('Archivio salvataggio: '+type(exc).__name__,flush=True)
                self.json_response(503,{'error':'Salvataggio non riuscito. Riprova.'})
            return
        if urlsplit(self.path).path != '/api/chat':
            self.json_response(405, {'error': 'Nessun comando di trading disponibile.'})
            return
        try:
            length = int(self.headers.get('Content-Length', '0'))
            if self.headers.get('Transfer-Encoding') or not 0 < length <= 24576:
                self.json_response(413, {'error': 'Domanda troppo lunga.'})
                return
            if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                self.json_response(415, {'error': 'Formato della domanda non valido.'})
                return
            self.connection.settimeout(8)
            data = json.loads(self.rfile.read(length))
            question, history = data.get('question'), data.get('history', [])
            if not isinstance(question, str) or not question.strip() or len(question) > 4000 or not isinstance(history, list) or len(history) > 12:
                raise ValueError()
        except (ValueError, OSError, AttributeError):
            self.json_response(400, {'error': 'Domanda non valida.'})
            return
        if not ai_chat.GENERATING.acquire(blocking=False):
            self.json_response(429, {'error': 'L’AI sta rispondendo a una domanda. Attendi la risposta e riprova.'})
            return
        job_id = secrets.token_urlsafe(18)
        with LOCK:
            snapshot = dict(SNAPSHOT)
        with JOBS_LOCK:
            for key in list(JOBS):
                if JOBS[key]['status'] != 'pending' and (len(JOBS) >= 6 or time.time() - JOBS[key]['created'] > 900):
                    del JOBS[key]
            JOBS[job_id] = {'status': 'pending', 'model': ai_chat.MODEL, 'created': time.time()}
        threading.Thread(target=chat_worker, args=(job_id, question.strip(), history, snapshot), daemon=True).start()
        self.json_response(202, {'job_id': job_id, 'model': ai_chat.MODEL, 'status': 'pending'})

    def no_write(self):
        self.json_response(405, {'error': 'Nessun comando di trading disponibile.'})

    do_PUT = do_PATCH = do_DELETE = no_write

    def log_message(self, format, *args):
        # Non registrare cronologia, parametri, nomi del conto o identità del telefono.
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=CONFIG['port'])
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    threading.Thread(target=collector, daemon=True).start()
    threading.Thread(target=ai_chat.warmup, daemon=True).start()
    threading.Thread(target=push_notifications.worker, daemon=True).start()
    DATA.mkdir(exist_ok=True)
    (DATA / 'server.json').write_text(json.dumps({'pid': __import__('os').getpid(), 'port': args.port,
                                                'started_utc': datetime.now(timezone.utc).isoformat()}), encoding='utf-8')
    print(f'QUANTIX IQ http://127.0.0.1:{args.port}/01_app/ — lettura MT5 soltanto', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
