"""Web Push verso i dispositivi personali iscritti. Nessun canale pubblico."""
import base64
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlsplit
import journal

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '07_collegamento/dependencies'))
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec

KEY_FILE = ROOT / '06_dati/webpush_private.pem'
ALLOWED_HOSTS = {'fcm.googleapis.com', 'updates.push.services.mozilla.com', 'web.push.apple.com'}
TEST_LIMIT = {}
RATE_LOCK = threading.Lock()


def b64(value):
    return base64.urlsafe_b64encode(value).rstrip(b'=').decode()


def public_key():
    with journal.GUARD:
        if not KEY_FILE.exists():
            key = ec.generate_private_key(ec.SECP256R1())
            KEY_FILE.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                                  serialization.NoEncryption()))
        key = serialization.load_pem_private_key(KEY_FILE.read_bytes(), password=None)
        return b64(key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint))


def validate_subscription(value, resolve=False):
    if not isinstance(value, dict) or not isinstance(value.get('endpoint'), str):
        raise ValueError('Iscrizione notifiche non valida.')
    endpoint = value['endpoint']
    if len(endpoint) > 2048:
        raise ValueError('Indirizzo notifiche non valido.')
    parsed = urlsplit(endpoint)
    if (parsed.scheme != 'https' or parsed.hostname not in ALLOWED_HOSTS or parsed.port not in (None, 443)
            or parsed.username or parsed.password or parsed.fragment or not parsed.path or parsed.path == '/'):
        raise ValueError('Il servizio push del browser non è supportato.')
    keys = value.get('keys', {})
    if not isinstance(keys, dict):
        raise ValueError('Chiavi notifiche non valide.')
    for name, size in [('p256dh', 65), ('auth', 16)]:
        text = keys.get(name)
        if not isinstance(text, str) or len(text) > 150:
            raise ValueError('Chiavi notifiche non valide.')
        try:
            decoded = base64.urlsafe_b64decode(text + '=' * (-len(text) % 4))
            if len(decoded) != size:
                raise ValueError()
            if name == 'p256dh':
                ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), decoded)
        except Exception:
            raise ValueError('Chiavi notifiche non valide.')
    if resolve:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ValueError('Destinazione push non consentita.')
    return {'endpoint': endpoint, 'keys': {k: keys[k] for k in ('p256dh', 'auth')}}


def subscribe(value):
    clean = validate_subscription(value)
    sid = hashlib.sha256(clean['endpoint'].encode()).hexdigest()
    with journal.GUARD, journal.connect() as db:
        existing = db.execute('SELECT enabled FROM subscriptions WHERE id=?', (sid,)).fetchone()
        if not existing and db.execute('SELECT count(*) FROM subscriptions WHERE enabled=1').fetchone()[0] >= 10:
            raise ValueError('Sono già iscritti 10 dispositivi. Disattiva quelli non utilizzati.')
        if existing and existing[0]:
            db.execute('UPDATE subscriptions SET data=? WHERE id=?', (journal.dumps(clean), sid))
        else:
            db.execute('INSERT OR REPLACE INTO subscriptions VALUES (?,?,?,1,NULL)', (sid, journal.dumps(clean), journal.now()))
    return {'ok': True, 'subscription_id': sid}


def unsubscribe(value):
    if not isinstance(value, str) or len(value) > 2048:
        raise ValueError('Iscrizione non valida.')
    sid = hashlib.sha256(value.encode()).hexdigest()
    with journal.GUARD, journal.connect() as db:
        db.execute('UPDATE subscriptions SET enabled=0 WHERE id=?', (sid,))
    return {'ok': True}


def status():
    public = public_key()
    try:
        import pywebpush
        available = True
    except ImportError:
        available = False
    with journal.GUARD, journal.connect() as db:
        recent = [dict(r) for r in db.execute('SELECT last_status,created FROM subscriptions WHERE enabled=1')]
        count = len(recent)
    return {'ok': True, 'available': available, 'public_key': public, 'devices': count,
            'last_status': [r['last_status'] for r in recent if r['last_status']],
            'note': 'Notifiche anche con app chiusa. Richiedono consenso sul dispositivo, Internet e PC con servizio QUANTIX IQ avviato. La consegna è gestita dal servizio push del browser.'}


def test(endpoint):
    sid = hashlib.sha256(str(endpoint).encode()).hexdigest()
    with RATE_LOCK:
        if time.time() - TEST_LIMIT.get(sid, 0) < 30:
            raise ValueError('Attendi 30 secondi prima di riprovare la notifica.')
        TEST_LIMIT[sid] = time.time()
    with journal.GUARD, journal.connect() as db:
        if not db.execute('SELECT 1 FROM subscriptions WHERE id=? AND enabled=1', (sid,)).fetchone():
            raise ValueError('Attiva prima le notifiche su questo dispositivo.')
        key = 'test:' + sid + ':' + journal.now()
        journal.notice(db, key, 'test', 'Collegamento notifiche QUANTIX IQ attivo. Questa è una prova.')
        db.execute('INSERT OR IGNORE INTO deliveries(notice_id,subscription_id) VALUES (?,?)', (key, sid))
    return {'ok': True, 'message': 'Notifica di prova accodata. Verifica la ricezione sul telefono.'}


def deliver(subscription, payload):
    from pywebpush import webpush
    import requests
    validate_subscription(subscription, resolve=True)

    class NoRedirects(requests.Session):
        def post(self, url, **kwargs):
            kwargs['allow_redirects'] = False
            return super().post(url, **kwargs)

    cfg = json.loads((ROOT / '06_dati/accesso_privato.json').read_text(encoding='utf-8'))
    # Il contatto VAPID è un URI HTTPS del responsabile, senza porta/percorso.
    # Il validatore della libreria non accetta un indirizzo applicativo con :8444.
    subject = 'https://' + urlsplit('https://' + cfg['host']).hostname
    with NoRedirects() as session:
        response = webpush(subscription_info=subscription, data=journal.dumps(payload),
                           vapid_private_key=str(KEY_FILE), vapid_claims={'sub': subject},
                           ttl=1800, timeout=10, requests_session=session, headers={'Urgency': 'normal'})
    if response.status_code not in (200, 201, 202):
        raise ValueError('Consegna non accettata dal servizio push.')


def once():
    prefs = journal.profile()
    with journal.GUARD, journal.connect() as db:
        db.execute('''INSERT OR IGNORE INTO deliveries(notice_id,subscription_id)
                      SELECT n.id,s.id FROM notices n JOIN subscriptions s ON s.enabled=1 AND n.created>=s.created
                      WHERE n.kind<>'test' ''')
        rows = db.execute('''SELECT d.notice_id,d.subscription_id,d.attempts,n.kind,n.payload,n.created,s.data
                             FROM deliveries d JOIN notices n ON n.id=d.notice_id JOIN subscriptions s ON s.id=d.subscription_id
                             WHERE d.status='pending' AND d.attempts<3 AND d.next_try<=? AND s.enabled=1
                             ORDER BY n.created LIMIT 8''', (time.time(),)).fetchall()
    for row in rows:
        state = 'accepted'
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(row['created'])).total_seconds()
            if age > 1800:
                state = 'expired'
            elif row['kind'] != 'test' and not prefs.get('notify_' + row['kind'], False):
                state = 'disabled'
            else:
                payload = json.loads(row['payload'])
                if prefs['notify_amounts'] and payload.get('amount') is not None:
                    payload['body'] += f" Importo: {payload['amount']:.2f} {payload.get('currency', '')}."
                payload.pop('amount', None)
                payload.pop('currency', None)
                payload['tag'] = row['notice_id']
                deliver(json.loads(row['data']), payload)
        except Exception as exc:
            code = getattr(getattr(exc, 'response', None), 'status_code', None)
            state = 'invalid' if code in (404, 410) else 'pending' if row['attempts'] < 2 else 'failed'
        with journal.GUARD, journal.connect() as db:
            db.execute('UPDATE deliveries SET attempts=attempts+1,next_try=?,status=? WHERE notice_id=? AND subscription_id=?',
                       (time.time() + 30 * (row['attempts'] + 1), state, row['notice_id'], row['subscription_id']))
            label = {'accepted': 'Accettata dal servizio push', 'pending': 'Nuovo tentativo in attesa',
                     'failed': 'Invio non riuscito', 'invalid': 'Iscrizione scaduta', 'expired': 'Avviso scaduto', 'disabled': 'Categoria disattivata'}[state]
            db.execute('UPDATE subscriptions SET last_status=?,enabled=CASE WHEN ?=\'invalid\' THEN 0 ELSE enabled END WHERE id=?',
                       (label, state, row['subscription_id']))


def worker():
    public_key()
    while True:
        try:
            once()
        except Exception as exc:
            print('Notifiche: ' + type(exc).__name__, flush=True)
        time.sleep(3)
