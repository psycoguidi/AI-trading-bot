"""Avvia soltanto il servizio QUANTIX IQ, senza avviare o modificare MT5."""
import json
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen
import webbrowser

ROOT = Path(__file__).resolve().parent.parent
CONFIG = json.loads((ROOT / '07_collegamento/config.json').read_text(encoding='utf-8'))


def running():
    try:
        with urlopen(f"http://127.0.0.1:{CONFIG['port']}/api/health", timeout=2) as response:
            return json.load(response).get('service') == 'quantix-iq'
    except Exception:
        return False


def start(show=True):
    # Avvio locale senza finestre del motore AI già installato; nessun modello scaricato.
    try:
        with urlopen('http://127.0.0.1:11434/api/tags', timeout=2) as response:
            json.load(response)
    except Exception:
        ollama = Path.home() / 'AppData/Local/Programs/Ollama/ollama.exe'
        if ollama.is_file():
            with (ROOT / '05_verifiche/ollama_servizio.log').open('ab') as output:
                subprocess.Popen([str(ollama), 'serve'], stdout=output, stderr=output,
                                 creationflags=0x08000000 if sys.platform == 'win32' else 0)
    if not running():
        logs = ROOT / '05_verifiche'
        with (logs / 'servizio.log').open('ab') as output:
            subprocess.Popen([CONFIG['python_runtime'], str(ROOT / '07_collegamento/server.py')], cwd=ROOT,
                             stdout=output, stderr=output, creationflags=0x08000000 if sys.platform == 'win32' else 0)
        for _ in range(40):
            if running():
                break
            time.sleep(.2)
        else:
            raise RuntimeError('Servizio non avviato. Consulta 05_verifiche/servizio.log.')
    url = f"http://127.0.0.1:{CONFIG['port']}/01_app/"
    if show:
        webbrowser.open(url)
    return url


if __name__ == '__main__':
    print(start('--no-browser' not in sys.argv))
