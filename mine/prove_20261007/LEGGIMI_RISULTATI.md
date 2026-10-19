# Prove ORB del 7 ottobre 2026

Aggiornamento: 2026-10-07 18:22. Prove registrate: 130; test completati: 130.

## Condizioni identiche

XAUUSD; timeframe tester M5; 01/01/2025-27/09/2026; 10.000 EUR; leva 1:33; modello MT5 ogni tick basato su tick reali; parametri predefiniti specifici di ogni file.

Il conto storico MetaQuotes-Demo ha tick reali disponibili dal 27/05/2025. MT5 indica 76% di tick reali sul periodo intero e genera i tick dove mancano. I risultati includono i costi disponibili nel tester del broker.

## Migliori prove entro 15% di drawdown di equity

Per il confronto si usa il maggiore fra il drawdown percentuale associato alla massima perdita monetaria e il drawdown relativo di equity del report MT5.

| File | Profitto EUR | Saldo finale EUR | Nuove posizioni | Giorni con ingressi | DD relativo | Profit factor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| prova01.mq5 | 5.367,83 | 15.367,83 | 63 | 60 | 6.50% | 2.03 |
| prova130.mq5 | 56.777,30 | 66.777,30 | 196 | 151 | 14.02% | 2.06 |
| prova129.mq5 | 53.814,09 | 63.814,09 | 195 | 150 | 13.84% | 2.05 |
| prova128.mq5 | 53.350,77 | 63.350,77 | 194 | 149 | 14.54% | 2.01 |
| prova125.mq5 | 52.755,72 | 62.755,72 | 195 | 150 | 13.56% | 2.10 |
| prova126.mq5 | 49.870,98 | 59.870,98 | 190 | 148 | 14.23% | 2.04 |
| prova112.mq5 | 45.414,29 | 55.414,29 | 199 | 150 | 14.47% | 2.03 |
| prova120.mq5 | 43.989,14 | 53.989,14 | 200 | 150 | 13.75% | 2.01 |
| prova119.mq5 | 43.946,76 | 53.946,76 | 200 | 150 | 14.11% | 2.02 |
| prova113.mq5 | 43.871,27 | 53.871,27 | 198 | 149 | 14.08% | 2.00 |
| prova111.mq5 | 43.223,00 | 53.223,00 | 200 | 150 | 14.84% | 1.98 |

Migliore: **prova130**. Obiettivo storico di 60.000 EUR raggiunto.

## Lettura dei risultati

La colonna operazioni del report MT5 conta anche le chiusure parziali. Le nuove posizioni sono contate dagli effettivi ingressi; gli incrementi protetti sono posizioni aggiuntive sullo stesso movimento, non nuovi segnali indipendenti.

Le prove selezionate hanno piu ingressi del bot originale; non viene garantita almeno una operazione ogni giorno.

Tutte le varianti sono state confrontate sullo stesso periodo. Il periodo e stato usato per scegliere i parametri: questi risultati non sono una verifica su dati mai osservati e non dimostrano i rendimenti futuri. Il limite del 15% e un criterio di selezione storico, non una garanzia in reale. I file con gestione adaptive dichiarata includono una loro soglia di arresto specifica; gli altri non hanno un arresto permanente del drawdown totale.

## Conservazione e tracciabilita

Il sorgente e il compilato originali sono rimasti invariati. Ogni numero identifica un file separato con parametri predefiniti, compilazione e risultato; prova28 non ha compilato ed e riportata senza metriche inventate.

I risultati sono nel file Desktop/Confronto_Bot_Trading.xlsx, dopo tre righe vuote rispetto alla tabella originale. La colonna DD relativo% permette di controllare il limite concordato.

Backup originali: C:\Users\Marco\Documents\Codex\2026-09-28\anali\work\research_20261007\backup
Report MT5 e grafici: C:\Users\Marco\Documents\Codex\2026-09-28\anali\work\research_20261007\reports
CSV con una riga per posizione, disponibili da prova85: C:\Users\Marco\Documents\Codex\2026-09-28\anali\work\research_20261007\trades
Parametri, descrizioni e impronte dei file: C:\Users\Marco\Documents\Codex\2026-09-28\anali\work\research_20261007
