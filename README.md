# AI Trading Bot · QUANTIX IQ

Archivio dei bot MetaTrader 5, delle prove e dei risultati del progetto. Aggiornamento: **19 ottobre 2026**.

## Da dove iniziare

- **Bot di riferimento su XAUUSD M5:** [prova298.mq5](mine/prove_ATR_20261016/prova298.mq5) · [compilato EX5](mine/prove_ATR_20261016/prova298.ex5).
- **Installazione e avvio su MetaTrader:** [guida completa](docs/INSTALLAZIONE_MT5.md).
- **Parametri e limiti della prova298:** [scheda del bot](docs/PROVA298.md).
- **Confronto sintetico delle versioni selezionate:** [confronto_finale.xlsx](risultati/confronto_finale.xlsx).
- **Tutti i risultati registrati:** [Confronto_Bot_Trading.xlsx](risultati/Confronto_Bot_Trading.xlsx).
- **Applicazione di monitoraggio:** [QUANTIX IQ](QUANTIX_IQ/README.md).

## Cartelle

| Percorso | Contenuto |
|---|---|
| `mine/` | Copia completa della cartella MT5: 1.335 sorgenti `.mq5`, 1.332 compilati `.ex5`, due preset e due documenti originali |
| `parametri/` | Input esportati dal profilo demo salvato della prova298, con conferma broker da rifare |
| `risultati/` | Excel originali del Desktop e istruzioni per leggerli |
| `docs/` | Installazione, configurazione e contesto delle prove |
| `inventari/` | Elenco dei file con dimensione e impronta SHA-256 |
| `QUANTIX_IQ/` | Interfaccia, immagini, documentazione e collegamento MT5 in sola lettura |
| `smc_bot (1)/` | Prototipo Python precedente, conservato come materiale storico |

### Archivio completo dei bot

Le sottocartelle di `mine` conservano nomi e struttura originali: ORB, SMC, Donchian, Moving Average, varianti FTMO, BTCUSD, NAS100, portafogli e prove ATR. L'archivio comprende anche esperimenti non profittevoli e versioni superate. I numeri di prova possono ripetersi in famiglie diverse: usare sempre il **percorso completo**.

Sono presenti tre sorgenti senza un compilato omonimo. L'aggiornamento conserva i file disponibili senza ricompilare o rilanciare tutte le prove. [Inventario verificabile](inventari/mine_sha256.csv).

### Come leggere i risultati

Gli Excel sono copie dei file del Desktop al momento dell'aggiornamento. Periodi, asset, feed e modelli di simulazione possono cambiare fra prove: il profitto maggiore, da solo, non rende due test confrontabili. Alcuni test hanno copertura di tick reali parziale; altri sono screening preliminari. Leggere le note di ciascuna riga. [Dettagli](risultati/README.md).

I risultati storici non sono una previsione e non certificano il rispetto delle regole di una prop firm. La versione di riferimento non è necessariamente adatta a un nuovo conto senza configurazione.

## Cosa contiene questo aggiornamento

Bot e Excel sono stati copiati senza modificare gli originali. Sono inclusi i sorgenti pubblicabili dell'app; configurazioni personali, database del conto, credenziali, chiavi push, log e dipendenze installate restano locali. Le configurazioni dell'app sono fornite come esempi da compilare.

Per questo aggiornamento non sono stati eseguiti nuovi backtest né modificati i bot attivi. [Riepilogo dell'aggiornamento](CHANGELOG.md).
