# Bot di riferimento e calcoli della demo

La copia archiviata è **prova298**, con sorgente e binario originali. Le impronte SHA256 sono in04_bot_riferimento/origine.json. Nessuna copia viene eseguita dall’applicazione.

## Regole rappresentate

- Range e breakout della sessione New York, valutazione di candele M5.
- Soglia iniziale del15% dell’ATR D1 e possibile ammissione successiva alla stessa soglia.
- Filtro direzionale e filtro giornaliero per gli short.
- Stop sul lato opposto del range, target6R.
- Protezione a+1R con stop a+0,1R; trailing da+2,5R a distanza1,25R.
- Chiusura del venerdì alle16 italiane, come nella configurazione attiva di riferimento al momento della progettazione.
- Rischio base2,5%, moltiplicatore NY0,8; rischio ridotto per ingresso ATR ritardato.

Il modello dimostrativo non riproduce l’intero motore MQL5: non simula tutte le sessioni, la piramidazione, la riconciliazione del conto, i tick del broker, commissioni, swap, cambio valuta o slippage. Il corpo delle candele è costruito per rappresentare i casi descritti.

## Contabilità

- Saldo = capitale iniziale + risultati realizzati.
- Equity = saldo + risultato delle posizioni aperte.
- Risultato del giorno = risultato realizzato oggi + risultato aperto.
- Protezione = massimo fra soglia giornaliera e soglia totale, più cuscinetto.
- Margine residuo = massimo fra zero ed equity meno protezione.
- Il rischio disponibile di un nuovo ingresso considera il percorso dal prezzo attuale agli stop già presenti e una riserva di5EUR.

La demo usa un contratto convenzionale da100 unità e contabilizza i risultati direttamente inEUR. È un modello illustrativo: nel collegamento futuro serviranno specifiche e conversioni reali del broker.

## Drawdown

La Home mostra il margine dalla protezione. Le Statistiche misurano il drawdown dal picco del **saldo realizzato** nello specifico intervallo, non il minimo dell’equity su ogni tick. Queste misure sono etichettate separatamente.
