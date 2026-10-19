# Criteri di ingresso nella Home — 19 ottobre 2026

La Home contiene la scheda **Criteri di entrata**, tra riepilogo del conto e grafico. Si può scegliere Attuale, Asia, Londra, New York, Extra e la direzione LONG o SHORT. Le scelte e i pannelli aperti rimangono tali durante gli aggiornamenti periodici.

Ogni criterio mostra stato, valore osservato, eventuale distanza numerica e cosa deve accadere. Gli stati sono Soddisfatto, Non soddisfatto, In attesa, Da verificare, Non attivo e Recupero ATR. Nessuno di essi equivale a un ordine autorizzato o confermato.

## Calcoli mostrati

- Finestra oraria, formazione del range e tempo alla prossima chiusura M5.
- Massimo/minimo esposti dal bot; in mancanza, ricostruzione da M1 con indicazione delle candele disponibili.
- Chiusura dell'ultima M5, distinta dal prezzo intraday. Il breakout richiede una chiusura strettamente oltre il bordo; la distanza include un tick minimo del broker.
- Corpo della candela: valore percentuale e punti percentuali mancanti; la soglia è strettamente maggiore del parametro, non maggiore o uguale.
- Direzione della candela e filtri giornalieri, con controllo separato LONG/SHORT.
- ATR D1 ricostruito dai True Range dei giorni chiusi, secondo la formula del sorgente MetaQuotes locale `MQL5/Indicators/Examples/ATR.mq5`; non si usa la giornata ancora aperta. Il valore non è una lettura dell'handle interno del bot.
- Percorso ordinario oppure recupero ATR: stessa soglia, candela successiva al range, sessioni consentite, esposizione flat quando richiesta, direzione ed estensione massima. Il recupero è un percorso alternativo al filtro iniziale, non un allentamento della soglia.
- Posti liberi, protezione dell'equity, rischio residuo dopo stop delle posizioni e riserve. Le stime monetarie usano la valuta effettiva del conto.
- Stop candidato, rischio del lotto minimo e margine necessario, tramite le sole funzioni di calcolo del broker. Si applicano le riduzioni configurate di rischio per sessione e ingresso differito, ove ricostruibili. I prezzi sono quelli della lettura attuale: il calcolo verrà rifatto dal bot all'eventuale segnale.
- Capitale necessario per la sessione Extra e condizioni aggiuntive della piramidazione.

## Limiti espliciti

I contatori della singola sessione, il riarmo, i blocchi giornalieri persistenti, la mappa del rischio iniziale delle posizioni e i flag già eseguiti di break-even/piramide non vengono letti direttamente dall'EA. Sono indicati come **Da verificare**. Il contatore nel commento è un valore esposto, non un heartbeat interno.

ATR, EMA, rischio residuo e controllo del broker sono ricostruzioni. Il profilo salvato può non includere modifiche non ancora salvate nel terminale. Quotazioni vecchie o collegamento non aggiornato impediscono alla scheda di confermare criteri in verde. Non viene promesso un ingresso quando tutti i controlli osservabili passano.

La logica del bot e i suoi parametri operativi non sono modificati da questa funzione. Le nuove chiamate al broker sono calcoli e letture, senza invio di ordini. L'AI riceve soltanto un riepilogo compatto dei criteri della sessione attuale.

## Verifiche

12 casi di calcolo e 8 controlli nel browser superati: soglie strette, M5 chiusa anziché rottura intraday, ATR sui giorni chiusi, recupero ATR, posizioni aperte, lotto minimo dopo riduzione del rischio, dati incompleti, cambio sessione/direzione, permanenza dei pannelli aperti, larghezze 820/390/320 px e quotazioni non aggiornate. Le verifiche non aprono operazioni.
