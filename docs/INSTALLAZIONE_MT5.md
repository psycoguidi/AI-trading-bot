# Scaricare, installare e avviare un bot su MT5

Questa guida usa **prova298 su XAUUSD M5** come riferimento. Le altre versioni dell'archivio possono avere input e requisiti differenti.

## 1. Scarica il progetto

Apri [la repository](https://github.com/psycoguidi/AI-trading-bot), scegli **Code → Download ZIP** ed estrai l'archivio. In alternativa, con Git:

```sh
git clone https://github.com/psycoguidi/AI-trading-bot.git
```

Il sorgente di riferimento è `mine/prove_ATR_20261016/prova298.mq5`; nella stessa cartella trovi il compilato `.ex5`.

## 2. Copia i file nel terminale giusto

Nel terminale MT5 Windows destinato alla prova, apri **File → Apri cartella dati**. Entra in `MQL5/Experts` e copia lì la cartella `mine`, mantenendo le sottocartelle. Fai prima una copia della cartella esistente se contiene modifiche locali. La cartella dati varia tra installazioni: usa quella aperta dal terminale. [Struttura delle cartelle MT5](https://www.metatrader5.com/en/terminal/help/start_advanced/structure).

Un `.mq5` è il sorgente; un `.ex5` è il programma eseguibile. Per ricompilare, apri il sorgente in MetaEditor e premi **F7**. Controlla gli errori e gli avvisi: con errori non viene prodotto un nuovo eseguibile. Se un compilato richiede un'architettura CPU diversa, ricompila per una compatibile. [Compilazione in MetaEditor](https://www.metatrader5.com/en/metaeditor/help/development/compile).

## 3. Verifica gli input della prova298

Prima di consentire nuovi ordini, leggi la [scheda dei parametri](PROVA298.md). Questa versione parte da un riferimento di capitale **10.000 nella valuta del conto**; i nomi `EUR` delle variabili non effettuano conversioni valutarie.

- Controlla simbolo, contratto, lotto minimo, leva, margine e modalità del conto. Le prove operative di riferimento usavano un conto hedge.
- Verifica l'orario del server, la sua regola di ora legale e il cambio del giorno usato dalle protezioni.
- `InpFTMOBrokerClockConfirmed=false` impedisce l'avvio operativo fuori dal tester. Impostalo a `true` soltanto dopo la verifica del broker; non è una correzione generica per un bot che non entra.
- Il preset in `parametri/prova298_profilo_demo_da_verificare.set` contiene gli input del profilo demo salvato. La conferma broker è stata riportata a `false`; non è un preset universale per altri conti.

La prova298 contiene vincoli fissi anche nel codice: modificare soltanto il capitale iniziale non la adatta automaticamente a conti più piccoli o alle regole di un'altra prop.

## 4. Provala nello Strategy Tester

Apri il tester con **Ctrl+R**, seleziona `mine/prove_ATR_20261016/prova298`, **XAUUSD**, **M5**, capitale **10.000** e la valuta del confronto. Per riprendere le condizioni di ricerca, usa leva **1:33** se disponibile e **Ogni tick basato su tick reali**.

Scegli un periodo dal **01/01/2025 all'ultima data realmente disponibile**. Per confrontare una riga dell'Excel, ripeti invece le sue date esatte, lo stesso broker e gli stessi input. Disattiva l'ottimizzazione per una prova singola. Salva report e preset e controlla nel Journal dati mancanti, costi e copertura dei tick reali. Non confrontare uno screening con un test completo come se fossero equivalenti. [Guida al tester](https://www.metatrader5.com/en/terminal/help/algotrading/testing).

## 5. Avviala sul conto demo

Dal Navigatore aggiorna l'elenco degli Expert Advisor, apri il grafico corretto e trascina il bot sul grafico. Nelle proprietà controlla gli input e abilita il trading automatico dell'EA; anche **Algo Trading** del terminale deve essere abilitato. Un grafico ospita un solo EA. [Avvio e permessi dei robot MT5](https://www.metatrader5.com/en/terminal/help/algotrading/trade_robots_indicators).

Per la prova298 usa XAUUSD M5 e verifica nel pannello **Experts/Journal** il messaggio `Tri-Session Master v5.5 (Londra senza fade) Ready.`. Un errore di inizializzazione richiede di correggere la causa prima di proseguire. Il bot può risultare pronto e non avere ancora un segnale valido.

Evita copie duplicate della stessa strategia sul conto: riconoscimento delle posizioni e protezioni possono interferire anche cambiando il Magic Number. Prima del passaggio a un conto reale o prop servono prove nella configurazione effettiva, incluse le regole sulle notizie e sulle pause di mercato.

## 6. Se non apre operazioni

Leggi prima Experts e Journal. Distingui un errore di avvio da un segnale scartato. Nella prova298 contano finestre di sessione, chiusure delle candele, filtro ATR, direzione giornaliera, cooldown, numero di posizioni e rischio residuo. Spread, margine e lotto minimo possono inoltre impedire l'ordine. Non è prevista un'operazione obbligatoria ogni giorno.

Per continuare a elaborare nuovi segnali il terminale deve ricevere prezzi e rimanere attivo su un PC o VPS connesso e non in standby. Gli eventuali SL/TP già accettati dal server non sostituiscono le protezioni che richiedono l'esecuzione del codice.
