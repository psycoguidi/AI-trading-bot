# QUANTIX IQ

App personale, versione 0.4 del 19 ottobre 2026. Navigazione inferiore, logo e colori approvati.

## Avvio

Apri **Avvia QUANTIX IQ.cmd**. Avvia il servizio locale e apre l’app nel browser del computer. Non avvia MT5 e non cambia il bot. Per i dati aggiornati anche MT5 deve essere aperto sul conto previsto. Non è stato modificato l’avvio automatico di Windows: dopo un riavvio apri nuovamente questo collegamento.

La versione mostrata in chat rimane una demo autonoma. Per installazione Android e collegamento MT5 usa l’app aperta nel browser. Le preferenze locali della demo e della chat sono separate.

## Telefono e tablet Android

Leggi **COME_INSTALLARLA_SU_ANDROID.md** e apri l’indirizzo in **INDIRIZZO_TELEFONO.txt**, con Tailscale connesso allo stesso account del PC. Il PC deve restare acceso e non in sospensione. Il collegamento è HTTPS privato; non è un sito pubblico.

## MT5 · sola lettura

Il selettore in alto permette di passare tra dati MT5 e Demo. La vista MT5 legge saldo, equity, quotazioni, posizioni ed esecuzioni recenti del conto configurato. Importi nella valuta effettiva del conto, attualmente USD. Il lettore richiede la cronologia dal 2025, secondo quanto disponibile presso il broker. Il calendario e il diario la conservano nell’archivio personale sul PC, includendo eventuali altre strategie sul conto.

Il grafico principale mostra prezzo Bid/Ask, range ORB e posizioni MT5 con entrata, SL e TP. Permette di scegliere sessione e periodo di 2, 4 oppure 8 ore. «Mostra tutti gli SL/TP» include anche i livelli lontani dal prezzo.

Quando disponibili, massimo e minimo vengono letti dal commento del bot tramite l’indicatore passivo QUANTIX_Observer e indicati come «livelli del bot». Negli altri casi il range resta etichettato «ricostruito». Finestre temporali e margine dalla protezione sono ricostruzioni dal profilo salvato e dai dati disponibili. L’app non legge l’ATR interno, il motivo certo di uno scarto o un drawdown storico intraday: questi dati non vengono inventati.

Non sono disponibili comandi per aprire, chiudere, attivare o modificare bot o ordini.

## Cosa puoi provare nella Demo

- **Home:** stato del bot, saldo, equity, risultato giornaliero, distanza dalla protezione, grafico e operazioni simulate.
- **Chat:** grafico sopra la conversazione, domande sui filtri e sul rischio, spiegazioni relative allo scenario attuale.
- **Attività:** registro degli eventi, filtri, ricerca, dettagli e trasferimento di un evento nella Chat. Esportazione CSV.
- **Statistiche:** periodi diversi, risultati realizzati, curva del saldo, tabella giornaliera ed esportazione CSV.
- **Impostazioni:** tema, nome, capitale e limiti della simulazione, connessione simulata, pausa, esportazione e ripristino.

Con **Prova scenario** puoi confrontare attesa, ingresso, filtro ATR, vendita scartata, protezione giornaliera e disconnessione. Con **Avanza una candela** prosegui il mercato fittizio di5 minuti. Dopo un ingresso, apri i dettagli della posizione per simulare1R, trailing, stop e target.

## Cosa rappresentano i dati

Nella modalità **Demo**, prezzi, profitti, orari e connessioni sono dimostrativi. Non sono i risultati attuali del conto. Il riferimento è una sessione fittizia di venerdì 16 ottobre 2026. La vista MT5 è distinta e non viene sostituita automaticamente con dati di esempio se cade il collegamento.

La Chat della modalità MT5 usa il modello AI locale **llama3.1:8b**, tramite Ollama sul PC. Accetta domande libere e riceve i dati osservati del conto quando disponibili. Non può inviare ordini, cambiare il bot o consultare il web. Le risposte possono contenere errori e non sono il ragionamento interno dell'EA. Il primo avvio può richiedere qualche minuto. La modalità Demo mantiene le spiegazioni del suo scenario e un pulsante apre la Chat AI. Le notifiche Web Push si attivano in Account con il consenso del browser su ciascun dispositivo.

Il lettore MT5 controlla che il terminale sia già aperto sul conto previsto; non effettua login e non invia ordini. Le copie della prova298 sono soltanto riferimenti, non vengono caricate o attivate dall’app.

Dopo la prima apertura riuscita, grafica e demo sono disponibili dalla cache offline. Le API MT5, lo storico e le risposte AI non vengono conservati nella cache offline. La conversazione live rimane nella memoria della pagina fino al ricaricamento o alla chiusura; le richieste AI recenti restano temporaneamente nella RAM del servizio. In assenza di collegamento la pagina segnala dati assenti o ultimi dati non attuali.

## Grafico navigabile

Trascina con mouse o dito per tornare indietro, oppure usa **Indietro / Avanti**. **Torna in diretta** segue di nuovo il prezzo. I timeframe **M1, M5, M15, M30, H1, H4 e D1** cambiano la vista; il bot resta operativo su M5. Le candele precedenti vengono lette da MT5 a blocchi, secondo lo storico disponibile nel terminale.

La scheda **Nuovi ingressi tra sessioni** spiega la convivenza delle posizioni Asia con ingressi ordinari e differiti ATR. I posti liberi sono teorici: il bot deve comunque superare filtri e limiti di rischio.

La **Home** mostra operazioni aperte e profitti in cima, riepilogo compatto del conto, stato del collegamento e grafico. La voce **Analisi** nel menu inferiore raccoglie criteri LONG/SHORT, distanze e condizioni mancanti, range delle sessioni, rischio stimato, ultima candela e strumenti completi del grafico. I controlli interni del bot non leggibili restano **Da verificare**. **Account** contiene profilo personale, preferenze, notifiche e impostazioni del collegamento. Dettagli in `03_ragionamenti/11_home_essenziale_e_analisi.md`.

Dettagli in `03_ragionamenti/07_storico_multitimeframe_e_AI.md`. Per ricevere le modifiche sul tablet basta ricaricare la pagina dell'app, senza reinstallarla.

## Cartelle

| Cartella | Contenuto |
|---|---|
| 01_app | Interfaccia, modello demo, vista MT5, installazione PWA e icone |
| 02_immagini | Logo originale, logo ottimizzato per l’app e mockup |
| 03_ragionamenti | Obiettivi, scelte progettuali, funzioni e limiti |
| 04_bot_riferimento | Copia della prova298 e impronte dei file originali |
| 05_verifiche | Test, esiti, schermate e copia precedente alla PWA |
| 06_dati | Scenari demo e configurazioni locali del servizio |
| 07_collegamento | Lettore MT5, server, avvio e accesso privato |

Il tema e le preferenze vengono conservati nel browser quando il salvataggio locale è disponibile. **Ripristina demo** cancella soltanto lo stato del prototipo.

Solo l’interfaccia e le API previste sono raggiungibili dal telefono. Il codice del bot, i documenti del Desktop e le configurazioni locali non sono distribuiti dal server.

Per interrompere il collegamento remoto usa **Disattiva accesso telefono.cmd**. Per riattivarlo usa **Collega telefono.cmd**. Il bot MT5 non viene fermato da queste azioni.


## Calendario, diario e Account

In **Statistiche**, il calendario mensile mostra P&L e numero di operazioni interessate per giorno, con riepiloghi settimanali. Tocca una data per gli importi esatti, le singole posizioni, le esecuzioni e le statistiche del giorno. Il P&L include i costi registrati nel giorno italiano, esclude il flottante e i versamenti. Il win rate considera soltanto posizioni interamente chiuse. I filtri distinguono prova298 dalle altre operazioni.

In **Attività → Diario**, ogni posizione conserva esecuzioni, risultati, note personali e, quando osservati dall’app, contesto e variazioni SL/TP. Le note restano sul PC e sono disponibili da tutti i tuoi dispositivi autorizzati. I dati d’ingresso non registrati per vecchie operazioni rimangono esplicitamente assenti.

In **Account**, puoi salvare nome, email facoltativa, nome del conto e preferenze. La protezione resta l’accesso privato Tailscale; non è un sistema pubblico di registrazione. Non vengono salvate password MT5 nel profilo. Puoi scaricare una copia dei tuoi dati.

Per le **notifiche Android**, ricarica l’app, apri Account, premi **Attiva notifiche** e consenti la richiesta del browser; poi usa **Invia una prova**. La ricezione può funzionare anche con l’app chiusa grazie al servizio push del browser, con avvisi cifrati. Il PC e il servizio QUANTIX IQ devono restare accesi e connessi. La consegna dipende dal browser, da Android e dalla rete: lo stato «Accettata dal servizio push» non certifica la visualizzazione sul telefono. Gli avvisi non cambiano il bot e non sono comandi di trading.

Dettagli in `03_ragionamenti/12_calendario_diario_account_notifiche.md`.
