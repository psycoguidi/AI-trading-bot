# Aggiornamento del 19 ottobre 2026

## Interfaccia conservata

Marchio, colori, cinque sezioni e navigazione inferiore rimangono quelli approvati. L’app installabile usa la stessa interfaccia locale.

## Collegamento

- Server QUANTIX IQ su `127.0.0.1:8876`, senza ascolto sulla rete pubblica o sulla LAN.
- Tailscale Serve su HTTPS 8444, separato dalla porta 8443 del precedente progetto.
- Accesso consentito solo all’identità personale che ha configurato il collegamento. Il server verifica host, identità inoltrata da Serve, origine della richiesta e provenienza loopback.
- Nessun Funnel pubblico e nessun cambiamento alle altre applicazioni.
- Il server espone soltanto i file dell’interfaccia, `/api/health` e `/api/mt5`. Configurazioni locali, codice del bot e documenti del Desktop non sono pubblicati.

## Lettura del conto

Il lettore controlla che MT5 sia già aperto e che conto e server corrispondano a quelli configurati. Non effettua login e non invia ordini. Un conto diverso comporta la sospensione della lettura. Tutti gli importi effettivi mantengono la valuta del conto.

L’API di MT5 viene interrogata in un processo separato, al massimo ogni cinque secondi. La pagina aggiorna i dati ogni cinque secondi quando è visibile. I dati reali restano in memoria e non vengono salvati nello stato della chat o nella cache offline.

## Dati osservati e deduzioni

Saldo, equity, quotazioni, posizioni ed esecuzioni provengono dall’API. Le finestre ORB, i range e la distanza dalla protezione sono ricostruzioni dal profilo salvato e dai dati disponibili, con etichette visibili.

Il profilo salvato potrebbe differire da modifiche non ancora salvate nel terminale. L’EA non espone un heartbeat, il suo ATR o l’intero stato interno: la pagina non dichiara confermate le singole decisioni del bot. L’interfaccia non equivale a un controllo di conformità FTMO.

## Installazione e modalità offline

Manifest con icone 192 e 512, icona maskable, avvio standalone e percorso stabile. Service worker per la sola interfaccia. API esclusa dalla cache; nessuna sostituzione automatica con prezzi simulati se cade il collegamento. La demo è una modalità separata selezionabile.

## Verifica da completare sul dispositivo

L’accesso HTTPS è verificato dal PC. La verifica su un telefono Android reale, con dati mobili e installazione dalla Home, richiede la conferma dell’utente. Gli esiti automatici sono nella cartella `05_verifiche`.
