# Calendario, diario, Account e notifiche

Implementazione del 19 ottobre 2026. Accesso privato Tailscale e navigazione inferiore conservati. Nessun nuovo comando o modifica al bot di trading.

## Calendario dei risultati

Statistiche presenta un mese navigabile, da lunedì a domenica, con celle verdi e rosse, riepiloghi settimanali e quattro indicatori: P&L netto, posizioni interessate, win rate delle posizioni chiuse e giorni attivi. Le celle mostrano importi arrotondati per la leggibilità sul telefono; il dettaglio del giorno contiene i centesimi esatti.

Il netto giornaliero somma le esecuzioni di trading e i costi nel giorno italiano di registrazione. Depositi, prelievi e flottante sono esclusi. Una posizione con chiusure parziali rimane una posizione: il win rate e il profit factor includono soltanto posizioni interamente concluse. Il risultato complessivo di una posizione può differire dal suo contributo a una singola giornata se costi o uscite sono distribuiti su più giorni.

I filtri distinguono il magic number e l’asset della prova298 dalle altre operazioni. L’assenza di dati per una giornata non certifica che il broker conservi uno storico completo.

## Archivio e diario

Il servizio importa la cronologia disponibile dal 2025 e la conserva in `06_dati/quantix_iq.sqlite`, con deduplicazione per ticket. Ogni posizione ha esecuzioni, risultati, note e, quando osservati dall’app, prima rilevazione, candele M5 e variazioni di SL/TP. Un contesto osservato dopo l’ingresso resta etichettato come tale: non viene presentato come motivo certo dell’ingresso. Per operazioni precedenti all’attivazione mancano gli SL/TP iniziali e i filtri storici.

Il diario si apre in Attività, accanto alle esecuzioni recenti. È disponibile anche con MT5 chiuso, se il servizio sul PC è raggiungibile. Il grafico storico può richiedere MT5 connesso; la sua indisponibilità non nasconde i dati salvati. I dati grafici di XAUUSD non vengono attribuiti a operazioni su altri asset.

## Account

Profilo personale con nome, email facoltativa, nome del conto e preferenze degli avvisi, salvati sul PC e condivisi sui dispositivi autorizzati. L’email non è un canale di notifica. Account è l’ultima voce del menu e conserva anche installazione, collegamento e impostazioni precedenti.

Il profilo non contiene password MT5 e non crea un servizio pubblico di registrazione. L’archivio è vincolato al conto configurato, per evitare di mescolare cronologie di conti diversi. L’esportazione personale include esecuzioni, note e contesti, ma non chiavi push né password.

## Notifiche Web Push

In Account, ogni dispositivo richiede il proprio consenso tramite «Attiva notifiche». Il browser crea un’iscrizione al suo servizio push; il server cifra i messaggi con Web Push e VAPID. Le destinazioni ammesse sono quelle dei servizi push supportati, con HTTPS e chiavi validate; indirizzi locali e destinazioni arbitrarie vengono rifiutati.

Avvisi: entrate, uscite anche parziali, variazioni SL/TP osservate, margine stimato alla protezione sotto la soglia scelta, perdita e ripristino del collegamento MT5. Importi nelle notifiche disattivati inizialmente. Lo storico importato al primo avvio non genera una raffica di notifiche. La coda persistente deduplica gli eventi e limita i tentativi; gli avvisi scadono dopo 30 minuti.

Il browser può ricevere un push con app chiusa. Il PC e QUANTIX IQ devono comunque essere accesi e connessi; il telefono deve avere Internet. Una disconnessione del PC stesso impedisce al servizio locale di inviare un avviso in quel momento. «Accettata dal servizio push» indica l’accettazione del fornitore e non prova che Android abbia mostrato il messaggio.

Le chiavi VAPID restano in `06_dati/webpush_private.pem`, escluso dai percorsi pubblicati. Dipendenze in `07_collegamento/dependencies`, versione richiesta in `requirements_push.txt`.

## Verifiche

- Confini di accesso e lettore MT5: 9 verifiche precedenti mantenute.
- Diario e contabilità: deduplicazione, centesimi, costi su giorni differenti, chiusure parziali, operazioni manuali, storico incompleto e vincolo al conto.
- Notifiche: destinazioni non ammesse, preferenze, deduplicazione della coda; consenso e ricezione sul telefono reale confermati nella chat. Il contatto VAPID iniziale non valido e stato corretto; firma e cifratura sono coperte da una verifica aggiuntiva senza rete.
- Interfaccia: calendario a 320, 390, 820 e 1200 pixel; selezione del giorno, diario e scheda posizione, salvataggio del profilo, caricamento in una seconda pagina.

Esiti e immagini sono in `05_verifiche`. Cache dell’interfaccia v9; le API restano escluse dalla cache del service worker.

Riferimenti tecnici: [Push API](https://developer.mozilla.org/en-US/docs/Web/Progressive_web_apps/Guides/Offline_and_background_operation), [iscrizione Web Push](https://developer.mozilla.org/en-US/docs/Web/API/PushManager/subscribe), [pywebpush](https://github.com/web-push-libs/pywebpush).
