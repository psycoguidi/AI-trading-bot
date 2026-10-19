# Grafico e Chat AI — 19 ottobre 2026

## Grafico

- Trascina verso destra per leggere le candele precedenti, con mouse o dito. Sul tablet resta possibile scorrere verticalmente la pagina.
- I pulsanti Indietro e Avanti consentono lo stesso spostamento. Le frecce della tastiera funzionano quando il grafico è selezionato; Fine riporta all'ultima candela, + e − cambiano l'ampiezza.
- Torna in diretta riattiva il seguito del mercato. Gli aggiornamenti periodici dei prezzi non spostano una vista storica scelta dall'utente.
- M1, M5, M15, M30, H1, H4, D1 leggono candele effettive da MT5. Il timeframe operativo del bot resta M5.
- Lo storico viene caricato a blocchi di 300 candele. La disponibilità dipende dal broker e dallo storico restituito dal terminale. Non vengono inventate candele mancanti.
- I livelli ORB e le linee delle posizioni attuali sono nascosti quando la vista termina su giorni precedenti. Le schede delle posizioni restano aggiornate e sono etichettate come aperte ora.
- Il range del bot, quando esposto nel suo commento, è distinto dal range ricostruito da M1. Nessuno dei due è una promessa di ingresso.

## Chat AI locale

La Chat nella modalità MT5 è collegata a Ollama e al modello già installato `llama3.1:8b`. Accetta domande libere, anche fuori dal trading. La modalità Demo mantiene le spiegazioni dello scenario e offre un pulsante per aprire la Chat AI.

Il modello riceve un contesto del conto al momento della domanda: prezzi, posizioni, sessioni, riepilogo e regole del profilo salvato. Login e percorsi del PC non vengono inviati al modello. Il modello gira su questo PC e l'app non usa un servizio AI esterno. Le risposte non sono lo stato interno o il ragionamento effettivo dell'EA; possono essere errate.

L'AI non ha strumenti per eseguire comandi, inviare ordini, cambiare il bot o navigare sul web. Le risposte sono testo visualizzato con escaping, mai codice eseguito. Le conversazioni live restano nella memoria della pagina; i risultati delle richieste sul server restano temporaneamente in RAM, al massimo sei, e non vengono salvati nel registro o nella cache offline. Nessun modello viene scaricato automaticamente.

Le risposte possono richiedere decine di secondi o più sul PC, soprattutto al primo avvio. La pagina mostra l'attesa e si può cambiare sezione durante la generazione. Il PC deve rimanere acceso; Tailscale deve essere connesso per accedere da altri dispositivi. Avvia QUANTIX IQ tenta anche l'avvio del motore Ollama già installato se non risponde.

## Convivenza delle sessioni nella prova298

Dal codice e dal profilo salvato:

- massimo tre posizioni proprie, stesso simbolo e Magic Number;
- massimo due ingressi per sessione, dodici ingressi e quattro perdite al giorno;
- rientri della stessa sessione soggetti al ritorno nel range e ad almeno venti minuti;
- rischio complessivo allo stop limitato al 4% dell'equity, oltre alle protezioni giornaliere e totali;
- un ingresso ordinario di Londra o New York può convivere con posizioni Asia, se tutti i filtri e il budget consentono l'ordine;
- gli ingressi differiti dopo lo scarto ATR richiedono zero posizioni proprie, con DeferredOnlyWhenFlat attivo;
- con PrimaryEntryPriority attivo, l'EA tenta di chiudere eventuali posizioni DATR prima di un ingresso ordinario;
- il cambio sessione non chiude da solo le posizioni Asia ordinarie.

I posti disponibili mostrati nell'app sono un conteggio osservato, non una conferma del superamento dei filtri né del rischio residuo interno dell'EA.
