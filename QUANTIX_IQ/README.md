# QUANTIX IQ

Applicazione personale di monitoraggio MT5, copia dei sorgenti della versione **0.4.0** del 19 ottobre 2026. Tema chiaro/scuro, navigazione inferiore e interfaccia per telefono, tablet e PC.

## Funzioni presenti

- **Home:** posizioni e profitto in primo piano, riepilogo conto, stato e grafico.
- **Analisi:** criteri di ingresso osservabili, distanze, range e dati mancanti dichiarati.
- **Grafico:** storico navigabile, diversi timeframe, livelli del bot quando osservabili, entrate e SL/TP.
- **Chat:** domande libere al modello locale Ollama, con i dati osservati del conto. Le risposte non sono il ragionamento interno dell'EA e possono essere errate.
- **Attività:** diario automatico con esecuzioni, costi, note e contesto registrato quando disponibile.
- **Statistiche:** calendario P&L, dettaglio giornaliero, riepiloghi settimanali e filtri.
- **Account:** profilo personale e preferenze, notifiche Web Push per i dispositivi autorizzati.

Il collegamento legge un terminale MT5 già aperto sul conto previsto. Non effettua login, non avvia MT5 e non offre comandi di trading. Il progetto è legato alla prova298 e al formato del suo profilo; non è ancora un connettore generico per qualunque EA.

## Preparare una nuova installazione Windows

Occorrono Python 3.11 a 64 bit con supporto ai pacchetti indicati e MT5 Windows. In PowerShell, dalla cartella `QUANTIX_IQ` estratta:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .\07_collegamento\config.example.json .\07_collegamento\config.json
Copy-Item .\06_dati\accesso_privato.example.json .\06_dati\accesso_privato.json
```

Apri `07_collegamento/config.json` e inserisci i percorsi reali del tuo terminale e della sua cartella dati, il nome del profilo salvato, il numero conto atteso, il server e il simbolo. I percorsi Python devono puntare alla `.venv` appena creata. Il campo `expected_login: 0` dell'esempio deve essere sostituito: non è un login demo disponibile e il lettore sospende la lettura su un conto diverso.

Non serve inserire una password: il lettore usa la sessione già aperta in MT5. Attualmente legge il file `MQL5/Profiles/Charts/<profilo>/chart01.chr`; il profilo deve contenere lì la prova298 con i suoi input salvati. Non cambiare un profilo operativo soltanto per fare questa installazione.

Avvia **Avvia QUANTIX IQ.cmd**. L'indirizzo locale è `http://127.0.0.1:8876/01_app/`. Se una copia dell'app è già attiva sulla porta 8876, usa quella o prepara la nuova installazione su un altro PC: non avviare due copie sulla stessa porta.

Per la Chat libera serve anche Ollama con il modello `llama3.1:8b` installato localmente. Il progetto non include i pesi del modello. La demo ha invece dati e spiegazioni simulate, distinti dai dati MT5.

Per osservare il commento con i range esatti del bot, è incluso il sorgente dell'indicatore passivo `07_collegamento/QUANTIX_Observer.mq5`. Copialo in `MQL5/Indicators`, ricompilalo in MetaEditor e imposta nei suoi input conto e server attesi, simbolo e nome del bot. La copia pubblicata richiede questa configurazione; il compilato personale non è incluso. L'indicatore è separato dall'EA. Se l'osservazione non è disponibile, l'app indica i range come ricostruiti.

## Telefono e notifiche

Il collegamento remoto è privato e usa Tailscale Serve, con PC e dispositivo nello stesso account autorizzato. Dopo aver configurato l'app e completato l'accesso personale a Tailscale, **Collega telefono.cmd** configura l'HTTPS privato sulla porta 8444 e crea localmente `INDIRIZZO_TELEFONO.txt`. L'abilitazione di Serve sul proprio account può richiedere un passaggio manuale.

Sul telefono Android apri quel link in Chrome con Tailscale connesso e installa l'app dal menu del browser. In Account puoi attivare le notifiche e inviarne una di prova. Il vecchio indirizzo personale, le sottoscrizioni e la chiave push non sono distribuiti: ogni installazione crea i propri dati. [Guida Android](COME_INSTALLARLA_SU_ANDROID.md).

Per i dati aggiornati il PC, il servizio e MT5 devono restare accesi e connessi. In standby il PC non elabora nuovi dati. Le notifiche dipendono anche da rete, browser e permessi Android. **Disattiva accesso telefono.cmd** interrompe l'accesso remoto dell'app, senza fermare il bot.

## File locali e privacy

`config.json`, identità Tailscale, database, sottoscrizioni, chiave privata push, log e dipendenze sono esclusi da Git. Non copiarli in una repository pubblica. L'app ascolta su loopback; l'accesso dal telefono verifica l'identità inoltrata da Tailscale. È un'app personale, non un servizio pubblico con registrazione utenti.

La cartella `03_ragionamenti` conserva le scelte progettuali. `LEGGIMI_LOCALE_ORIGINALE.md` documenta la precedente installazione personale; per ripristinare questa copia usa le istruzioni qui sopra. I bot completi si trovano nella cartella [`mine`](../mine), senza duplicarne qui l'archivio.

Questo aggiornamento ha copiato e controllato i sorgenti; non ha ricreato un ambiente completo su un nuovo computer né attivato un altro terminale.
