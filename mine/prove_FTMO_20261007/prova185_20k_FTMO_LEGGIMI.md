# prova185_20k_FTMO

Derivata da prova183 per **XAUUSD, M5, conto in EUR da 20.000 €**, profilo **FTMO CFD 1-Step**. La leva si configura nel conto o nel tester, non nel codice: per il confronto richiesto selezionare **1:100**, verificando anche le specifiche di XAUUSD.

## Due modalità nello stesso bot

- **FTMO_CHALLENGE**: modalità predefinita. Target netto +2.000 €, tutte le posizioni e gli ordini del conto chiusi, Consistency del 50%. Si ferma quando gli obiettivi calcolati risultano raggiunti; la conferma ufficiale resta quella di Account MetriX.
- **FTMO_FUNDED_STANDARD**: prosegue senza target complessivo. Mantiene perdita giornaliera, trailing e calcolo della Consistency; aggiunge le pause per notizie e chiusure del mercato. Non richiede né esegue prelievi e non resetta automaticamente il conto.

I file `.set` Challenge e Funded contengono tutti gli 87 parametri. Sono da caricare nelle proprietà del bot o negli input del tester, scegliendo il preset appropriato.

## Protezioni e impostazioni

- Perdita giornaliera: **600 €**, rispetto al saldo delle 00:00 CE(S)T.
- Perdita totale: **2.000 €**, sottratti al più alto saldo registrato a mezzanotte, con minimo iniziale di riferimento 20.000 €. La soglia totale iniziale è quindi 18.000 € e può salire.
- Buffer prudenziale: **150 €** prima della soglia applicabile. Per esempio, senza profitto nel primo giorno il bot tenta di fermarsi intorno a 19.550 €, prima del limite ufficiale di 19.400 €.
- Rischio base **0,75% dell'equity**, poi moltiplicato per la sessione e limitato dalla capienza residua. Sotto il capitale iniziale applica un limite di rischio di 100 € per ingresso. Piramide base 0,20%, solo dopo la protezione della posizione originale.
- Riserva fissa di 10 € per posizione e ulteriore riserva di 12 € per lotto per costi e slippage. Le riserve sono stime prudenziali, non il tariffario ufficiale FTMO.
- Challenge: tetto prudenziale giornaliero al 45% del target, cioè **900 €**. Chiusura e pausa fino al giorno successivo; è modificabile con `InpFTMOChallengeDayCapPct`, anche a zero. Questo tetto favorisce la distribuzione dei profitti e generalmente richiede almeno tre giornate positive per raggiungere 2.000 €.
- Funded: `InpFTMOFundedDayCapEUR=0`, quindi nessun tetto di profitto fisso predefinito.
- Magic Number distinto: **2026101485**.

La ricostruzione include tutte le operazioni e i costi del conto, anche con altri Magic o su altri simboli. I costi di ingresso vengono attribuiti pro quota alle chiusure, comprese quelle parziali. Lo stato di protezione non si azzera cambiando Magic o riavviando il bot.

## Prima di un utilizzo operativo

Il bot **non è stato attivato su alcun conto**. Usare un conto dedicato: non installare contemporaneamente prova183 e prova185 sul medesimo conto per questo confronto. Non cambiare arbitrariamente la fase o la data di inizio per azzerare i limiti.

`InpFTMOBrokerClockConfirmed=false` è intenzionale. Fuori dal tester l'avvio operativo richiede una verifica dell'orario del server e del suo DST; soltanto dopo quella verifica impostare il parametro a true. Valori iniziali: UTC+2 in inverno, UTC+3 in estate, DST europeo. Il bot controlla anche la corrispondenza tra orario del server e UTC del computer. Orari ORB e chiusure programmate sono in **orario server**, non automaticamente nell'orario locale del computer.

Lo storico deve consentire di ricostruire il saldo del conto da 20.000 €, senza depositi/prelievi intermedi. Le inversioni netting `DEAL_ENTRY_INOUT` vengono rifiutate: questa versione è destinata a un conto dedicato hedging. Una ricostruzione non coerente sospende gli ingressi con una spiegazione nel registro.

Le protezioni monitorano l'equity presente e la capienza degli stop. Non possono ricostruire tutte le oscillazioni dell'equity avvenute mentre il terminale era spento o prima del primo avvio. Stop, buffer e timer non impediscono con certezza violazioni dovute a gap, slippage, disconnessioni o esecuzioni ritardate.

## Notizie e chiusure nel Funded Standard

In modalità Funded il calendario economico di MT5 viene aggiornato ogni 5 minuti. Il filtro usa gli eventi US/USD selezionati e **tutti quelli ad alta importanza**, quindi è più ampio della sola lista FTMO. Tenta la chiusura delle posizioni del bot da 15 minuti prima; sospende gli ingressi fino a 3 minuti dopo.

Dentro la finestra ufficiale di +/-2 minuti il bot non invia ordini. **SL e TP già sul server possono comunque essere eseguiti**: mantenere il terminale acceso e verificare che le chiusure anticipate siano riuscite. Con calendario assente, incompleto o con eventi senza orario preciso, gli ordini EA vengono sospesi; controllare eventuali posizioni già aperte.

Le sessioni sono lette dal contratto del simbolo. Il preset Funded prevede la chiusura 15 minuti prima di tutte le pause di sessione, anche quelle brevi, e conserva la chiusura prudenziale del venerdì della prova183. Le festività o modifiche straordinarie non dichiarate nel contratto richiedono di aggiornare `InpFTMOExtraClosuresServer` con date server separate da `;`, dopo aver controllato FTMO Trading Updates.

La Challenge non applica il filtro notizie né le restrizioni Funded sulle pause; conserva le regole prudenziali di rollover ereditate dalla prova183.

## Tester

Per provare gli ingressi: **XAUUSD, M5, deposito 20.000 EUR, leva 1:100, tick reali, inizio 01/01/2025 e fine alla data disponibile**. Il tester della Challenge può partire con i parametri predefiniti.

Per il Funded il calendario integrato di MT5 non viene usato nello Strategy Tester. Serve un CSV verificato, salvato in `Terminal\Common\Files`, indicato in `InpFTMONewsTesterCSV`. Senza questo file il Funded nel tester **non apre operazioni**, anziché simulare un filtro notizie inesistente.

Formato CSV, separatore `;`, date **UTC**, intervallo di copertura con estremo finale escluso:

```text
coverage;2025.01.01 00:00;2026.10.15 00:00
event;2025.01.10 13:30;nome evento verificato
```

La riga dell'evento è solo un esempio di formato: non costituisce un calendario completo. Non dichiarare copertura per date non verificate.

Il bot esporta nel tester un riepilogo, un registro giornaliero e le operazioni aggregate per posizione in `Terminal\Common\Files`. La soglia trailing è salvata separatamente per ogni giorno; i minimi osservati dall'EA sono distinti dai minimi complessivi riportati da MT5.

## Verifiche eseguite

Compilazione e 43 controlli MQL5 con dati sintetici: saldi di mezzanotte, trailing che non segue il picco intraday, Consistency comprensiva delle perdite, costi dei parziali, cambio giorno con perdita flottante, capienza degli stop, conversioni CE(S)T, bordi della finestra notizie e rifiuto di storico incoerente.

Non è stato eseguito un backtest di redditività completo della nuova variante. I controlli tecnici non certificano la conformità a FTMO, la copertura del calendario, il comportamento sul server assegnato o risultati futuri. La prova183 originale e il suo binario sono stati conservati.

Fonti delle regole consultate il 14/10/2026:

- https://ftmo.com/en/trading-objectives/
- https://ftmo.com/en/faq/can-i-trade-news/
- https://ftmo.com/en/faq/can-i-hold-positions-overnight-or-over-the-weekend/
- https://ftmo.com/en/faq/what-are-the-account-specifications/
