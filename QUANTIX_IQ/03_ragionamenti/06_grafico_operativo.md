# Grafico operativo — 19 ottobre 2026

## Dati e controlli

Il grafico principale usa tutta la larghezza disponibile. Mostra candele M5 reali di MT5, Bid/Ask e spread, massimo e minimo della sessione scelta e linee di entrata, stop e target delle posizioni aperte su XAUUSD. Periodi selezionabili: 2, 4 e 8 ore. I timestamp sono italiani.

I target lontani non comprimono automaticamente la zona del prezzo: sono indicati come «fuori vista». «Mostra tutti gli SL/TP» adatta invece la scala per includerli. Toccare una candela ne mostra OHLC e orario.

## Livelli effettivamente esposti dal bot

È stato aggiunto al grafico MT5 l’indicatore passivo **QUANTIX_Observer**. Non invia ordini, non modifica oggetti grafici, non scrive commenti e non cambia l’EA prova298. Legge soltanto il commento pubblicato dall’EA con `ChartGetString(CHART_COMMENT)` e lo rende disponibile all’app in un file locale, riscritto ogni secondo.

Il lettore dell’app accetta soltanto dati recenti del conto, server, simbolo e bot configurati. Scarta valori non inizializzati come `0 / 999999`. I file sorgenti ed eseguibili della prova298 sono rimasti invariati.

La dicitura «livelli del bot» compare solo se massimo e minimo validi sono presenti nel suo commento. Negli altri casi viene mostrata l’origine «livelli ricostruiti», con la relativa nota. Dopo il riavvio del terminale, ad esempio, il bot potrebbe non esporre il range di una sessione precedente.

Il lettore osserva il commento corrente: non è una lettura di tutte le variabili interne, dei filtri o della loro cronologia. La presenza del lettore non dimostra da sola che ogni segnale sia stato valutato dall’EA.

## Aggiornamento

L’osservatore aggiorna il file ogni secondo; il servizio dell’app e la pagina mantengono il loro aggiornamento periodico. La visualizzazione non è un flusso tick per tick e può avere alcuni secondi di ritardo. Nessun dato di conto viene aggiunto alla cache offline.

## Verifiche

Indicatore compilato: zero errori e zero avvisi. Verifiche sul riconoscimento di dati scaduti, conto differente e livelli non inizializzati; confronto dei livelli mostrati con l’API; apertura su desktop, tablet e larghezza 320px; controlli di sessione, scala SL/TP e dettaglio candela. Esiti nella cartella `05_verifiche`.

Fonti: https://www.mql5.com/en/docs/chart_operations/chartgetstring e https://www.mql5.com/en/docs/eventfunctions/eventsettimer .
