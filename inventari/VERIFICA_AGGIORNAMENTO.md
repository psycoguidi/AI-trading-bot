# Verifica dell'aggiornamento · 19 ottobre 2026

- Cartella `mine`: **2.671 file**, dimensione originale complessiva **202.495.666 byte**.
- Ogni copia confrontata con l'originale mediante SHA-256. Tutti i file `mine` e i due Excel conservano gli stessi byte anche negli oggetti preparati per il commit Git.
- I due archivi Excel superano il controllo di integrità ZIP; contenuti e formule non sono stati modificati.
- Controllo sintattico di 38 sorgenti Python e 9 file JavaScript; nessun errore rilevato. Collegamenti locali dei documenti Markdown verificati.
- Controllo dei file di testo e degli XML degli Excel per credenziali riconoscibili e riferimenti all'account o all'indirizzo privato dell'installazione locale. I candidati residui sono riferimenti a variabili, esempi o generazione locale delle chiavi.
- Esempi di configurazione pubblicati con valori da compilare. Configurazioni reali, database, chiavi e log assenti dal contenuto corrente e protetti dalle esclusioni Git.
- La sola copia pubblicabile di `QUANTIX_Observer.mq5` è stata resa configurabile e compilata separatamente: **0 errori, 0 avvisi**, X64 Regular. Il sorgente verificato è incluso; il compilato personale è escluso.
- Nessun nuovo backtest o controllo statistico aggiuntivo dei risultati storici. Nessuna modifica agli originali MT5, agli Excel del Desktop o all'installazione QUANTIX IQ in esecuzione.

Queste verifiche riguardano l'aggiornamento del repository. Non equivalgono a una prova operativa dei 1.335 sorgenti, né a una certificazione di sicurezza, profitto o conformità prop.
