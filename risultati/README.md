# Risultati e confronto

| File | Uso |
|---|---|
| [Confronto_Bot_Trading.xlsx](Confronto_Bot_Trading.xlsx) | Registro esteso delle versioni e dei test |
| [confronto_finale.xlsx](confronto_finale.xlsx) | Riepilogo delle versioni selezionate per strategia e asset |

Copie dei file del Desktop disponibili il **19 ottobre 2026**. Il contenuto e le formule sono preservati; non sono stati aggiunti risultati per questo aggiornamento. Le impronte delle copie sono registrate in [snapshot_2026-10-19.json](../inventari/snapshot_2026-10-19.json).

## Leggere una riga

Controllare bot e percorso della famiglia, asset, periodo, capitale, valuta, leva, input e tipo di simulazione. Le prove non condividono necessariamente tutte queste condizioni. Le righe di screening preliminare non sostituiscono una verifica completa su tick reali.

Il riepilogo finale conserva versioni storicamente profittevoli, comprese alcune con drawdown elevato: non è una lista di bot approvati per operare in reale o su prop. I rendimenti di periodi vicini ai due anni non sono rendimenti annuali.

## Drawdown: tre numeri diversi

- **Assoluto:** distanza tra capitale iniziale e minimo inferiore a quel capitale, misurato sul saldo o sull'equity a seconda del campo.
- **Dal picco:** calo da un massimo precedente al minimo successivo. Può essere grande anche se il conto non torna sotto il capitale iniziale.
- **Giornaliero:** richiede la ricostruzione rispetto al riferimento della giornata e all'orario corretto. Non coincide con il massimo DD complessivo del report.

Le etichette del report vanno mantenute: massimo monetario e massimo percentuale possono riguardare episodi differenti.
