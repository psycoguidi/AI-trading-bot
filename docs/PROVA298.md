# Prova298 · riferimento XAUUSD M5

File: [sorgente](../mine/prove_ATR_20261016/prova298.mq5) · [EX5 disponibile](../mine/prove_ATR_20261016/prova298.ex5).

Il collegamento locale QUANTIX IQ indicava `prova298` su XAUUSD durante l'aggiornamento del 19 ottobre 2026. Qui sono descritti i **default del sorgente**; non viene affermato che ogni backtest o ogni grafico usi quegli stessi input.

## Strategia

ORB su Asia, Londra e New York, con segnali elaborati su M5. Il codice usa range di sessione, filtri ATR e corpo della candela, controlli della direzione giornaliera, rientri limitati e gestione del rischio. È presente l'ammissione differita dopo uno scarto ATR iniziale. Non basta il semplice attraversamento del massimo o del minimo per avere un ingresso.

## Valori principali nel sorgente

| Parametro | Default |
|---|---:|
| `InpEntryTF` | M5 |
| `InpUseEquityRisk` | true |
| `InpRiskPercent` | 2,5% prima dei limiti effettivi di rischio |
| `InpRR_Target` | 6 |
| `InpUsePartialClose` | false |
| `InpUseBreakEven` / trigger / lock | true / 1R / 0,1R |
| `InpTrailStartR` / `InpTrailGapR` | 2,5R / 1,25R |
| `InpMaximumPositions` | 3 |
| `InpSessionMaxEntries` / `InpMaxTradesDay` | 2 / 12 |
| `InpMaxLossesDay` / `InpReentryCooldownMin` | 4 / 20 minuti |
| `InpFTMOInitialCapital` | 10.000 |
| `InpFTMODailyLossEUR` / `InpFTMOTotalLossEUR` | 450 / 1.000 |
| `InpFTMOBufferEUR` / `InpFTMOReservePerTradeEUR` | 75 / 5 |
| `InpFTMOStopAtTarget` | false |
| `InpFTMOServerUTCWinter` / `InpFTMOServerUTCSummer` | UTC+2 / UTC+3 |
| `InpFTMOServerDST` | regola europea |
| `InpFTMOBrokerClockConfirmed` | false: verifica richiesta fuori dal tester |
| `InpExtraORBEnabled` / `InpExtraMinCapital` | true / 20.000 |

Il rischio realmente ammesso dipende anche dagli altri moltiplicatori, dalle posizioni già aperte e dalle protezioni: `2,5%` non descrive da solo la perdita massima di ogni operazione.

Le ore delle sessioni sono espresse nel tempo server previsto dal codice. La sessione aggiuntiva ha un requisito di capitale: abilitarla non la rende automaticamente disponibile su un saldo di 10.000.

## Preset del profilo salvato

[prova298_profilo_demo_da_verificare.set](../parametri/prova298_profilo_demo_da_verificare.set) contiene i 135 input estratti dal profilo MT5 salvato, senza informazioni di accesso. Per una nuova installazione la sola conferma dell'orario broker è riportata a `false`. Non è un'esportazione certificata degli input di ogni prova storica e non prova lo stato corrente della memoria del terminale.

## Limiti da conoscere

- L'inizializzazione impone, fra gli altri vincoli, una perdita totale configurata di 1.000 e una giornaliera non superiore a 500. È una versione costruita intorno al riferimento da 10.000.
- Buffer, controlli dell'equity e stop non garantiscono un limite esatto con gap, slippage, disconnessioni o esecuzioni rifiutate.
- Il drawdown dal picco, il minimo assoluto dall'inizio e la perdita giornaliera sono misure diverse. Non dedurre la conformità prop dal solo DD percentuale del report.
- Il nome delle variabili FTMO non implica conformità completa a un prodotto o alle sue regole correnti.
- Le altre prove in `mine` sono versioni autonome. Il numero più alto non identifica automaticamente il bot migliore o quello attivo.

Per i risultati storici usare gli [Excel](../risultati/README.md), conservando date, costi, feed e note della riga. Questo aggiornamento non ha eseguito nuovi backtest.
