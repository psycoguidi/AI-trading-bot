//+------------------------------------------------------------------+
//|                              Tri_Session_Master_v5.8_Stats.mq5    |
//|                   Logic: London/NY/Asia ORB - Sniper Filters     |
//|                   + FADE dei falsi breakout (solo NY/Asia)       |
//|                   + gestione dinamica + statistiche avanzate     |
//|                                                                    |
//|   v5.8 NOVITA' (rispetto a v5.7):                                 |
//|    - NESSUNA MODIFICA ALLA LOGICA DI TRADING. Tutte le condizioni |
//|      di ingresso, filtri, calcolo Entry/SL/TP, break-even,        |
//|      partial close, timing sessioni e money management sono      |
//|      IDENTICI alla v5.7.                                          |
//|    - Aggiunto esclusivamente un sistema di raccolta dati/         |
//|      statistiche avanzate per singolo trade (MAE/MFE, RR reale,   |
//|      costi stimati, slippage reale, ecc.) e un export CSV con     |
//|      analisi per sessione/setup, mensile, oraria, per giorno      |
//|      della settimana, long/short, robustness test, cost analysis  |
//|      e quantitative summary.                                      |
//|    - Ogni riga aggiunta e' marcata con il tag [STATS v5.8] nei    |
//|      commenti per essere facilmente individuabile.                 |
//+------------------------------------------------------------------+
#property copyright "Tri-Session Master v5.8 Stats"
#property version   "5.80"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS
// ═══════════════════════════════════════════
input group "London Session (ORB)"
input int      InpLon_StartHour     = 8;
input int      InpLon_StartMin      = 0;
input int      InpLon_DurationMin   = 60;
input bool     InpLon_UseDailyFilter= true;
input bool     InpLon_IgnoreBias    = false;

input group "New York Session (ORB)"
input int      InpNY_StartHour      = 15;
input int      InpNY_StartMin       = 30;
input int      InpNY_DurationMin    = 30;
input bool     InpNY_UseDailyFilter = true;

input group "Asian Session (ORB)"
input bool     InpUseAsia            = true;
input int      InpAsia_StartHour     = 1;
input int      InpAsia_StartMin      = 0;
input int      InpAsia_DurationMin   = 60;
input bool     InpAsia_UseDailyFilter= true;

input group "Sniper Filters (Crucial)"
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M5;
input int      InpMaxEntryDelayMin  = 60;
input double   InpBodyFilterPct     = 50.0;

input group "Fade dei Falsi Breakout (NON si applica a LONDON)"
input bool     InpFadeEnabled       = true;
input double   InpFadeRiskPct       = 50.0;
input double   InpFadeSLBufferPoints= 30;

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;
input double   InpATRMinRangeMult   = 0.35;

input group "Break-Even & Partial Close"
input bool     InpUseBreakEven      = true;
input double   InpBreakEvenTriggerR = 1.0;
input double   InpBreakEvenLockR    = 0.1;
input bool     InpUsePartialClose   = true;
input double   InpPartialClosePct   = 50.0;
input double   InpPartialCloseAtR   = 1.0;

input group "Risk & Money Management"
input double   InpFixedRiskAmount   = 80.0;
input bool     InpUseEquityRisk     = false;
input double   InpRiskPercent       = 0.8;
input double   InpRR_Target         = 2;
input int      InpMaxLossesDay      = 3;
input int      InpMaxTradesDay      = 6;
input ulong    InpMagicNumber       = 20260911;
input bool     InpShowComments      = true;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = true;
input double   InpMaxDailyLossPct    = 3.0;

// [STATS v5.8] ── Nuovo gruppo di input, NON influisce sul trading ──
input group "Statistics & CSV Export (v5.8 - NUOVO, nessun effetto sul trading)"
input bool     InpExportCSV          = true;                      // Esporta CSV a fine backtest/OnDeinit
input string   InpCSVFileName        = "TriSessionMaster_Stats.csv"; // Nome file (cartella MQL5/Files)
input double   InpSpreadCostPer001Lot= 0.018;                     // € di spread stimato per 0.01 lot (formula utente)

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ══════════════════════════════════════════
CTrade trade;
int    g_atrHandle = INVALID_HANDLE;

double g_lonHigh = 0.0, g_lonLow = 999999.0;
bool   g_lonRangeDefined = false;
bool   g_lonExecutedToday = false;

double g_nyHigh = 0.0, g_nyLow = 999999.0;
bool   g_nyRangeDefined = false;
bool   g_nyExecutedToday = false;

double g_asiaHigh = 0.0, g_asiaLow = 999999.0;
bool   g_asiaRangeDefined = false;
bool   g_asiaExecutedToday = false;

int    g_dailyBias = 0;
double g_daily_swing_highs[];
double g_daily_swing_lows[];
int    InpSwingLen_Daily = 5;

datetime g_currentDay = 0;
double   g_dayStartEquity = 0.0;
int      g_tradesToday = 0;
int      g_lossesToday = 0;
bool     g_dayLocked = false;
long     g_countedLossIds[];

// [STATS v5.8] struct estesa con campi aggiuntivi di SOLA LETTURA per le statistiche.
// I campi originali (posId, session, entryPrice, slDistance, dir, beDone, partialDone, entryTime)
// sono IDENTICI alla v5.7 e usati esattamente come prima per BE/partial close.
struct SessionMap {
   long     posId;
   string   session;         // etichetta originale (LONDON/NY/ASIA/NY_FADE/ASIA_FADE/NY_REVERSAL) - invariato
   double   entryPrice;
   double   slDistance;
   int      dir;
   bool     beDone;
   bool     partialDone;
   datetime entryTime;
   // --- campi aggiunti in v5.8, solo per statistiche ---
   string   sessionLabel;     // LONDON/NY/ASIA (sessione pura)
   string   setupLabel;       // NORMAL/NY_FADE/ASIA_FADE/NY_REV
   double   initialTP;
   double   requestedPrice;   // prezzo passato a trade.Buy/Sell (m5_close)
   double   actualFillPrice;  // prezzo di apertura effettivo della posizione
   double   lots;
   double   riskMoneyAtOpen;  // = MathAbs(profitPerLot)*lots, stesso calcolo gia' usato per il sizing
   double   rewardMoneyAtOpen;
   double   maeR;             // massima escursione avversa raggiunta, in multipli di R
   double   mfeR;             // massima escursione favorevole raggiunta, in multipli di R
};
SessionMap g_posSessionMap[];

// Stats continuazione
double g_profitLon = 0, g_profitNY = 0, g_profitAsia = 0;
int    g_winsLon = 0, g_winsNY = 0, g_winsAsia = 0;
int    g_totalLon = 0, g_totalNY = 0, g_totalAsia = 0;

// Stats fade (NY/Asia soltanto)
double g_profitNYFade = 0, g_profitAsiaFade = 0;
int    g_winsNYFade = 0, g_winsAsiaFade = 0;
int    g_totalNYFade = 0, g_totalAsiaFade = 0;

// Stats reversal - doppio fakeout (NY soltanto, Londra non ha fade quindi non arma mai il reversal)
double g_profitNYRev = 0;
int    g_winsNYRev = 0;
int    g_totalNYRev = 0;

// Stato per il rientro reversal dopo stop della fade (NY)
int      g_nyFadeDir = 0;    double g_nyFadeSL = 0;
bool     g_nyReArmActive = false;  datetime g_nyReArmDeadline = 0;

// ═══════════════════════════════════════════
// 📒 TRADE LOG (per statistiche avanzate nel report finale) - INVARIATO dalla v5.7
// ═══════════════════════════════════════════
struct TradeRecord {
    datetime entryTime;
    datetime closeTime;
    double   profit;
    int      dir;
    string   session;
};
TradeRecord g_tradeLog[];

void LogTrade(datetime entryTime, datetime closeTime, double profit, int dir, string session) {
    int s = ArraySize(g_tradeLog);
    ArrayResize(g_tradeLog, s+1);
    g_tradeLog[s].entryTime = entryTime;
    g_tradeLog[s].closeTime = closeTime;
    g_tradeLog[s].profit    = profit;
    g_tradeLog[s].dir       = dir;
    g_tradeLog[s].session   = session;
}

// ═══════════════════════════════════════════════════════════════════
// [STATS v5.8] NUOVO: record esteso per singolo trade + array dedicato
// Non sostituisce g_tradeLog (lasciato intatto), lo affianca.
// ═══════════════════════════════════════════════════════════════════
struct StatsRecord {
    long     posId;
    datetime openTime;
    datetime closeTime;
    string   dateStr;
    string   timeStr;
    string   dayName;      // Monday..Sunday
    int      dowNum;       // 0=Domenica..6=Sabato (day_of_week MQL5)
    string   sessionLabel; // LONDON/NY/ASIA
    string   setupLabel;   // NORMAL/NY_FADE/ASIA_FADE/NY_REV
    int      dir;          // 1 long, -1 short
    double   lots;
    double   entryReq;     // prezzo richiesto (m5_close)
    double   actualFill;   // prezzo di fill reale
    double   initialSL;
    double   initialTP;
    double   riskMoney;    // rischio monetario teorico (dal calcolo di sizing dell'EA)
    double   rewardMoney;  // reward monetario teorico (Entry->Initial TP)
    double   rrTheoretical;
    double   rrRealized;
    double   profit;       // profit puro del deal (esclude commissioni/swap)
    double   spreadCost;   // formula utente: lots/0.01*InpSpreadCostPer001Lot
    double   commission;
    double   swap;
    double   slippage;     // stimato da requestedPrice vs actualFill, convertito in €
    double   netAfterCosts;
    double   maeMoney;
    double   maeR;
    double   mfeMoney;
    double   mfeR;
    long     durationSec;
    string   news;         // sempre "N/D": nessun filtro news presente nell'EA originale
    string   result;       // WIN/LOSS/BE
};
StatsRecord g_statsLog[];

// [STATS v5.8] separa l'etichetta originale (usata da sempre nel codice) in sessione+setup puri
void SplitSessionSetup(string raw, string &sess, string &setup) {
    if(raw=="LONDON")        { sess="LONDON"; setup="NORMAL";  }
    else if(raw=="NY")       { sess="NY";     setup="NORMAL";  }
    else if(raw=="ASIA")     { sess="ASIA";   setup="NORMAL";  }
    else if(raw=="NY_FADE")  { sess="NY";     setup="NY_FADE"; }
    else if(raw=="ASIA_FADE"){ sess="ASIA";   setup="ASIA_FADE"; }
    else if(raw=="NY_REVERSAL"){ sess="NY";   setup="NY_REV";  }
    else                     { sess=raw;      setup="UNKNOWN"; }
}

string DowNameEN(int d) {
    string names[7] = {"Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"};
    if(d<0 || d>6) return "???";
    return names[d];
}

// [STATS v5.8] Registra il record esteso di un trade chiuso. Usa solo dati storici
// (deal di chiusura) + i campi salvati in g_posSessionMap al momento dell'apertura.
void LogTradeFull(long posId, datetime openTime, datetime closeTime,
                   string sessionLabel, string setupLabel, int dir, double lots,
                   double entryReq, double actualFill, double initialSL, double initialTP,
                   double riskMoney, double rewardMoney,
                   double rawProfit, double commission, double swap,
                   double maeR, double mfeR) {

    int s = ArraySize(g_statsLog);
    ArrayResize(g_statsLog, s+1);

    MqlDateTime dtOpen; TimeToStruct(openTime, dtOpen);

    g_statsLog[s].posId        = posId;
    g_statsLog[s].openTime     = openTime;
    g_statsLog[s].closeTime    = closeTime;
    g_statsLog[s].dateStr      = StringFormat("%04d.%02d.%02d", dtOpen.year, dtOpen.mon, dtOpen.day);
    g_statsLog[s].timeStr      = StringFormat("%02d:%02d:%02d", dtOpen.hour, dtOpen.min, dtOpen.sec);
    g_statsLog[s].dowNum       = dtOpen.day_of_week;
    g_statsLog[s].dayName      = DowNameEN(dtOpen.day_of_week);
    g_statsLog[s].sessionLabel = sessionLabel;
    g_statsLog[s].setupLabel   = setupLabel;
    g_statsLog[s].dir          = dir;
    g_statsLog[s].lots         = lots;
    g_statsLog[s].entryReq     = entryReq;
    g_statsLog[s].actualFill   = actualFill;
    g_statsLog[s].initialSL    = initialSL;
    g_statsLog[s].initialTP    = initialTP;
    g_statsLog[s].riskMoney    = riskMoney;
    g_statsLog[s].rewardMoney  = rewardMoney;
    g_statsLog[s].profit       = rawProfit;
    g_statsLog[s].commission   = commission;
    g_statsLog[s].swap         = swap;
    g_statsLog[s].maeR         = maeR;
    g_statsLog[s].mfeR         = mfeR;
    g_statsLog[s].durationSec  = (long)(closeTime - openTime);
    g_statsLog[s].news         = "N/D"; // nessun filtro news nel codice originale

    // Spread cost: formula fissa richiesta dall'utente (non e' un dato del tester)
    g_statsLog[s].spreadCost = (lots/0.01) * InpSpreadCostPer001Lot;

    // Slippage: differenza tra prezzo richiesto e prezzo di fill reale, convertita in €
    // usando lo stesso rapporto €/prezzo del rischio teorico (riskMoney / distanza Entry-SL).
    double slDistancePrice = MathAbs(entryReq - initialSL);
    double moneyPerPriceUnit = (slDistancePrice > 0 && riskMoney > 0) ? (riskMoney/slDistancePrice) : 0;
    double costPriceDiff = (dir==1) ? (actualFill-entryReq) : (entryReq-actualFill); // >0 = sfavorevole
    g_statsLog[s].slippage = costPriceDiff * moneyPerPriceUnit;

    // RR teorico e realizzato
    g_statsLog[s].rrTheoretical = (riskMoney>0) ? (rewardMoney/riskMoney) : 0;
    double netTesterProfit = rawProfit + commission + swap; // = "profitto netto" del tester (come nelle stat originali)
    g_statsLog[s].rrRealized = (riskMoney>0) ? (netTesterProfit/riskMoney) : 0;

    // MAE/MFE in € (approssimazione: basata sul rischio all'apertura; se e' avvenuto un
    // partial close la size e' cambiata durante la vita del trade, quindi questo e' un
    // valore indicativo, non un P/L istantaneo esatto post-partial)
    g_statsLog[s].maeMoney = maeR * riskMoney;
    g_statsLog[s].mfeMoney = mfeR * riskMoney;

    g_statsLog[s].netAfterCosts = netTesterProfit - g_statsLog[s].spreadCost - g_statsLog[s].slippage;

    if(netTesterProfit > 0)      g_statsLog[s].result = "WIN";
    else if(netTesterProfit < 0) g_statsLog[s].result = "LOSS";
    else                          g_statsLog[s].result = "BE";
}

// ═══════════════════════════════════════════
// 🚀 INIT & DEINIT
// ══════════════════════════════════════════
int OnInit() {
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    ArrayResize(g_posSessionMap, 0);
    ArrayResize(g_tradeLog, 0);
    ArrayResize(g_statsLog, 0); // [STATS v5.8]

    g_atrHandle = iATR(_Symbol, PERIOD_D1, InpATRPeriod);
    if(g_atrHandle == INVALID_HANDLE) Print("ATTENZIONE: impossibile creare handle ATR, filtro volatilità disattivato di fatto.");

    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

    RestoreDailyStats();
    Print("Tri-Session Master v5.8 (logica IDENTICA alla v5.7 + statistiche estese/CSV) Ready.");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    PrintFinalReport();
    PrintAdvancedStats();
    if(InpExportCSV) ExportStatsToCSV(); // [STATS v5.8]
    if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
    Comment("");
}

// ═══════════════════════════════════════════
// 🔧 UTILS & STATS
// ═══════════════════════════════════════════
void AddToArray(double &arr[], double val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }
void ShiftArray(double &arr[]) {
    if(ArraySize(arr)<=1){ArrayResize(arr,0);return;}
    for(int i=0;i<ArraySize(arr)-1;i++) arr[i]=arr[i+1];
    ArrayResize(arr,ArraySize(arr)-1);
}
void AddToLongArray(long &arr[], long val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }

double GetRiskAmount() {
    if(InpUseEquityRisk) return AccountInfoDouble(ACCOUNT_EQUITY) * (InpRiskPercent/100.0);
    return InpFixedRiskAmount;
}

double GetATRValue() {
    if(g_atrHandle == INVALID_HANDLE) return -1;
    double buf[];
    ArraySetAsSeries(buf, true);
    if(CopyBuffer(g_atrHandle, 0, 1, 1, buf) <= 0) return -1;
    return buf[0];
}

// [STATS v5.8] firma estesa con parametri opzionali (default = comportamento non cambiato
// per eventuali altri chiamanti). I campi originali sono assegnati esattamente come in v5.7.
void RegisterPosition(long posId, string session, double entryPrice, double slDistance, int dir,
                       double initialTP=0, double requestedPrice=0, double actualFillPrice=0,
                       double lots=0, double riskMoneyAtOpen=0, double rewardMoneyAtOpen=0) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) if(g_posSessionMap[i].posId == posId) ArrayRemove(g_posSessionMap, i, 1);
    int s = ArraySize(g_posSessionMap); ArrayResize(g_posSessionMap, s+1);
    g_posSessionMap[s].posId = posId;
    g_posSessionMap[s].session = session;
    g_posSessionMap[s].entryPrice = entryPrice;
    g_posSessionMap[s].slDistance = slDistance;
    g_posSessionMap[s].dir = dir;
    g_posSessionMap[s].beDone = false;
    g_posSessionMap[s].partialDone = false;
    g_posSessionMap[s].entryTime = TimeCurrent();

    // --- campi statistici v5.8 (sola lettura, nessun impatto sulla gestione) ---
    string sessLbl, setupLbl;
    SplitSessionSetup(session, sessLbl, setupLbl);
    g_posSessionMap[s].sessionLabel    = sessLbl;
    g_posSessionMap[s].setupLabel      = setupLbl;
    g_posSessionMap[s].initialTP       = initialTP;
    g_posSessionMap[s].requestedPrice  = requestedPrice;
    g_posSessionMap[s].actualFillPrice = actualFillPrice;
    g_posSessionMap[s].lots            = lots;
    g_posSessionMap[s].riskMoneyAtOpen = riskMoneyAtOpen;
    g_posSessionMap[s].rewardMoneyAtOpen = rewardMoneyAtOpen;
    g_posSessionMap[s].maeR = 0;
    g_posSessionMap[s].mfeR = 0;
}

string GetSessionByPosId(long posId) {
    for(int i=0; i<ArraySize(g_posSessionMap); i++) if(g_posSessionMap[i].posId == posId) return g_posSessionMap[i].session;
    return "UNKNOWN";
}

int FindSessionMapIndex(long posId) {
    for(int i=0; i<ArraySize(g_posSessionMap); i++) if(g_posSessionMap[i].posId == posId) return i;
    return -1;
}

void UnregisterPosition(long posId) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) if(g_posSessionMap[i].posId == posId) { ArrayRemove(g_posSessionMap, i, 1); return; }
}

bool IsIdAlreadyCounted(long posId) {
    for(int i=0; i<ArraySize(g_countedLossIds); i++) if(g_countedLossIds[i] == posId) return true;
    return false;
}

bool HasOpenPosition() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) return true;
    }
    return false;
}

double GetPivot(ENUM_TIMEFRAMES tf, int len, int shift, bool isHigh) {
    double val = isHigh ? iHigh(_Symbol, tf, shift) : iLow(_Symbol, tf, shift);
    for(int i=1; i<=len; i++) {
        double c1 = isHigh ? iHigh(_Symbol, tf, shift+i) : iLow(_Symbol, tf, shift+i);
        double c2 = isHigh ? iHigh(_Symbol, tf, shift-i) : iLow(_Symbol, tf, shift-i);
        if((isHigh && (c1 >= val || c2 >= val)) || (!isHigh && (c1 <= val || c2 <= val))) return EMPTY_VALUE;
    }
    return val;
}

int GetDailyBias() {
    if(ArraySize(g_daily_swing_highs) < 2 || ArraySize(g_daily_swing_lows) < 2) return 0;
    double lastHH = g_daily_swing_highs[ArraySize(g_daily_swing_highs)-1];
    double prevHH = g_daily_swing_highs[ArraySize(g_daily_swing_highs)-2];
    double lastLL = g_daily_swing_lows[ArraySize(g_daily_swing_lows)-1];
    double prevLL = g_daily_swing_lows[ArraySize(g_daily_swing_lows)-2];
    double close = iClose(_Symbol, PERIOD_D1, 1);
    if(close > lastHH || (lastHH > prevHH && lastLL > prevLL)) return 1;
    if(close < lastLL || (lastHH < prevHH && lastLL < prevLL)) return -1;
    return 0;
}

void RestoreDailyStats() {
    datetime todayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
    ArrayResize(g_countedLossIds, 0);
    if(!HistorySelect(todayStart, TimeCurrent())) return;
    int totalDeals = HistoryDealsTotal();
    for(int i=0; i<totalDeals; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
        if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) g_tradesToday++;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
            long posId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_SWAP) + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            bool stillOpen = false;
            for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
            if(!stillOpen && profit < 0 && !IsIdAlreadyCounted(posId)) { g_lossesToday++; AddToLongArray(g_countedLossIds, posId); }
        }
    }
    if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;
}

void PrintSessionStats(string name, double prof, int wins, int total) {
    double wr = (total > 0) ? (double)wins / total * 100.0 : 0.0;
    Print(name, " | Profit: ", DoubleToString(prof, 2), " | Wins: ", IntegerToString(wins), "/", IntegerToString(total), " | WR: ", DoubleToString(wr, 1), "%");
}

void PrintFinalReport() {
    Print("--------------------------------------------------");
    Print("TRI-SESSION MASTER v5.8 (logica invariata dalla v5.7) - FINAL REPORT");
    Print("--------------------------------------------------");
    Print("--- Continuazione ---");
    PrintSessionStats("LONDON      ", g_profitLon, g_winsLon, g_totalLon);
    PrintSessionStats("NY          ", g_profitNY, g_winsNY, g_totalNY);
    PrintSessionStats("ASIA        ", g_profitAsia, g_winsAsia, g_totalAsia);
    Print("--- Fade Falsi Breakout (solo NY/Asia) ---");
    PrintSessionStats("NY_FADE     ", g_profitNYFade, g_winsNYFade, g_totalNYFade);
    PrintSessionStats("ASIA_FADE   ", g_profitAsiaFade, g_winsAsiaFade, g_totalAsiaFade);
    Print("--- Reversal dopo stop fade (solo NY) ---");
    PrintSessionStats("NY_REV      ", g_profitNYRev, g_winsNYRev, g_totalNYRev);
    Print("--- Totale ---");
    double totProfit = g_profitLon + g_profitNY + g_profitAsia + g_profitNYFade + g_profitAsiaFade + g_profitNYRev;
    int totWins = g_winsLon + g_winsNY + g_winsAsia + g_winsNYFade + g_winsAsiaFade + g_winsNYRev;
    int totTrades = g_totalLon + g_totalNY + g_totalAsia + g_totalNYFade + g_totalAsiaFade + g_totalNYRev;
    PrintSessionStats("TOTALE      ", totProfit, totWins, totTrades);
    Print("--------------------------------------------------");
}

// ══════════════════════════════════════════
// 📊 STATISTICHE AVANZATE (v5.7) - INVARIATO
// ══════════════════════════════════════════
string MonthName(int m) {
    string names[13] = {"", "Gen", "Feb", "Mar", "Apr", "Mag", "Giu", "Lug", "Ago", "Set", "Ott", "Nov", "Dic"};
    if(m < 1 || m > 12) return "???";
    return names[m];
}

string DowName(int d) {
    string names[7] = {"Domenica", "Lunedi", "Martedi", "Mercoledi", "Giovedi", "Venerdi", "Sabato"};
    if(d < 0 || d > 6) return "???";
    return names[d];
}

struct MonthStat {
    int    ym;
    int    trades;
    int    wins;
    double profit;
    double grossProfit;
    double grossLoss;
    double equity;
    double peakEquity;
    double maxDD;
};
MonthStat g_monthStats[];

int FindOrAddMonth(int ym) {
    for(int i=0; i<ArraySize(g_monthStats); i++) if(g_monthStats[i].ym == ym) return i;
    int s = ArraySize(g_monthStats);
    ArrayResize(g_monthStats, s+1);
    g_monthStats[s].ym = ym;
    g_monthStats[s].trades = 0;
    g_monthStats[s].wins = 0;
    g_monthStats[s].profit = 0;
    g_monthStats[s].grossProfit = 0;
    g_monthStats[s].grossLoss = 0;
    g_monthStats[s].equity = 0;
    g_monthStats[s].peakEquity = 0;
    g_monthStats[s].maxDD = 0;
    return s;
}

void PrintAdvancedStats() {
    int n = ArraySize(g_tradeLog);
    if(n == 0) {
        Print("STATISTICHE AVANZATE: nessun trade registrato in questa sessione.");
        return;
    }

    ArrayResize(g_monthStats, 0);

    int    distBuckets = 7;
    string distLabels[7] = {"<= -100", "-100 / -50", "-50 / -20", "-20 / 0", "0 / 20", "20 / 50", "> 50"};
    int    distCount[7]; ArrayInitialize(distCount, 0);
    double distProfit[7]; ArrayInitialize(distProfit, 0);

    int curWinStreak = 0, curLossStreak = 0, maxWinStreak = 0, maxLossStreak = 0;

    int    hourCount[24]; ArrayInitialize(hourCount, 0);
    double hourProfit[24]; ArrayInitialize(hourProfit, 0);
    int    hourWins[24]; ArrayInitialize(hourWins, 0);

    int    dowCount[7]; ArrayInitialize(dowCount, 0);
    double dowProfit[7]; ArrayInitialize(dowProfit, 0);
    int    dowWins[7]; ArrayInitialize(dowWins, 0);

    int    longCount = 0, shortCount = 0, longWins = 0, shortWins = 0;
    double longProfit = 0, shortProfit = 0;

    for(int i = 0; i < n; i++) {
        double profit = g_tradeLog[i].profit;
        int    dir    = g_tradeLog[i].dir;

        MqlDateTime dtEntry;
        TimeToStruct(g_tradeLog[i].entryTime, dtEntry);
        int ym = dtEntry.year * 100 + dtEntry.mon;
        int mIdx = FindOrAddMonth(ym);
        g_monthStats[mIdx].trades++;
        g_monthStats[mIdx].profit += profit;
        if(profit > 0) { g_monthStats[mIdx].grossProfit += profit; g_monthStats[mIdx].wins++; }
        else if(profit < 0) g_monthStats[mIdx].grossLoss += (-profit);
        g_monthStats[mIdx].equity += profit;
        if(g_monthStats[mIdx].equity > g_monthStats[mIdx].peakEquity) g_monthStats[mIdx].peakEquity = g_monthStats[mIdx].equity;
        double ddNow = g_monthStats[mIdx].peakEquity - g_monthStats[mIdx].equity;
        if(ddNow > g_monthStats[mIdx].maxDD) g_monthStats[mIdx].maxDD = ddNow;

        int b;
        if(profit <= -100)      b = 0;
        else if(profit <= -50)  b = 1;
        else if(profit <= -20)  b = 2;
        else if(profit < 0)     b = 3;
        else if(profit < 20)    b = 4;
        else if(profit < 50)    b = 5;
        else                    b = 6;
        distCount[b]++;
        distProfit[b] += profit;

        if(profit > 0) {
            curWinStreak++;
            curLossStreak = 0;
            if(curWinStreak > maxWinStreak) maxWinStreak = curWinStreak;
        } else if(profit < 0) {
            curLossStreak++;
            curWinStreak = 0;
            if(curLossStreak > maxLossStreak) maxLossStreak = curLossStreak;
        }

        int h = dtEntry.hour;
        hourCount[h]++;
        hourProfit[h] += profit;
        if(profit > 0) hourWins[h]++;

        int dw = dtEntry.day_of_week;
        dowCount[dw]++;
        dowProfit[dw] += profit;
        if(profit > 0) dowWins[dw]++;

        if(dir == 1) {
            longCount++;
            longProfit += profit;
            if(profit > 0) longWins++;
        } else if(dir == -1) {
            shortCount++;
            shortProfit += profit;
            if(profit > 0) shortWins++;
        }
    }

    Print("====================================================");
    Print("STATISTICHE AVANZATE v5.7/5.8 - trade totali registrati: ", n);
    Print("====================================================");

    Print("--- RENDIMENTO MENSILE (Profit / PF / Max DD mensile) ---");
    for(int i = 0; i < ArraySize(g_monthStats); i++) {
        int y = g_monthStats[i].ym / 100;
        int m = g_monthStats[i].ym % 100;
        double pf = (g_monthStats[i].grossLoss > 0) ? g_monthStats[i].grossProfit / g_monthStats[i].grossLoss : 0.0;
        string pfStr = (g_monthStats[i].grossLoss > 0) ? DoubleToString(pf, 2) : "N/D (nessuna perdita)";
        double wr = (g_monthStats[i].trades > 0) ? (double)g_monthStats[i].wins / g_monthStats[i].trades * 100.0 : 0.0;
        Print(MonthName(m), " ", IntegerToString(y),
              " | Trade: ", IntegerToString(g_monthStats[i].trades),
              " | WR: ", DoubleToString(wr, 1), "%",
              " | Profit: ", DoubleToString(g_monthStats[i].profit, 2),
              " | PF: ", pfStr,
              " | Max DD mese: ", DoubleToString(g_monthStats[i].maxDD, 2));
    }

    Print("--- DISTRIBUZIONE VINCITE/PERDITE (per trade, valuta account) ---");
    for(int i = 0; i < distBuckets; i++) {
        Print(distLabels[i], " | Trade: ", IntegerToString(distCount[i]), " | Profit totale bucket: ", DoubleToString(distProfit[i], 2));
    }

    Print("--- SERIE MASSIME ---");
    Print("Massima serie di vittorie consecutive: ", IntegerToString(maxWinStreak));
    Print("Massima serie di perdite consecutive : ", IntegerToString(maxLossStreak));

    Print("--- DISTRIBUZIONE PER ORARIO DI INGRESSO (ora server broker) ---");
    for(int h = 0; h < 24; h++) {
        if(hourCount[h] == 0) continue;
        double wr = (double)hourWins[h] / hourCount[h] * 100.0;
        Print(StringFormat("%02d:00", h), " | Trade: ", IntegerToString(hourCount[h]),
              " | WR: ", DoubleToString(wr, 1), "%",
              " | Profit: ", DoubleToString(hourProfit[h], 2));
    }

    Print("--- DISTRIBUZIONE PER GIORNO DELLA SETTIMANA ---");
    for(int d = 0; d < 7; d++) {
        if(dowCount[d] == 0) continue;
        double wr = (double)dowWins[d] / dowCount[d] * 100.0;
        Print(DowName(d), " | Trade: ", IntegerToString(dowCount[d]),
              " | WR: ", DoubleToString(wr, 1), "%",
              " | Profit: ", DoubleToString(dowProfit[d], 2));
    }

    Print("--- LONG vs SHORT ---");
    double wrLong  = (longCount > 0) ? (double)longWins / longCount * 100.0 : 0.0;
    double wrShort = (shortCount > 0) ? (double)shortWins / shortCount * 100.0 : 0.0;
    Print("LONG  | Trade: ", IntegerToString(longCount), " | WR: ", DoubleToString(wrLong, 1), "% | Profit: ", DoubleToString(longProfit, 2));
    Print("SHORT | Trade: ", IntegerToString(shortCount), " | WR: ", DoubleToString(wrShort, 1), "% | Profit: ", DoubleToString(shortProfit, 2));

    Print("====================================================");
}

// ══════════════════════════════════════════
// 🎯 GESTIONE DINAMICA POSIZIONI - INVARIATA (solo 2 righe di lettura aggiunte, tag [STATS v5.8])
// ══════════════════════════════════════════
void ManageOpenPositions() {
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        long posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
        int idx = FindSessionMapIndex(posId);
        if(idx < 0) continue;

        double entry  = g_posSessionMap[idx].entryPrice;
        double slDist = g_posSessionMap[idx].slDistance;
        int    dir    = g_posSessionMap[idx].dir;
        if(slDist <= 0) continue;

        double currentPrice = (dir == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double rMultiple = (dir == 1) ? (currentPrice - entry) / slDist : (entry - currentPrice) / slDist;

        // [STATS v5.8] aggiornamento MAE/MFE in R - SOLA LETTURA, non influisce su BE/partial/SL/TP
        if(rMultiple < g_posSessionMap[idx].maeR) g_posSessionMap[idx].maeR = rMultiple;
        if(rMultiple > g_posSessionMap[idx].mfeR) g_posSessionMap[idx].mfeR = rMultiple;

        double volume    = PositionGetDouble(POSITION_VOLUME);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);

        // --- CHIUSURA PARZIALE --- (invariata)
        if(InpUsePartialClose && !g_posSessionMap[idx].partialDone && rMultiple >= InpPartialCloseAtR) {
            double volMin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            double closeVol = MathFloor((volume * (InpPartialClosePct/100.0)) / volStep) * volStep;
            if(closeVol >= volMin && closeVol < volume) {
                if(trade.PositionClosePartial(ticket, closeVol)) {
                    g_posSessionMap[idx].partialDone = true;
                    Print("PARTIAL CLOSE: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Vol: ", DoubleToString(closeVol,2), " @ R=", DoubleToString(rMultiple,2));
                }
            } else {
                g_posSessionMap[idx].partialDone = true;
            }
        }

        // --- BREAK-EVEN --- (invariata)
        if(InpUseBreakEven && !g_posSessionMap[idx].beDone && rMultiple >= InpBreakEvenTriggerR) {
            double beSL = (dir == 1) ? entry + (InpBreakEvenLockR * slDist) : entry - (InpBreakEvenLockR * slDist);
            beSL = NormalizeDouble(beSL, digits);
            bool improves = (dir == 1) ? (beSL > currentSL) : (currentSL == 0 || beSL < currentSL);
            if(improves) {
                if(trade.PositionModify(ticket, beSL, currentTP)) {
                    g_posSessionMap[idx].beDone = true;
                    Print("BREAK-EVEN: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Nuovo SL: ", DoubleToString(beSL,digits));
                }
            }
        }
    }
}

// ══════════════════════════════════════════
// 🔄 MAIN LOGIC - INVARIATA AL 100% DALLA v5.7
// ═══════════════════════════════════════════
void OnTick() {
    ManageOpenPositions();

    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;

    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(today != g_currentDay) {
        g_currentDay = today;
        g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
        g_lonRangeDefined = false; g_nyRangeDefined = false; g_asiaRangeDefined = false;
        g_lonExecutedToday = false; g_nyExecutedToday = false; g_asiaExecutedToday = false;
        g_lonHigh = 0.0; g_lonLow = 999999.0;
        g_nyHigh = 0.0; g_nyLow = 999999.0;
        g_asiaHigh = 0.0; g_asiaLow = 999999.0;
        g_nyReArmActive = false;
        ArrayResize(g_daily_swing_highs, 0); ArrayResize(g_daily_swing_lows, 0);
        ArrayResize(g_countedLossIds, 0);
    }

    if(InpUseEquityDailyLock && g_dayStartEquity > 0) {
        double eq = AccountInfoDouble(ACCOUNT_EQUITY);
        double lossPct = (g_dayStartEquity - eq) / g_dayStartEquity * 100.0;
        if(lossPct >= InpMaxDailyLossPct) g_dayLocked = true;
    }

    if(g_dayLocked) { if(InpShowComments) Comment("STOP TRADING: Max Losses / Daily Loss Limit Reached"); return; }
    if(g_tradesToday >= InpMaxTradesDay) return;
    if(HasOpenPosition()) return;

    static datetime lastD1 = 0;
    if(iTime(_Symbol, PERIOD_D1, 0) != lastD1) {
        lastD1 = iTime(_Symbol, PERIOD_D1, 0);
        double ph = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, true);
        double pl = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, false);
        if(ph != EMPTY_VALUE) { AddToArray(g_daily_swing_highs, ph); if(ArraySize(g_daily_swing_highs)>10) ShiftArray(g_daily_swing_highs); }
        if(pl != EMPTY_VALUE) { AddToArray(g_daily_swing_lows, pl); if(ArraySize(g_daily_swing_lows)>10) ShiftArray(g_daily_swing_lows); }
        g_dailyBias = GetDailyBias();
    }

    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    if(g_nyReArmActive) {
        if(TimeCurrent() > g_nyReArmDeadline) {
            g_nyReArmActive = false;
        } else if(!HasOpenPosition()) {
            double price = (g_nyFadeDir == -1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
            int revDir = -g_nyFadeDir;
            double sl = g_nyFadeSL;
            ExecuteTrade(revDir, price, sl, "NY_REVERSAL", GetRiskAmount(), 0);
            g_nyReArmActive = false;
        }
    }

    // --- LONDON SESSION LOGIC (invariata dalla v5.7) ---
    datetime lonStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpLon_StartHour, InpLon_StartMin));
    datetime lonEnd   = lonStart + (InpLon_DurationMin * 60);
    datetime lonScanLimit = lonEnd + (InpMaxEntryDelayMin * 60);

    if(!g_lonExecutedToday && TimeCurrent() >= lonStart) {
        if(TimeCurrent() < lonEnd) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_lonHigh) g_lonHigh = h;
            if(l < g_lonLow) g_lonLow = l;
            g_lonRangeDefined = true;
        } else if(TimeCurrent() < lonScanLimit && g_lonRangeDefined) {

            bool atrOkLon = true;
            if(InpUseATRFilter) {
                double atrVal = GetATRValue();
                double sessionRange = g_lonHigh - g_lonLow;
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkLon = false;
            }

            if(!atrOkLon) {
                g_lonExecutedToday = true;
            } else {
                double m5_close = iClose(_Symbol, InpEntryTF, 1);
                double m5_open  = iOpen(_Symbol, InpEntryTF, 1);
                double m5_high  = iHigh(_Symbol, InpEntryTF, 1);
                double m5_low   = iLow(_Symbol, InpEntryTF, 1);

                double bodySize = MathAbs(m5_close - m5_open);
                double totalSize = m5_high - m5_low;
                bool validBody = (totalSize > 0) && ((bodySize / totalSize) * 100.0 > InpBodyFilterPct);

                bool bull = (m5_close > g_lonHigh);
                bool bear = (m5_close < g_lonLow);

                if(!InpLon_IgnoreBias && InpLon_UseDailyFilter) {
                    if(bull && g_dailyBias == -1) bull = false;
                    if(bear && g_dailyBias == 1) bear = false;
                }

                if(bull || bear) {
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                        if(bull) ExecuteTrade(1, m5_close, g_lonLow, "LONDON", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_lonHigh, "LONDON", riskAmt, 0);
                    }
                    if(HasOpenPosition()) g_lonExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= lonScanLimit) {
             g_lonExecutedToday = true;
        }
    }

    // --- NEW YORK SESSION LOGIC (invariata dalla v5.7) ---
    datetime nyStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpNY_StartHour, InpNY_StartMin));
    datetime nyEnd   = nyStart + (InpNY_DurationMin * 60);
    datetime nyScanLimit = nyEnd + (InpMaxEntryDelayMin * 60);

    if(!g_nyExecutedToday && TimeCurrent() >= nyStart) {
        if(TimeCurrent() < nyEnd) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_nyHigh) g_nyHigh = h;
            if(l < g_nyLow) g_nyLow = l;
            g_nyRangeDefined = true;
        } else if(TimeCurrent() < nyScanLimit && g_nyRangeDefined) {

            bool atrOkNY = true;
            if(InpUseATRFilter) {
                double atrVal = GetATRValue();
                double sessionRange = g_nyHigh - g_nyLow;
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkNY = false;
            }

            if(!atrOkNY) {
                g_nyExecutedToday = true;
            } else {
                double m5_close = iClose(_Symbol, InpEntryTF, 1);
                double m5_open  = iOpen(_Symbol, InpEntryTF, 1);
                double m5_high  = iHigh(_Symbol, InpEntryTF, 1);
                double m5_low   = iLow(_Symbol, InpEntryTF, 1);

                double bodySize = MathAbs(m5_close - m5_open);
                double totalSize = m5_high - m5_low;
                bool validBody = (totalSize > 0) && ((bodySize / totalSize) * 100.0 > InpBodyFilterPct);

                bool bull = (m5_close > g_nyHigh);
                bool bear = (m5_close < g_nyLow);

                if(InpNY_UseDailyFilter) {
                    if(bull && g_dailyBias == -1) bull = false;
                    if(bear && g_dailyBias == 1) bear = false;
                }

                if(bull || bear) {
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                        if(bull) ExecuteTrade(1, m5_close, g_nyLow, "NY", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_nyHigh, "NY", riskAmt, 0);
                    } else if(InpFadeEnabled) {
                        double fadeRisk = riskAmt * (InpFadeRiskPct / 100.0);
                        double buffer = InpFadeSLBufferPoints * point;
                        if(bull) {
                            double sl = m5_high + buffer;
                            g_nyFadeDir = -1; g_nyFadeSL = sl;
                            ExecuteTrade(-1, m5_close, sl, "NY_FADE", fadeRisk, g_nyLow);
                        } else if(bear) {
                            double sl = m5_low - buffer;
                            g_nyFadeDir = 1; g_nyFadeSL = sl;
                            ExecuteTrade(1, m5_close, sl, "NY_FADE", fadeRisk, g_nyHigh);
                        }
                    }
                    if(HasOpenPosition()) g_nyExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= nyScanLimit) {
             g_nyExecutedToday = true;
        }
    }

    // --- ASIAN SESSION LOGIC (invariata dalla v5.7) ---
    if(InpUseAsia) {
        datetime asiaStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpAsia_StartHour, InpAsia_StartMin));
        datetime asiaEnd   = asiaStart + (InpAsia_DurationMin * 60);
        datetime asiaScanLimit = asiaEnd + (InpMaxEntryDelayMin * 60);

        if(!g_asiaExecutedToday && TimeCurrent() >= asiaStart) {
            if(TimeCurrent() < asiaEnd) {
                double h = iHigh(_Symbol, PERIOD_M1, 1);
                double l = iLow(_Symbol, PERIOD_M1, 1);
                if(h > g_asiaHigh) g_asiaHigh = h;
                if(l < g_asiaLow) g_asiaLow = l;
                g_asiaRangeDefined = true;
            } else if(TimeCurrent() < asiaScanLimit && g_asiaRangeDefined) {

                bool atrOkAsia = true;
                if(InpUseATRFilter) {
                    double atrVal = GetATRValue();
                    double sessionRange = g_asiaHigh - g_asiaLow;
                    if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkAsia = false;
                }

                if(!atrOkAsia) {
                    g_asiaExecutedToday = true;
                } else {
                    double m5_close = iClose(_Symbol, InpEntryTF, 1);
                    double m5_open  = iOpen(_Symbol, InpEntryTF, 1);
                    double m5_high  = iHigh(_Symbol, InpEntryTF, 1);
                    double m5_low   = iLow(_Symbol, InpEntryTF, 1);

                    double bodySize = MathAbs(m5_close - m5_open);
                    double totalSize = m5_high - m5_low;
                    bool validBody = (totalSize > 0) && ((bodySize / totalSize) * 100.0 > InpBodyFilterPct);

                    bool bull = (m5_close > g_asiaHigh);
                    bool bear = (m5_close < g_asiaLow);

                    if(InpAsia_UseDailyFilter) {
                        if(bull && g_dailyBias == -1) bull = false;
                        if(bear && g_dailyBias == 1) bear = false;
                    }

                    if(bull || bear) {
                        double riskAmt = GetRiskAmount();
                        if(validBody) {
                            if(bull) ExecuteTrade(1, m5_close, g_asiaLow, "ASIA", riskAmt, 0);
                            else if(bear) ExecuteTrade(-1, m5_close, g_asiaHigh, "ASIA", riskAmt, 0);
                        } else if(InpFadeEnabled) {
                            double fadeRisk = riskAmt * (InpFadeRiskPct / 100.0);
                            double buffer = InpFadeSLBufferPoints * point;
                            if(bull) {
                                double sl = m5_high + buffer;
                                ExecuteTrade(-1, m5_close, sl, "ASIA_FADE", fadeRisk, g_asiaLow);
                            } else if(bear) {
                                double sl = m5_low - buffer;
                                ExecuteTrade(1, m5_close, sl, "ASIA_FADE", fadeRisk, g_asiaHigh);
                            }
                        }
                        if(HasOpenPosition()) g_asiaExecutedToday = true;
                    }
                }
            } else if(TimeCurrent() >= asiaScanLimit) {
                g_asiaExecutedToday = true;
            }
        }
    }

    if(InpShowComments) {
        Comment(StringFormat("Tri-Session v5.8 (logica v5.7 + stats) | Bias: %s | Lon H/L: %.2f/%.2f | NY H/L: %.2f/%.2f | Asia H/L: %.2f/%.2f | Trades: %d",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")),
              g_lonHigh, g_lonLow, g_nyHigh, g_nyLow, g_asiaHigh, g_asiaLow, g_tradesToday));
    }
}

// ══════════════════════════════════════════
// 💼 EXECUTION - IDENTICA alla v5.7, solo con cattura dati aggiuntivi dopo l'apertura
// ══════════════════════════════════════════
void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    double sl_final = NormalizeDouble(sl_ref, digits);

    double tp_final = 0;
    if(customTP > 0) {
        tp_final = NormalizeDouble(customTP, digits);
    } else {
        double dist = MathAbs(entry - sl_final);
        if(dist == 0) return;
        tp_final = NormalizeDouble((dir == 1) ? entry + (dist * InpRR_Target) : entry - (dist * InpRR_Target), digits);
    }

    if(dir == 1 && tp_final <= entry) {
        Print("DEBUG ExecuteTrade: SCARTATO (", session, ") - TP ", DoubleToString(tp_final,digits),
              " <= entry ", DoubleToString(entry,digits));
        return;
    }
    if(dir == -1 && tp_final >= entry) {
        Print("DEBUG ExecuteTrade: SCARTATO (", session, ") - TP ", DoubleToString(tp_final,digits),
              " >= entry ", DoubleToString(entry,digits));
        return;
    }

    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double profitPerLot = 0;

    bool calcRes = false;
    if(dir == 1) calcRes = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, sl_final, profitPerLot);
    else calcRes = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, sl_final, profitPerLot);

    if(!calcRes || profitPerLot >= 0) return;

    double lots = MathFloor((riskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
    if(lots < minLot) lots = minLot;

    // [STATS v5.8] reward teorico al prezzo di TP, con lo stesso metodo (OrderCalcProfit)
    // gia' usato dal codice originale per il rischio. Puramente informativo, non altera
    // ne' lots ne' tp_final ne' alcuna decisione successiva.
    double rewardPerLot = 0;
    bool rewardCalcRes = false;
    if(dir == 1) rewardCalcRes = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, tp_final, rewardPerLot);
    else rewardCalcRes = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, tp_final, rewardPerLot);
    double riskMoneyAtOpen = MathAbs(profitPerLot) * lots;
    double rewardMoneyAtOpen = rewardCalcRes ? MathAbs(rewardPerLot) * lots : 0;

    string cmt = StringFormat("%s|v5.8|R:%.2f", session, riskAmount);
    bool res = false;

    if(dir == 1) res = trade.Buy(lots, _Symbol, entry, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, _Symbol, entry, sl_final, tp_final, cmt);

    if(res) {
        g_tradesToday++;

        long posId = 0;
        double actualFillPrice = 0; // [STATS v5.8]
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong t = PositionGetTicket(i);
            if(t == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
            posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
            actualFillPrice = PositionGetDouble(POSITION_PRICE_OPEN); // [STATS v5.8] lettura, nessun impatto
            break;
        }
        if(posId > 0) RegisterPosition(posId, session, entry, MathAbs(entry - sl_final), dir,
                                        tp_final, entry, actualFillPrice, lots, riskMoneyAtOpen, rewardMoneyAtOpen);

        if(session == "LONDON") g_totalLon++;
        else if(session == "NY") g_totalNY++;
        else if(session == "ASIA") g_totalAsia++;
        else if(session == "NY_FADE") g_totalNYFade++;
        else if(session == "ASIA_FADE") g_totalAsiaFade++;
        else if(session == "NY_REVERSAL") g_totalNYRev++;

        Print("TRADE OPENED: ", session, " | PosID: ", posId, " | Risk: ", DoubleToString(riskAmount,2), " | TP: ", DoubleToString(tp_final, digits));
    }
}

// ══════════════════════════════════════════
// 📊 STATS TRACKING - la logica esistente e' invariata; ho solo aggiunto
// l'accumulo separato di commissioni/swap e la chiamata a LogTradeFull
// ══════════════════════════════════════════
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;

    long posId = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
    int dealEntry = (int)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);

    if(dealEntry == DEAL_ENTRY_OUT) {
        string session = GetSessionByPosId(posId);
        if(session == "UNKNOWN") return;

        if(!HistorySelectByPosition(posId)) return;

        double totalPnL = 0;
        double rawProfitSum = 0, commissionSum = 0, swapSum = 0; // [STATS v5.8]
        int dealsCount = HistoryDealsTotal();
        for(int i=0; i<dealsCount; i++) {
            ulong dticket = HistoryDealGetTicket(i);
            if(HistoryDealGetInteger(dticket, DEAL_POSITION_ID) == posId &&
               HistoryDealGetInteger(dticket, DEAL_MAGIC) == (long)InpMagicNumber &&
               HistoryDealGetInteger(dticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                double p = HistoryDealGetDouble(dticket, DEAL_PROFIT);
                double s = HistoryDealGetDouble(dticket, DEAL_SWAP);
                double c = HistoryDealGetDouble(dticket, DEAL_COMMISSION);
                totalPnL += p + s + c;
                rawProfitSum += p;   // [STATS v5.8]
                swapSum      += s;   // [STATS v5.8]
                commissionSum+= c;   // [STATS v5.8]
            }
        }

        bool stillOpen = false;
        for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
        if(stillOpen) return;

        if(!IsIdAlreadyCounted(posId)) {
            int idx = FindSessionMapIndex(posId);
            datetime entryTimeForLog = TimeCurrent();
            int dirForLog = 0;
            // [STATS v5.8] variabili estese, recuperate PRIMA di rimuovere la posizione dalla mappa
            string sessLblForLog = "UNKNOWN", setupLblForLog = "UNKNOWN";
            double lotsForLog = 0, entryReqForLog = 0, actualFillForLog = 0;
            double initialSLForLog = 0, initialTPForLog = 0;
            double riskMoneyForLog = 0, rewardMoneyForLog = 0;
            double maeRForLog = 0, mfeRForLog = 0;
            if(idx >= 0) {
                entryTimeForLog   = g_posSessionMap[idx].entryTime;
                dirForLog         = g_posSessionMap[idx].dir;
                sessLblForLog     = g_posSessionMap[idx].sessionLabel;
                setupLblForLog    = g_posSessionMap[idx].setupLabel;
                lotsForLog        = g_posSessionMap[idx].lots;
                entryReqForLog    = g_posSessionMap[idx].requestedPrice;
                actualFillForLog  = g_posSessionMap[idx].actualFillPrice;
                initialSLForLog   = g_posSessionMap[idx].entryPrice - (g_posSessionMap[idx].dir==1 ? g_posSessionMap[idx].slDistance : -g_posSessionMap[idx].slDistance);
                initialTPForLog   = g_posSessionMap[idx].initialTP;
                riskMoneyForLog   = g_posSessionMap[idx].riskMoneyAtOpen;
                rewardMoneyForLog = g_posSessionMap[idx].rewardMoneyAtOpen;
                maeRForLog        = g_posSessionMap[idx].maeR;
                mfeRForLog        = g_posSessionMap[idx].mfeR;
            }

            if(session == "LONDON") { g_profitLon += totalPnL; if(totalPnL > 0) g_winsLon++; }
            else if(session == "NY") { g_profitNY += totalPnL; if(totalPnL > 0) g_winsNY++; }
            else if(session == "ASIA") { g_profitAsia += totalPnL; if(totalPnL > 0) g_winsAsia++; }
            else if(session == "NY_FADE") { g_profitNYFade += totalPnL; if(totalPnL > 0) g_winsNYFade++; }
            else if(session == "ASIA_FADE") { g_profitAsiaFade += totalPnL; if(totalPnL > 0) g_winsAsiaFade++; }
            else if(session == "NY_REVERSAL") { g_profitNYRev += totalPnL; if(totalPnL > 0) g_winsNYRev++; }

            // Log originale (INVARIATO)
            LogTrade(entryTimeForLog, TimeCurrent(), totalPnL, dirForLog, session);

            // [STATS v5.8] Log esteso, in affiancamento
            if(idx >= 0) {
                LogTradeFull(posId, entryTimeForLog, TimeCurrent(),
                             sessLblForLog, setupLblForLog, dirForLog, lotsForLog,
                             entryReqForLog, actualFillForLog, initialSLForLog, initialTPForLog,
                             riskMoneyForLog, rewardMoneyForLog,
                             rawProfitSum, commissionSum, swapSum,
                             maeRForLog, mfeRForLog);
            }

            AddToLongArray(g_countedLossIds, posId);
            if(totalPnL < 0) {
                g_lossesToday++;
                if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;

                if(session == "NY_FADE") {
                    g_nyReArmActive = true;
                    g_nyReArmDeadline = TimeCurrent() + 30*60;
                }
            }
            UnregisterPosition(posId);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// [STATS v5.8] SEZIONE COMPLETAMENTE NUOVA: aggregazioni, robustness test,
// cost analysis, quantitative summary, controlli di coerenza ed export CSV.
// Tutto cio' che segue lavora SOLO sul dataset g_statsLog[] gia' registrato,
// senza mai toccare Entry/SL/TP/timing dell'EA.
// ═══════════════════════════════════════════════════════════════════════
struct AggRes {
    int    trades, wins, losses;
    double winRate;
    double grossProfit, grossLoss, netProfit, pf;
    double avgWin, avgLoss, avgR, expectancy;
    double avgMAE, avgMFE, maxWin, maxLoss, maxDD;
    int    maxWinStreak, maxLossStreak;
};

// Calcola l'aggregato su un sottoinsieme di indici di g_statsLog (ordine cronologico preservato)
AggRes ComputeAggregate(int &idxList[]) {
    AggRes r;
    r.trades=0; r.wins=0; r.losses=0; r.winRate=0;
    r.grossProfit=0; r.grossLoss=0; r.netProfit=0; r.pf=0;
    r.avgWin=0; r.avgLoss=0; r.avgR=0; r.expectancy=0;
    r.avgMAE=0; r.avgMFE=0; r.maxWin=0; r.maxLoss=0; r.maxDD=0;
    r.maxWinStreak=0; r.maxLossStreak=0;

    int n = ArraySize(idxList);
    r.trades = n;
    if(n==0) return r;

    double sumR=0, peak=0, eq=0;
    int curWinStreak=0, curLossStreak=0;

    for(int k=0; k<n; k++) {
        int i = idxList[k];
        double p = g_statsLog[i].profit + g_statsLog[i].commission + g_statsLog[i].swap; // profitto netto originale (no costi extra stimati)
        double rr = (g_statsLog[i].riskMoney>0) ? p/g_statsLog[i].riskMoney : 0;
        sumR += rr;
        r.avgMAE += g_statsLog[i].maeMoney;
        r.avgMFE += g_statsLog[i].mfeMoney;

        if(p>0) {
            r.wins++; r.grossProfit += p; if(p>r.maxWin) r.maxWin=p;
            curWinStreak++; curLossStreak=0;
            if(curWinStreak>r.maxWinStreak) r.maxWinStreak=curWinStreak;
        } else if(p<0) {
            r.losses++; r.grossLoss += -p; if(p<r.maxLoss) r.maxLoss=p;
            curLossStreak++; curWinStreak=0;
            if(curLossStreak>r.maxLossStreak) r.maxLossStreak=curLossStreak;
        }
        r.netProfit += p;
        eq += p;
        if(eq>peak) peak=eq;
        double dd = peak-eq;
        if(dd>r.maxDD) r.maxDD=dd;
    }

    r.winRate = (double)r.wins/n*100.0;
    r.pf = (r.grossLoss>0) ? r.grossProfit/r.grossLoss : 0;
    r.avgWin = (r.wins>0) ? r.grossProfit/r.wins : 0;
    r.avgLoss = (r.losses>0) ? -r.grossLoss/r.losses : 0;
    r.avgR = sumR/n;
    r.expectancy = r.netProfit/n;
    r.avgMAE /= n;
    r.avgMFE /= n;
    return r;
}

void BuildAllIndexes(int &idx[]) {
    int n = ArraySize(g_statsLog);
    ArrayResize(idx, n);
    for(int i=0;i<n;i++) idx[i]=i;
}

void BuildIndexesBySessionSetup(string sessFilter, string setupFilter, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        if(g_statsLog[i].sessionLabel==sessFilter && g_statsLog[i].setupLabel==setupFilter) {
            int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i;
        }
    }
}

void BuildIndexesExcludeSession(string sess, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        if(g_statsLog[i].sessionLabel != sess) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
    }
}

void BuildIndexesExcludeDow(int dow, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        if(g_statsLog[i].dowNum != dow) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
    }
}

void BuildIndexesByDow(int dow, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        if(g_statsLog[i].dowNum == dow) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
    }
}

void BuildIndexesByHour(int hour, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        MqlDateTime dtE; TimeToStruct(g_statsLog[i].openTime, dtE);
        if(dtE.hour == hour) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
    }
}

void BuildIndexesByMonth(int ym, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        MqlDateTime dtE; TimeToStruct(g_statsLog[i].openTime, dtE);
        int thisYm = dtE.year*100+dtE.mon;
        if(thisYm == ym) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
    }
}

void BuildIndexesByDir(int dir, int &idx[]) {
    ArrayResize(idx,0);
    int n = ArraySize(g_statsLog);
    for(int i=0;i<n;i++) {
        if(g_statsLog[i].dir == dir) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
    }
}

// Esclude i top N trade per "Profit €" (campo .profit, come da colonna richiesta)
void BuildIndexesExcludeTopN(int topN, int &idx[]) {
    int n = ArraySize(g_statsLog);
    int order[]; ArrayResize(order, n);
    for(int i=0;i<n;i++) order[i]=i;
    // ordinamento a bolle per profit decrescente (dataset di backtest, dimensioni contenute)
    for(int a=0;a<n-1;a++)
        for(int b=0;b<n-1-a;b++)
            if(g_statsLog[order[b]].profit < g_statsLog[order[b+1]].profit) {
                int tmp=order[b]; order[b]=order[b+1]; order[b+1]=tmp;
            }
    bool excluded[]; ArrayResize(excluded, n); ArrayInitialize(excluded, false);
    for(int k=0; k<topN && k<n; k++) excluded[order[k]] = true;

    ArrayResize(idx,0);
    for(int i=0;i<n;i++) if(!excluded[i]) { int s=ArraySize(idx); ArrayResize(idx,s+1); idx[s]=i; }
}

void WriteAggRow(int fh, string label, AggRes &r) {
    FileWrite(fh, label,
              IntegerToString(r.trades), IntegerToString(r.wins), IntegerToString(r.losses),
              DoubleToString(r.winRate,2),
              DoubleToString(r.grossProfit,2), DoubleToString(r.grossLoss,2), DoubleToString(r.netProfit,2),
              DoubleToString(r.pf,2),
              DoubleToString(r.avgWin,2), DoubleToString(r.avgLoss,2), DoubleToString(r.avgR,3),
              DoubleToString(r.expectancy,2),
              DoubleToString(r.avgMAE,2), DoubleToString(r.avgMFE,2),
              DoubleToString(r.maxWin,2), DoubleToString(r.maxLoss,2),
              IntegerToString(r.maxWinStreak), IntegerToString(r.maxLossStreak),
              DoubleToString(r.maxDD,2));
}

void WriteAggHeader(int fh) {
    FileWrite(fh, "Label","Trades","Wins","Losses","WinRate%","GrossProfit","GrossLoss","NetProfit",
              "ProfitFactor","AvgWin","AvgLoss","AvgR","Expectancy","AvgMAE€","AvgMFE€",
              "MaxWin","MaxLoss","MaxWinStreak","MaxLossStreak","MaxDD€");
}

void ExportStatsToCSV() {
    int n = ArraySize(g_statsLog);
    if(n==0) {
        Print("[STATS v5.8] Nessun trade nel dataset esteso: CSV non generato.");
        return;
    }

    int fh = FileOpen(InpCSVFileName, FILE_WRITE|FILE_CSV|FILE_ANSI, ',');
    if(fh==INVALID_HANDLE) {
        Print("[STATS v5.8] ERRORE: impossibile creare il file CSV '", InpCSVFileName, "' (errore ", GetLastError(), ")");
        return;
    }

    // ─── NOTE METODOLOGICHE (documentate come richiesto, prima delle tabelle) ───
    FileWrite(fh, "=== NOTE METODOLOGICHE ===");
    FileWrite(fh, "News: nessun filtro news presente nell'EA originale -> sempre N/D.");
    FileWrite(fh, "Slippage: differenza reale tra prezzo richiesto e prezzo di fill effettivo, convertita in € tramite il rapporto Rischio€/Distanza(Entry-SL).");
    FileWrite(fh, "Spread Cost: formula fissa specificata dall'utente (lots/0.01 * "+DoubleToString(InpSpreadCostPer001Lot,3)+" €), NON e' il costo reale del tester: e' una stima supplementare, non va confusa con lo spread gia' incorporato nei prezzi bid/ask della simulazione.");
    FileWrite(fh, "MAE/MFE €: derivati da MAE/MFE in R moltiplicati per il Rischio€ registrato all'apertura. Approssimazione: se e' avvenuta una chiusura parziale, la size e' cambiata durante la vita del trade.");
    FileWrite(fh, "TEST 1R/2R/3R e simulazioni di trailing/BE/partial: NON SIMULABILI con i dati attualmente disponibili, perche' il sistema registra MAE/MFE come valori estremi (min/max) ma non la sequenza temporale con cui sono stati raggiunti; senza quella sequenza non si puo' stabilire con certezza se un target sarebbe stato toccato prima dello stop.");
    FileWrite(fh, "");

    // ─── TABLE 1 - ALL TRADES ───
    FileWrite(fh, "=== TABLE 1 - ALL TRADES ===");
    FileWrite(fh,"Ticket(PosID)","Date","Time","Day","Session","Setup","Direction","Lots",
              "Entry","Initial SL","Initial TP","Risk €","Reward €","RR Theoretical","RR Realized",
              "Profit €","Spread Cost €","Commission €","Swap €","Slippage €","Net Profit After Costs €",
              "MAE €","MAE R","MFE €","MFE R","Duration(sec)","Duration(min)","News","Result");

    double sumProfitAll=0;
    for(int i=0;i<n;i++) {
        string dirStr = (g_statsLog[i].dir==1) ? "LONG" : "SHORT";
        FileWrite(fh,
            IntegerToString(g_statsLog[i].posId),
            g_statsLog[i].dateStr, g_statsLog[i].timeStr, g_statsLog[i].dayName,
            g_statsLog[i].sessionLabel, g_statsLog[i].setupLabel, dirStr,
            DoubleToString(g_statsLog[i].lots,2),
            DoubleToString(g_statsLog[i].entryReq,5),
            DoubleToString(g_statsLog[i].initialSL,5),
            DoubleToString(g_statsLog[i].initialTP,5),
            DoubleToString(g_statsLog[i].riskMoney,2),
            DoubleToString(g_statsLog[i].rewardMoney,2),
            DoubleToString(g_statsLog[i].rrTheoretical,2),
            DoubleToString(g_statsLog[i].rrRealized,2),
            DoubleToString(g_statsLog[i].profit,2),
            DoubleToString(g_statsLog[i].spreadCost,3),
            DoubleToString(g_statsLog[i].commission,2),
            DoubleToString(g_statsLog[i].swap,2),
            DoubleToString(g_statsLog[i].slippage,2),
            DoubleToString(g_statsLog[i].netAfterCosts,2),
            DoubleToString(g_statsLog[i].maeMoney,2),
            DoubleToString(g_statsLog[i].maeR,2),
            DoubleToString(g_statsLog[i].mfeMoney,2),
            DoubleToString(g_statsLog[i].mfeR,2),
            IntegerToString((int)g_statsLog[i].durationSec),
            DoubleToString(g_statsLog[i].durationSec/60.0,1),
            g_statsLog[i].news,
            g_statsLog[i].result);
        sumProfitAll += g_statsLog[i].profit + g_statsLog[i].commission + g_statsLog[i].swap;
    }
    FileWrite(fh,"");

    // ─── TABLE 2 - SESSION/SETUP ANALYSIS ───
    FileWrite(fh, "=== TABLE 2 - SESSION ANALYSIS ===");
    WriteAggHeader(fh);
    int idxT[];
    BuildIndexesBySessionSetup("LONDON","NORMAL",idxT);   AggRes rLon=ComputeAggregate(idxT); WriteAggRow(fh,"LONDON",rLon);
    BuildIndexesBySessionSetup("NY","NORMAL",idxT);        AggRes rNy=ComputeAggregate(idxT);  WriteAggRow(fh,"NY",rNy);
    BuildIndexesBySessionSetup("ASIA","NORMAL",idxT);      AggRes rAsia=ComputeAggregate(idxT);WriteAggRow(fh,"ASIA",rAsia);
    BuildIndexesBySessionSetup("NY","NY_FADE",idxT);       AggRes rNyFade=ComputeAggregate(idxT); WriteAggRow(fh,"NY_FADE",rNyFade);
    BuildIndexesBySessionSetup("ASIA","ASIA_FADE",idxT);   AggRes rAsiaFade=ComputeAggregate(idxT); WriteAggRow(fh,"ASIA_FADE",rAsiaFade);
    BuildIndexesBySessionSetup("NY","NY_REV",idxT);        AggRes rNyRev=ComputeAggregate(idxT); WriteAggRow(fh,"NY_REV",rNyRev);
    FileWrite(fh,"");

    // ─── TABLE 3 - MONTHLY ANALYSIS ───
    FileWrite(fh, "=== TABLE 3 - MONTHLY ANALYSIS ===");
    WriteAggHeader(fh);
    int monthsSeen[]; ArrayResize(monthsSeen,0);
    for(int i=0;i<n;i++) {
        MqlDateTime dtE; TimeToStruct(g_statsLog[i].openTime, dtE);
        int ym = dtE.year*100+dtE.mon;
        bool found=false;
        for(int k=0;k<ArraySize(monthsSeen);k++) if(monthsSeen[k]==ym) { found=true; break; }
        if(!found) { int s=ArraySize(monthsSeen); ArrayResize(monthsSeen,s+1); monthsSeen[s]=ym; }
    }
    // ordina i mesi crescente
    for(int a=0;a<ArraySize(monthsSeen)-1;a++)
        for(int b=0;b<ArraySize(monthsSeen)-1-a;b++)
            if(monthsSeen[b]>monthsSeen[b+1]) { int t=monthsSeen[b]; monthsSeen[b]=monthsSeen[b+1]; monthsSeen[b+1]=t; }
    for(int k=0;k<ArraySize(monthsSeen);k++) {
        BuildIndexesByMonth(monthsSeen[k], idxT);
        AggRes rM = ComputeAggregate(idxT);
        string lbl = StringFormat("%04d-%02d", monthsSeen[k]/100, monthsSeen[k]%100);
        WriteAggRow(fh, lbl, rM);
    }
    FileWrite(fh,"");

    // ─── TABLE 4 - HOURLY ANALYSIS ───
    FileWrite(fh, "=== TABLE 4 - HOURLY ANALYSIS ===");
    WriteAggHeader(fh);
    for(int h=0; h<24; h++) {
        BuildIndexesByHour(h, idxT);
        if(ArraySize(idxT)==0) continue;
        AggRes rH = ComputeAggregate(idxT);
        WriteAggRow(fh, StringFormat("%02d:00",h), rH);
    }
    FileWrite(fh,"");

    // ─── TABLE 5 - WEEKDAY ANALYSIS ───
    FileWrite(fh, "=== TABLE 5 - WEEKDAY ANALYSIS ===");
    WriteAggHeader(fh);
    for(int d=1; d<=5; d++) { // 1=Monday..5=Friday
        BuildIndexesByDow(d, idxT);
        if(ArraySize(idxT)==0) continue;
        AggRes rD = ComputeAggregate(idxT);
        WriteAggRow(fh, DowNameEN(d), rD);
    }
    FileWrite(fh,"");

    // ─── TABLE 6 - LONG VS SHORT ───
    FileWrite(fh, "=== TABLE 6 - LONG VS SHORT ===");
    WriteAggHeader(fh);
    BuildIndexesByDir(1, idxT);  AggRes rLong = ComputeAggregate(idxT);  WriteAggRow(fh,"LONG",rLong);
    BuildIndexesByDir(-1, idxT); AggRes rShort = ComputeAggregate(idxT); WriteAggRow(fh,"SHORT",rShort);
    FileWrite(fh,"");

    // ─── TABLE 7 - ROBUSTNESS TESTS ───
    FileWrite(fh, "=== TABLE 7 - ROBUSTNESS TESTS ===");
    WriteAggHeader(fh);
    BuildAllIndexes(idxT); AggRes rAll = ComputeAggregate(idxT); WriteAggRow(fh,"ALL TRADES (baseline)", rAll);
    BuildIndexesExcludeSession("ASIA", idxT); AggRes rNoAsia = ComputeAggregate(idxT); WriteAggRow(fh,"REMOVE ASIA", rNoAsia);
    BuildIndexesExcludeDow(5, idxT); AggRes rNoFri = ComputeAggregate(idxT); WriteAggRow(fh,"REMOVE FRIDAY", rNoFri); // 5=Venerdi in day_of_week MQL5
    BuildIndexesExcludeTopN(2, idxT); AggRes rNoTop2 = ComputeAggregate(idxT); WriteAggRow(fh,"REMOVE TOP 2 TRADES", rNoTop2);
    BuildIndexesExcludeTopN(5, idxT); AggRes rNoTop5 = ComputeAggregate(idxT); WriteAggRow(fh,"REMOVE TOP 5 TRADES", rNoTop5);
    FileWrite(fh, "TEST 1R/2R/3R", "Non simulabile con i dati attualmente disponibili (vedi note metodologiche)");
    FileWrite(fh, "TEST TRAILING/BE/PARTIAL SIMULATI", "Non simulabile con i dati attualmente disponibili (vedi note metodologiche)");
    FileWrite(fh,"");

    // ─── TABLE 8 - COST ANALYSIS ───
    FileWrite(fh, "=== TABLE 8 - COST ANALYSIS ===");
    double sumSpread=0, sumSlippage=0, sumCommission=0, sumSwap=0, sumRawProfit=0;
    for(int i=0;i<n;i++) {
        sumSpread += g_statsLog[i].spreadCost;
        sumSlippage += g_statsLog[i].slippage;
        sumCommission += g_statsLog[i].commission;
        sumSwap += g_statsLog[i].swap;
        sumRawProfit += g_statsLog[i].profit;
    }
    double originalNet = sumRawProfit + sumCommission + sumSwap;
    double afterCostsNet = originalNet - sumSpread - sumSlippage;
    FileWrite(fh, "Scenario","Net Profit €");
    FileWrite(fh, "A - Original (tester)", DoubleToString(originalNet,2));
    FileWrite(fh, "B - After Costs (Spread stimato + Slippage reale)", DoubleToString(afterCostsNet,2));
    FileWrite(fh, "Differenza (A-B)", DoubleToString(originalNet-afterCostsNet,2));
    FileWrite(fh, "Totale Spread Cost stimato €", DoubleToString(sumSpread,2));
    FileWrite(fh, "Totale Slippage €", DoubleToString(sumSlippage,2));
    FileWrite(fh, "Totale Commissioni €", DoubleToString(sumCommission,2));
    FileWrite(fh, "Totale Swap €", DoubleToString(sumSwap,2));
    FileWrite(fh,"");

    // ─── TABLE 9 - QUANTITATIVE SUMMARY ───
    FileWrite(fh, "=== TABLE 9 - QUANTITATIVE SUMMARY ===");
    // equity curve cronologica (ordine di chiusura = ordine in g_statsLog) per DD/Recovery Factor
    double peakEq=0, eq=0, maxDDMoney=0;
    double rValues[]; ArrayResize(rValues, n);
    double sumR=0;
    for(int i=0;i<n;i++) {
        double p = g_statsLog[i].profit + g_statsLog[i].commission + g_statsLog[i].swap;
        eq += p;
        if(eq>peakEq) peakEq=eq;
        double dd = peakEq-eq;
        if(dd>maxDDMoney) maxDDMoney=dd;
        double rr = (g_statsLog[i].riskMoney>0) ? p/g_statsLog[i].riskMoney : 0;
        rValues[i]=rr;
        sumR += rr;
    }
    double avgR = sumR/n;
    // deviazione standard e mediana di R
    double sumSqDiff=0;
    for(int i=0;i<n;i++) sumSqDiff += (rValues[i]-avgR)*(rValues[i]-avgR);
    double stdevR = (n>1) ? MathSqrt(sumSqDiff/(n-1)) : 0;
    // mediana (ordinamento copia)
    double sortedR[]; ArrayResize(sortedR,n);
    for(int i=0;i<n;i++) sortedR[i]=rValues[i];
    for(int a=0;a<n-1;a++) for(int b=0;b<n-1-a;b++) if(sortedR[b]>sortedR[b+1]) { double t=sortedR[b]; sortedR[b]=sortedR[b+1]; sortedR[b+1]=t; }
    double medianR = (n%2==1) ? sortedR[n/2] : (sortedR[n/2-1]+sortedR[n/2])/2.0;
    // downside deviation per Sortino
    double sumSqNeg=0; int cntNeg=0;
    for(int i=0;i<n;i++) if(rValues[i]<0) { sumSqNeg += rValues[i]*rValues[i]; cntNeg++; }
    double downsideDev = (cntNeg>0) ? MathSqrt(sumSqNeg/cntNeg) : 0;

    double grossProfitAll=0, grossLossAll=0; int winsAll=0, lossesAll=0;
    for(int i=0;i<n;i++) {
        double p = g_statsLog[i].profit + g_statsLog[i].commission + g_statsLog[i].swap;
        if(p>0) { grossProfitAll+=p; winsAll++; } else if(p<0) { grossLossAll += -p; lossesAll++; }
    }
    double netProfitAll = grossProfitAll - grossLossAll;
    double pfAll = (grossLossAll>0) ? grossProfitAll/grossLossAll : 0;
    double avgWinAll = (winsAll>0) ? grossProfitAll/winsAll : 0;
    double avgLossAll = (lossesAll>0) ? grossLossAll/lossesAll : 0;
    double payoffRatio = (avgLossAll>0) ? avgWinAll/avgLossAll : 0;
    double winRateAll = (double)winsAll/n*100.0;
    double expectancyMoney = netProfitAll/n;
    double ddPct = (peakEq>0) ? (maxDDMoney/peakEq*100.0) : 0;
    double recoveryFactor = (maxDDMoney>0) ? netProfitAll/maxDDMoney : 0;
    double sharpe = (stdevR>0 && n>=20) ? avgR/stdevR : 0;
    double sortino = (downsideDev>0 && n>=20) ? avgR/downsideDev : 0;
    double calmar = (maxDDMoney>0 && n>=20) ? netProfitAll/maxDDMoney : 0; // nota: forma semplificata, coincide con Recovery Factor

    FileWrite(fh, "Metrica","Valore");
    FileWrite(fh, "Total Trades", IntegerToString(n));
    FileWrite(fh, "Win Rate %", DoubleToString(winRateAll,2));
    FileWrite(fh, "Gross Profit €", DoubleToString(grossProfitAll,2));
    FileWrite(fh, "Gross Loss €", DoubleToString(grossLossAll,2));
    FileWrite(fh, "Net Profit €", DoubleToString(netProfitAll,2));
    FileWrite(fh, "Profit Factor", DoubleToString(pfAll,2));
    FileWrite(fh, "Average Win €", DoubleToString(avgWinAll,2));
    FileWrite(fh, "Average Loss €", DoubleToString(avgLossAll,2));
    FileWrite(fh, "Payoff Ratio", DoubleToString(payoffRatio,2));
    FileWrite(fh, "Expectancy €", DoubleToString(expectancyMoney,2));
    FileWrite(fh, "Expectancy R", DoubleToString(avgR,3));
    FileWrite(fh, "Maximum Drawdown €", DoubleToString(maxDDMoney,2));
    FileWrite(fh, "Maximum Drawdown %", DoubleToString(ddPct,2));
    FileWrite(fh, "Recovery Factor", DoubleToString(recoveryFactor,2));
    FileWrite(fh, "Average R", DoubleToString(avgR,3));
    FileWrite(fh, "Median R", DoubleToString(medianR,3));
    FileWrite(fh, "Std Dev of R", DoubleToString(stdevR,3));
    FileWrite(fh, "Sharpe Ratio (semplificato, per-trade, non annualizzato)", (n>=20)?DoubleToString(sharpe,3):"N/D - campione <20 trade");
    FileWrite(fh, "Sortino Ratio (semplificato, per-trade, non annualizzato)", (n>=20)?DoubleToString(sortino,3):"N/D - campione <20 trade");
    FileWrite(fh, "Calmar Ratio (approssimato, coincide con Recovery Factor in questa forma semplificata)", (n>=20)?DoubleToString(calmar,3):"N/D - campione <20 trade");
    FileWrite(fh, "Maximum Winning Streak", IntegerToString(rAll.maxWinStreak));
    FileWrite(fh, "Maximum Losing Streak", IntegerToString(rAll.maxLossStreak));
    FileWrite(fh,"");

    // ─── CONTROLLO DI COERENZA ───
    FileWrite(fh, "=== CONSISTENCY CHECK ===");
    double sumOriginalCategories = g_profitLon+g_profitNY+g_profitAsia+g_profitNYFade+g_profitAsiaFade+g_profitNYRev;
    int totOriginalCount = g_totalLon+g_totalNY+g_totalAsia+g_totalNYFade+g_totalAsiaFade+g_totalNYRev;
    double diffProfit = MathAbs(sumOriginalCategories - originalNet);
    bool profitOk = diffProfit < 0.01;
    bool countOk = (n == totOriginalCount);
    FileWrite(fh, "Somma Profit (ALL TRADES esteso) €", DoubleToString(originalNet,2));
    FileWrite(fh, "Somma Profit (statistiche originali per sessione) €", DoubleToString(sumOriginalCategories,2));
    FileWrite(fh, "Match Profit", profitOk ? "OK" : "WARNING: DISCREPANZA RILEVATA");
    FileWrite(fh, "Numero trade (ALL TRADES esteso)", IntegerToString(n));
    FileWrite(fh, "Numero trade (statistiche originali per sessione)", IntegerToString(totOriginalCount));
    FileWrite(fh, "Match Count", countOk ? "OK" : "WARNING: DISCREPANZA RILEVATA");

    FileClose(fh);

    if(!profitOk) Print("[STATS v5.8] WARNING: discrepanza tra profitto esteso e profitto originale per sessione: ", DoubleToString(diffProfit,2));
    if(!countOk)  Print("[STATS v5.8] WARNING: discrepanza nel numero di trade tra dataset esteso e statistiche originali.");
    Print("[STATS v5.8] CSV esportato: ", InpCSVFileName, " (", n, " trade registrati)");
}
//+------------------------------------------------------------------+