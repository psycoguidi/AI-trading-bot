//+------------------------------------------------------------------+
//|                         ORB_FTMO_Adaptive_V3.mq5                  |
//|   London / New York / Asia ORB with optional NY/Asia fade.        |
//|   This version changes entry windows, sizing and execution, and   |
//|   adds configurable 1-Step / 2-Step evaluation protection.        |
//|   Session start inputs use broker-server time. FTMO daily limits  |
//|   use a configurable broker-to-CE(S)T conversion.                 |
//|   The original CSV statistics are retained; evaluate the broker  |
//|   report for definitive deal counts and account P&L.             |
//+------------------------------------------------------------------+
#property copyright "ORB FTMO Adaptive V3"
#property version   "6.00"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS
// ═══════════════════════════════════════════
input group "London Session (ORB)"
input int      InpLon_StartHour     = 8;
input int      InpLon_StartMin      = 0;
input int      InpLon_DurationMin   = 60;
input bool     InpLon_UseDailyFilter= false;
input bool     InpLon_IgnoreBias    = false;

input group "New York Session (ORB)"
input int      InpNY_StartHour      = 15;
input int      InpNY_StartMin       = 30;
input int      InpNY_DurationMin    = 30;
input bool     InpNY_UseDailyFilter = false;

input group "Asian Session (ORB)"
input bool     InpUseAsia            = true;
input int      InpAsia_StartHour     = 1;
input int      InpAsia_StartMin      = 0;
input int      InpAsia_DurationMin   = 60;
input bool     InpAsia_UseDailyFilter= false;

input group "Sniper Filters (Crucial)"
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M5;
input int      InpMaxEntryDelayMin  = 180;
input double   InpBodyFilterPct     = 40.0;

input group "Fade dei Falsi Breakout (NON si applica a LONDON)"
input bool     InpFadeEnabled       = true;
input double   InpFadeRiskPct       = 50.0;
input double   InpFadeSLBufferPoints= 30;

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;
input double   InpATRMinRangeMult   = 0.22;

input group "Break-Even & Partial Close"
input bool     InpUseBreakEven      = true;
input double   InpBreakEvenTriggerR = 1.0;
input double   InpBreakEvenLockR    = 0.1;
input bool     InpUsePartialClose   = true;
input double   InpPartialClosePct   = 50.0;
input double   InpPartialCloseAtR   = 1.0;

input group "Risk & Money Management"
input double   InpFixedRiskAmount   = 60.0;
input bool     InpUseEquityRisk     = true;
input double   InpRiskPercent       = 0.6;
input double   InpRR_Target         = 2;
input int      InpMaxLossesDay      = 3;
input int      InpMaxTradesDay      = 5;
input int      InpMaxConcurrentPositions = 1; // 2+ only on a hedging account
input ulong    InpMagicNumber       = 20260911;
input bool     InpShowComments      = true;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = true;
input double   InpMaxDailyLossPct    = 3.0;

enum ENUM_PROP_PROGRAM { PROP_OFF=0, PROP_1_STEP=1, PROP_2_STEP_CHALLENGE=2, PROP_2_STEP_VERIFICATION=3 };
enum ENUM_SERVER_DST { SERVER_DST_EU=0, SERVER_DST_US=1, SERVER_DST_NONE=2 };
input group "FTMO evaluation protection (account-wide equity)"
input ENUM_PROP_PROGRAM InpPropProgram = PROP_1_STEP;
input double   InpInitialCapital     = 10000.0;
input double   InpMaxRiskPctInitial  = 0.60;     // hard cap per new trade
input double   InpFloorBufferPct     = 0.30;     // 30 currency units on 10k
input double   InpCostReservePerTrade= 5.0;      // extra room for fees/slippage
input int      InpServerUTCWinter    = 2;        // verify broker server clock
input int      InpServerUTCSummer    = 3;
input ENUM_SERVER_DST InpServerDST   = SERVER_DST_EU;
input datetime InpEvaluationStart    = 0;        // 0 = full account history
input double   InpConfirmedHighMidnightBalance = 0.0; // required live for 1-Step
input bool     InpLiveClockAndHistoryVerified = false; // enable only after checking broker time/history
input bool     InpCloseOwnAtBuffer   = true;
input bool     InpPauseAtTarget      = true;
input double   InpMinMarginLevelPct  = 250.0;

// [STATS v5.8] ── Nuovo gruppo di input, NON influisce sul trading ──
input group "Statistics & CSV Export (legacy v5.8 format)"
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
// sono mantenuti per BE/partial close e per le statistiche.
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
string   g_lastOpenedSession = "";
int      g_propDayKey = 0;
double   g_propMidnightBalance = 0.0;
double   g_propHighMidnightBalance = 0.0;
bool     g_propHistoryReady = false;
bool     g_propLocked = false;
bool     g_propHardBreach = false;
int      g_counterDayKey = 0;

// ═══════════════════════════════════════════
// 📒 TRADE LOG (formato legacy mantenuto per il report finale)
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
    if(InpMaxEntryDelayMin < 5 || InpBodyFilterPct < 0 || InpBodyFilterPct > 100 ||
       InpATRMinRangeMult < 0 || InpMaxConcurrentPositions < 1 || InpMaxConcurrentPositions > 3 ||
       InpInitialCapital <= 0 || InpMaxRiskPctInitial <= 0 ||
       InpFloorBufferPct < 0 || InpCostReservePerTrade < 0 || InpServerUTCWinter < -12 || InpServerUTCWinter > 14 ||
       InpServerUTCSummer < -12 || InpServerUTCSummer > 14) return INIT_PARAMETERS_INCORRECT;
    if(!MQLInfoInteger(MQL_TESTER) && InpPropProgram != PROP_OFF &&
       (!InpLiveClockAndHistoryVerified ||
        (InpPropProgram == PROP_1_STEP && InpConfirmedHighMidnightBalance < InpInitialCapital))) {
        Print("FTMO live setup incomplete: verify broker UTC/DST, full account history and 1-Step midnight high in MetriX.");
        return INIT_PARAMETERS_INCORRECT;
    }
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFillingBySymbol(_Symbol);

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
    g_counterDayKey = (InpPropProgram == PROP_OFF) ? dtInit.year*10000+dtInit.mon*100+dtInit.day : FTMODayKey(TimeCurrent());
    if(InpPropProgram != PROP_OFF && !MQLInfoInteger(MQL_TESTER))
        g_propHardBreach = GlobalVariableCheck(PropBreachKey()) && GlobalVariableGet(PropBreachKey()) > 0.5;

    RestoreDailyStats();
    if(InpPropProgram != PROP_OFF && !RebuildPropState())
        Print("FTMO history unavailable: new entries blocked until it can be rebuilt.");
    RestoreOpenPositionMap();
    EventSetTimer(5);
    Print("ORB FTMO Adaptive V3 ready. Test on a demo before use.");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    EventKillTimer();
    PrintFinalReport();
    PrintAdvancedStats();
    if(InpExportCSV) ExportStatsToCSV(); // [STATS v5.8]
    if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
    Comment("");
}

void OnTimer() { PropGuard(); }

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
    double requested = InpUseEquityRisk ? AccountInfoDouble(ACCOUNT_EQUITY) * (InpRiskPercent/100.0) : InpFixedRiskAmount;
    if(InpPropProgram != PROP_OFF)
        requested = MathMin(requested, InpInitialCapital * InpMaxRiskPctInitial / 100.0);
    return MathMax(0.0, requested);
}

double GetATRValue() {
    if(g_atrHandle == INVALID_HANDLE) return -1;
    double buf[];
    ArraySetAsSeries(buf, true);
    if(CopyBuffer(g_atrHandle, 0, 1, 1, buf) <= 0) return -1;
    return buf[0];
}

// The strategy tester does not provide the broker's historical UTC offset.
// The two explicit offsets below describe the broker clock; EU daylight saving
// switches at 01:00 UTC on the last Sunday of March/October.
int LastSunday(int year, int month) {
    MqlDateTime d = {};
    d.year = year; d.mon = month + 1; d.day = 1;
    if(month == 12) { d.year = year + 1; d.mon = 1; }
    datetime last = StructToTime(d) - 86400;
    TimeToStruct(last, d);
    return d.day - d.day_of_week;
}

datetime EuropeSwitchUTC(int year, int month) {
    MqlDateTime d = {};
    d.year = year; d.mon = month; d.day = LastSunday(year, month);
    d.hour = 1;
    return StructToTime(d);
}

datetime USSwitchUTC(int year, int month) {
    MqlDateTime d = {};
    d.year = year; d.mon = month; d.day = 1;
    datetime first = StructToTime(d);
    TimeToStruct(first, d);
    int firstSunday = 1 + ((7 - d.day_of_week) % 7);
    d.day = firstSunday + (month == 3 ? 7 : 0);
    d.hour = (month == 3) ? 7 : 6; // 02:00 New York local
    return StructToTime(d);
}

int ServerUTCOffset(datetime serverTime) {
    if(InpServerDST == SERVER_DST_NONE) return InpServerUTCWinter;
    MqlDateTime d; TimeToStruct(serverTime, d);
    datetime springUTC = (InpServerDST == SERVER_DST_US) ? USSwitchUTC(d.year, 3) : EuropeSwitchUTC(d.year, 3);
    datetime autumnUTC = (InpServerDST == SERVER_DST_US) ? USSwitchUTC(d.year, 11) : EuropeSwitchUTC(d.year, 10);
    datetime spring = springUTC + InpServerUTCWinter * 3600;
    datetime autumn = autumnUTC + InpServerUTCSummer * 3600;
    return (serverTime >= spring && serverTime < autumn) ? InpServerUTCSummer : InpServerUTCWinter;
}

datetime FTMOClock(datetime serverTime) {
    datetime utc = serverTime - ServerUTCOffset(serverTime) * 3600;
    MqlDateTime d; TimeToStruct(utc, d);
    int cetOffset = (utc >= EuropeSwitchUTC(d.year, 3) && utc < EuropeSwitchUTC(d.year, 10)) ? 2 : 1;
    return utc + cetOffset * 3600;
}

int FTMODayKey(datetime serverTime) {
    MqlDateTime d; TimeToStruct(FTMOClock(serverTime), d);
    return d.year * 10000 + d.mon * 100 + d.day;
}

double DealBalanceDelta(ulong ticket) {
    return HistoryDealGetDouble(ticket, DEAL_PROFIT)
         + HistoryDealGetDouble(ticket, DEAL_SWAP)
         + HistoryDealGetDouble(ticket, DEAL_COMMISSION)
         + HistoryDealGetDouble(ticket, DEAL_FEE);
}

// Rebuild the midnight balance and EOD high from account-wide history. This
// also repairs the guard after an EA restart during the evaluation.
bool RebuildPropState() {
    if(InpPropProgram == PROP_OFF) return true;
    datetime now = TimeCurrent();
    if(!HistorySelect(0, now)) { g_propHistoryReady = false; return false; }
    bool live = !MQLInfoInteger(MQL_TESTER);
    double currentBalance = AccountInfoDouble(ACCOUNT_BALANCE);
    if(live && HistoryDealsTotal() == 0 &&
       (MathAbs(currentBalance-InpInitialCapital) > 0.01 ||
        InpConfirmedHighMidnightBalance > InpInitialCapital+0.01)) {
        g_propHistoryReady = false;
        Print("FTMO history is empty for an account with prior balance activity; new entries blocked.");
        return false;
    }
    int keys[]; double deltas[];
    int todayKey = FTMODayKey(now);
    datetime earliest = (InpEvaluationStart > 0) ? InpEvaluationStart : 0;
    double evaluationDeltas = 0.0;
    for(int i=0; i<HistoryDealsTotal(); i++) {
        ulong ticket = HistoryDealGetTicket(i);
        datetime t = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
        if(t < earliest) continue;
        evaluationDeltas += DealBalanceDelta(ticket);
        int key = FTMODayKey(t);
        if(key > todayKey) continue;
        int j = -1;
        for(int k=0; k<ArraySize(keys); k++) if(keys[k] == key) { j=k; break; }
        if(j < 0) { j=ArraySize(keys); ArrayResize(keys,j+1); ArrayResize(deltas,j+1); keys[j]=key; deltas[j]=0.0; }
        deltas[j] += DealBalanceDelta(ticket);
    }
    // InpEvaluationStart must be after the initial funding deal. A mismatch
    // then exposes at least some missing history; it is not a completeness proof.
    if(live && InpEvaluationStart > 0 &&
       MathAbs((currentBalance-evaluationDeltas)-InpInitialCapital) > 0.02) {
        g_propHistoryReady = false;
        Print("FTMO evaluation history cannot reproduce the initial balance; new entries blocked.");
        return false;
    }
    for(int i=0; i<ArraySize(keys); i++)
        for(int j=i+1; j<ArraySize(keys); j++)
            if(keys[j] > keys[i]) {
                int ki=keys[i]; keys[i]=keys[j]; keys[j]=ki;
                double di=deltas[i]; deltas[i]=deltas[j]; deltas[j]=di;
            }
    double balance = currentBalance;
    double todayDelta = 0.0;
    double high = MathMax(InpInitialCapital, InpConfirmedHighMidnightBalance);
    for(int i=0; i<ArraySize(keys); i++) {
        if(keys[i] == todayKey) todayDelta = deltas[i];
        balance -= deltas[i];
        high = MathMax(high, balance);
    }
    g_propDayKey = todayKey;
    g_propMidnightBalance = currentBalance - todayDelta;
    // Keep the most conservative midnight balance already seen today. This
    // protects a restart from underestimating today's floor after missing deals.
    if(live && GlobalVariableCheck(PropSnapshotDayKey()) && GlobalVariableCheck(PropMidnightKey()) &&
       (int)GlobalVariableGet(PropSnapshotDayKey()) == todayKey)
        g_propMidnightBalance = MathMax(g_propMidnightBalance, GlobalVariableGet(PropMidnightKey()));
    g_propHighMidnightBalance = MathMax(high, g_propMidnightBalance);
    if(live && GlobalVariableCheck(PropHighKey()))
        g_propHighMidnightBalance = MathMax(g_propHighMidnightBalance, GlobalVariableGet(PropHighKey()));
    if(live) {
        GlobalVariableSet(PropHighKey(), g_propHighMidnightBalance);
        GlobalVariableSet(PropSnapshotDayKey(), (double)todayKey);
        GlobalVariableSet(PropMidnightKey(), g_propMidnightBalance);
    }
    g_propHistoryReady = true;
    g_propLocked = live && GlobalVariableCheck(PropLockDayKey()) &&
                   (int)GlobalVariableGet(PropLockDayKey()) == todayKey;
    PrintFormat("FTMO %d: day balance %.2f; max midnight balance %.2f", todayKey,
                g_propMidnightBalance, g_propHighMidnightBalance);
    return true;
}

double PropFloor() {
    if(InpPropProgram == PROP_OFF) return -DBL_MAX;
    double dailyPct = (InpPropProgram == PROP_1_STEP) ? 3.0 : 5.0;
    double dailyFloor = g_propMidnightBalance - InpInitialCapital * dailyPct / 100.0;
    double overallFloor = (InpPropProgram == PROP_1_STEP)
        ? g_propHighMidnightBalance - InpInitialCapital * 0.10
        : InpInitialCapital * 0.90;
    return MathMax(dailyFloor, overallFloor);
}

string PropBreachKey() {
    return StringFormat("ORBFTMO_%I64d_%d_%I64d", AccountInfoInteger(ACCOUNT_LOGIN),
                        (int)InpPropProgram, (long)InpEvaluationStart);
}

string PropHighKey() { return PropBreachKey() + "_HIGH"; }
string PropSnapshotDayKey() { return PropBreachKey() + "_DAY"; }
string PropMidnightKey() { return PropBreachKey() + "_MIDNIGHT"; }
string PropLockDayKey() { return PropBreachKey() + "_LOCKDAY"; }

void CloseOwnPositionsForProtection() {
    if(!InpCloseOwnAtBuffer) return;
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
        if(!trade.PositionClose(ticket))
            PrintFormat("FTMO close failed ticket %I64u retcode %u %s", ticket,
                        trade.ResultRetcode(), trade.ResultRetcodeDescription());
        else if(trade.ResultRetcode() != TRADE_RETCODE_DONE && trade.ResultRetcode() != TRADE_RETCODE_DONE_PARTIAL)
            PrintFormat("FTMO close not filled ticket %I64u retcode %u %s", ticket,
                        trade.ResultRetcode(), trade.ResultRetcodeDescription());
    }
}

bool PropGuard() {
    if(InpPropProgram == PROP_OFF) return true;
    if(!g_propHistoryReady || FTMODayKey(TimeCurrent()) != g_propDayKey)
        if(!RebuildPropState()) return false;
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double trigger = PropFloor() + InpInitialCapital * InpFloorBufferPct / 100.0;
    if(equity < PropFloor() && !g_propHardBreach) {
        g_propHardBreach = true;
        PrintFormat("FTMO HARD BREACH: equity %.2f below floor %.2f", equity, PropFloor());
        if(!MQLInfoInteger(MQL_TESTER)) GlobalVariableSet(PropBreachKey(), 1.0);
    }
    if(equity <= trigger) {
        if(!g_propLocked) PrintFormat("FTMO safety lock: equity %.2f, floor %.2f, trigger %.2f", equity, PropFloor(), trigger);
        g_propLocked = true;
        if(!MQLInfoInteger(MQL_TESTER)) GlobalVariableSet(PropLockDayKey(), (double)g_propDayKey);
        CloseOwnPositionsForProtection();
    }
    if(g_propHardBreach) CloseOwnPositionsForProtection();
    return !g_propLocked && !g_propHardBreach;
}

double PropRiskCapacity() {
    if(InpPropProgram == PROP_OFF) return DBL_MAX;
    double capacity = AccountInfoDouble(ACCOUNT_EQUITY) - PropFloor()
                    - InpInitialCapital * InpFloorBufferPct / 100.0;
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        double stop = PositionGetDouble(POSITION_SL);
        if(stop <= 0.0) return 0.0;
        string symbol = PositionGetString(POSITION_SYMBOL);
        double volume = PositionGetDouble(POSITION_VOLUME);
        double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        ENUM_ORDER_TYPE side = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
        double pnlAtStop = 0.0;
        if(!OrderCalcProfit(side, symbol, volume, openPrice, stop, pnlAtStop)) return 0.0;
        double currentProfit = PositionGetDouble(POSITION_PROFIT);
        capacity -= MathMax(0.0, currentProfit - pnlAtStop) + InpCostReservePerTrade;
    }
    return MathMax(0.0, capacity);
}

// [STATS v5.8] firma estesa con parametri opzionali (default = comportamento non cambiato
// per eventuali altri chiamanti). I campi legacy restano disponibili.
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

void RestoreOpenPositionMap() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0 || PositionGetString(POSITION_SYMBOL) != _Symbol ||
           PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
        long posId = PositionGetInteger(POSITION_IDENTIFIER);
        if(FindSessionMapIndex(posId) >= 0) continue;
        string label = PositionGetString(POSITION_COMMENT);
        int sep = StringFind(label, "|");
        if(sep >= 0) label = StringSubstr(label, 0, sep);
        if(label == "") label = "RECOVERED";
        double openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
        double stop = PositionGetDouble(POSITION_SL);
        double volume = PositionGetDouble(POSITION_VOLUME);
        int dir = PositionGetInteger(POSITION_TYPE) == POSITION_TYPE_BUY ? 1 : -1;
        RegisterPosition(posId, label, openPrice, MathAbs(openPrice-stop), dir,
                         PositionGetDouble(POSITION_TP), openPrice, openPrice, volume, 0, 0);
        int idx = FindSessionMapIndex(posId);
        if(idx >= 0) {
            g_posSessionMap[idx].entryTime = (datetime)PositionGetInteger(POSITION_TIME);
            g_posSessionMap[idx].beDone = true;
            g_posSessionMap[idx].partialDone = true;
        }
        PrintFormat("Recovered open position %I64d; existing SL/TP kept", posId);
    }
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

int CountOwnOpenPositions() {
    int count=0;
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket=PositionGetTicket(i);
        if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol &&
           PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) count++;
    }
    return count;
}

int AllowedConcurrentPositions() {
    if(AccountInfoInteger(ACCOUNT_MARGIN_MODE) != ACCOUNT_MARGIN_MODE_RETAIL_HEDGING) return 1;
    return InpMaxConcurrentPositions;
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
    datetime todayStart = (InpPropProgram == PROP_OFF) ?
        StringToTime(TimeToString(TimeCurrent(), TIME_DATE)) : TimeCurrent() - 2*86400;
    int counterKey = g_counterDayKey;
    g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
    ArrayResize(g_countedLossIds, 0);
    if(!HistorySelect(todayStart, TimeCurrent())) return;
    int totalDeals = HistoryDealsTotal();
    for(int i=0; i<totalDeals; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        datetime dealTime = (datetime)HistoryDealGetInteger(ticket, DEAL_TIME);
        if(InpPropProgram != PROP_OFF && FTMODayKey(dealTime) != counterKey) continue;
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
        if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) g_tradesToday++;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
            long posId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_SWAP) + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            bool stillOpen = false;
            for(int p=PositionsTotal()-1; p>=0; p--) {
                if(PositionGetTicket(p) > 0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
            }
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
    Print("ORB FTMO Adaptive V3 - FINAL REPORT");
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
// 📊 STATISTICHE AVANZATE (formato legacy)
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
    Print("STATISTICHE AVANZATE ORB FTMO V3 - trade totali registrati: ", n);
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
// 🎯 GESTIONE DINAMICA POSIZIONI
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
                if(trade.PositionClosePartial(ticket, closeVol) &&
                   (trade.ResultRetcode() == TRADE_RETCODE_DONE || trade.ResultRetcode() == TRADE_RETCODE_DONE_PARTIAL)) {
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
                if(trade.PositionModify(ticket, beSL, currentTP) && trade.ResultRetcode() == TRADE_RETCODE_DONE) {
                    g_posSessionMap[idx].beDone = true;
                    Print("BREAK-EVEN: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Nuovo SL: ", DoubleToString(beSL,digits));
                }
            }
        }
    }
}

// MT5 can deliver a weekend SL deal before the position list has removed the
// position. The transaction handler then defers finalization; reconcile it on
// the next minute so CSV/session statistics include those completed trades.
void ReconcileClosedPositions() {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) {
        long posId = g_posSessionMap[i].posId;
        bool open = false;
        for(int p=PositionsTotal()-1; p>=0; p--)
            if(PositionGetTicket(p) > 0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) { open = true; break; }
        if(open || !HistorySelectByPosition(posId)) continue;
        double totalPnL=0, rawProfit=0, commission=0, swap=0;
        datetime finalCloseTime=0;
        int exits=0;
        for(int d=0; d<HistoryDealsTotal(); d++) {
            ulong deal = HistoryDealGetTicket(d);
            if(HistoryDealGetInteger(deal, DEAL_POSITION_ID) != posId ||
               HistoryDealGetInteger(deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
            double p=HistoryDealGetDouble(deal, DEAL_PROFIT);
            double c=HistoryDealGetDouble(deal, DEAL_COMMISSION);
            double s=HistoryDealGetDouble(deal, DEAL_SWAP);
            totalPnL += p+c+s; rawProfit += p; commission += c; swap += s; exits++;
            finalCloseTime = MathMax(finalCloseTime, (datetime)HistoryDealGetInteger(deal, DEAL_TIME));
        }
        if(exits == 0) continue;
        SessionMap m = g_posSessionMap[i];
        if(!IsIdAlreadyCounted(posId)) {
            if(m.session == "LONDON") { g_profitLon+=totalPnL; if(totalPnL>0) g_winsLon++; }
            else if(m.session == "NY") { g_profitNY+=totalPnL; if(totalPnL>0) g_winsNY++; }
            else if(m.session == "ASIA") { g_profitAsia+=totalPnL; if(totalPnL>0) g_winsAsia++; }
            else if(m.session == "NY_FADE") { g_profitNYFade+=totalPnL; if(totalPnL>0) g_winsNYFade++; }
            else if(m.session == "ASIA_FADE") { g_profitAsiaFade+=totalPnL; if(totalPnL>0) g_winsAsiaFade++; }
            else if(m.session == "NY_REVERSAL") { g_profitNYRev+=totalPnL; if(totalPnL>0) g_winsNYRev++; }
            LogTrade(m.entryTime, finalCloseTime, totalPnL, m.dir, m.session);
            double initialSL = m.entryPrice - (m.dir==1 ? m.slDistance : -m.slDistance);
            LogTradeFull(posId, m.entryTime, finalCloseTime, m.sessionLabel, m.setupLabel,
                         m.dir, m.lots, m.requestedPrice, m.actualFillPrice, initialSL,
                         m.initialTP, m.riskMoneyAtOpen, m.rewardMoneyAtOpen,
                         rawProfit, commission, swap, m.maeR, m.mfeR);
            AddToLongArray(g_countedLossIds, posId);
            if(totalPnL < 0) { g_lossesToday++; if(g_lossesToday >= InpMaxLossesDay) g_dayLocked=true; }
            PrintFormat("Recovered completed trade %I64d in session %s", posId, m.session);
        }
        UnregisterPosition(posId);
    }
}

// ══════════════════════════════════════════
// 🔄 MAIN ORB LOGIC - wider, configurable entry window
// ═══════════════════════════════════════════
void OnTick() {
    PropGuard();
    ManageOpenPositions();

    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;
    ReconcileClosedPositions();

    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(today != g_currentDay) {
        g_currentDay = today;
        g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        g_lonRangeDefined = false; g_nyRangeDefined = false; g_asiaRangeDefined = false;
        g_lonExecutedToday = false; g_nyExecutedToday = false; g_asiaExecutedToday = false;
        g_lonHigh = 0.0; g_lonLow = 999999.0;
        g_nyHigh = 0.0; g_nyLow = 999999.0;
        g_asiaHigh = 0.0; g_asiaLow = 999999.0;
        g_nyReArmActive = false;
        g_lastOpenedSession = "";
    }

    int counterDay = (InpPropProgram == PROP_OFF) ? dt.year*10000+dt.mon*100+dt.day : FTMODayKey(TimeCurrent());
    if(counterDay != g_counterDayKey) {
        g_counterDayKey = counterDay;
        RestoreDailyStats();
    }

    if(InpPropProgram == PROP_OFF && InpUseEquityDailyLock && g_dayStartEquity > 0) {
        double eq = AccountInfoDouble(ACCOUNT_EQUITY);
        double lossPct = (g_dayStartEquity - eq) / g_dayStartEquity * 100.0;
        if(lossPct >= InpMaxDailyLossPct) g_dayLocked = true;
    }

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
    static datetime lastEntryBar = 0;
    datetime entryBar = iTime(_Symbol, InpEntryTF, 0);
    bool newEntryBar = (entryBar != lastEntryBar && entryBar > 0);
    if(newEntryBar) lastEntryBar = entryBar;

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

    // --- LONDON SESSION LOGIC ---
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
        } else if(TimeCurrent() < lonScanLimit && g_lonRangeDefined && newEntryBar) {

            bool atrOkLon = true;
            if(InpUseATRFilter) {
                double atrVal = GetATRValue();
                double sessionRange = g_lonHigh - g_lonLow;
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkLon = false;
            }

            if(!atrOkLon) {
                PrintFormat("ORB SKIP LONDON %d: range %.5f < ATR threshold %.5f", FTMODayKey(TimeCurrent()),
                            g_lonHigh-g_lonLow, GetATRValue()*InpATRMinRangeMult);
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
                    if(g_lastOpenedSession == "LONDON") g_lonExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= lonScanLimit) {
             PrintFormat("ORB EXPIRED LONDON %d: no accepted signal in %d minutes", FTMODayKey(TimeCurrent()), InpMaxEntryDelayMin);
             g_lonExecutedToday = true;
        }
    }

    // --- NEW YORK SESSION LOGIC ---
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
        } else if(TimeCurrent() < nyScanLimit && g_nyRangeDefined && newEntryBar) {

            bool atrOkNY = true;
            if(InpUseATRFilter) {
                double atrVal = GetATRValue();
                double sessionRange = g_nyHigh - g_nyLow;
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkNY = false;
            }

            if(!atrOkNY) {
                PrintFormat("ORB SKIP NY %d: range %.5f < ATR threshold %.5f", FTMODayKey(TimeCurrent()),
                            g_nyHigh-g_nyLow, GetATRValue()*InpATRMinRangeMult);
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
                    if(g_lastOpenedSession == "NY" || g_lastOpenedSession == "NY_FADE") g_nyExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= nyScanLimit) {
             PrintFormat("ORB EXPIRED NY %d: no accepted signal in %d minutes", FTMODayKey(TimeCurrent()), InpMaxEntryDelayMin);
             g_nyExecutedToday = true;
        }
    }

    // --- ASIAN SESSION LOGIC ---
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
            } else if(TimeCurrent() < asiaScanLimit && g_asiaRangeDefined && newEntryBar) {

                bool atrOkAsia = true;
                if(InpUseATRFilter) {
                    double atrVal = GetATRValue();
                    double sessionRange = g_asiaHigh - g_asiaLow;
                    if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkAsia = false;
                }

                if(!atrOkAsia) {
                    PrintFormat("ORB SKIP ASIA %d: range %.5f < ATR threshold %.5f", FTMODayKey(TimeCurrent()),
                                g_asiaHigh-g_asiaLow, GetATRValue()*InpATRMinRangeMult);
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
                        if(g_lastOpenedSession == "ASIA" || g_lastOpenedSession == "ASIA_FADE") g_asiaExecutedToday = true;
                    }
                }
            } else if(TimeCurrent() >= asiaScanLimit) {
                PrintFormat("ORB EXPIRED ASIA %d: no accepted signal in %d minutes", FTMODayKey(TimeCurrent()), InpMaxEntryDelayMin);
                g_asiaExecutedToday = true;
            }
        }
    }

    if(InpShowComments) {
        Comment(StringFormat("ORB FTMO V3 | Bias: %s | Lon H/L: %.2f/%.2f | NY H/L: %.2f/%.2f | Asia H/L: %.2f/%.2f | Entries: %d | Floor: %.2f",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")),
               g_lonHigh, g_lonLow, g_nyHigh, g_nyLow, g_asiaHigh, g_asiaLow, g_tradesToday, PropFloor()));
    }
}

// ══════════════════════════════════════════
// 💼 EXECUTION - market quote, actual fill and evaluation risk budget
// ══════════════════════════════════════════
void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
    if(g_dayLocked || g_tradesToday >= InpMaxTradesDay ||
       CountOwnOpenPositions() >= AllowedConcurrentPositions() || !PropGuard()) return;
    if(InpPropProgram != PROP_OFF && InpPauseAtTarget) {
        double targetPct = (InpPropProgram == PROP_2_STEP_VERIFICATION) ? 5.0 : 10.0;
        if(AccountInfoDouble(ACCOUNT_BALANCE) >= InpInitialCapital * (1.0 + targetPct/100.0)) return;
    }
    MqlTick tick;
    if(!SymbolInfoTick(_Symbol, tick)) return;
    entry = (dir == 1) ? tick.ask : tick.bid;
    if(entry <= 0.0) return;
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    double sl_final = NormalizeDouble(sl_ref, digits);
    if((dir == 1 && sl_final >= entry) || (dir == -1 && sl_final <= entry)) return;

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

    double stopMin = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
    if(MathAbs(entry-sl_final) < stopMin || MathAbs(tp_final-entry) < stopMin) return;

    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    if(stepLot <= 0.0 || minLot <= 0.0 || maxLot < minLot) return;
    double profitPerLot = 0;

    bool calcRes = false;
    if(dir == 1) calcRes = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, sl_final, profitPerLot);
    else calcRes = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, sl_final, profitPerLot);

    if(!calcRes || profitPerLot >= 0) return;

    riskAmount = MathMin(riskAmount, PropRiskCapacity() - (InpPropProgram == PROP_OFF ? 0.0 : InpCostReservePerTrade));
    if(riskAmount <= 0.0) return;
    double lots = MathFloor((riskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
    lots = NormalizeDouble(MathMin(lots, maxLot), 8);
    if(lots < minLot) return; // never exceed the planned risk by forcing minimum volume

    double margin = 0.0;
    ENUM_ORDER_TYPE side = (dir == 1) ? ORDER_TYPE_BUY : ORDER_TYPE_SELL;
    if(!OrderCalcMargin(side, _Symbol, lots, entry, margin)) return;
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double usedMargin = AccountInfoDouble(ACCOUNT_MARGIN);
    double allowedMargin = AccountInfoDouble(ACCOUNT_MARGIN_FREE) / 1.05;
    if(InpMinMarginLevelPct > 0.0)
        allowedMargin = MathMin(allowedMargin, equity * 100.0 / InpMinMarginLevelPct - usedMargin);
    if(margin > allowedMargin && margin > 0.0) {
        lots = NormalizeDouble(MathFloor((lots * allowedMargin / margin) / stepLot) * stepLot, 8);
        if(lots < minLot || !OrderCalcMargin(side, _Symbol, lots, entry, margin) || margin > allowedMargin) return;
    }

    // [STATS v5.8] reward teorico al prezzo di TP, con lo stesso metodo (OrderCalcProfit)
    // gia' usato dal codice originale per il rischio. Puramente informativo, non altera
    // ne' lots ne' tp_final ne' alcuna decisione successiva.
    double rewardPerLot = 0;
    bool rewardCalcRes = false;
    if(dir == 1) rewardCalcRes = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, tp_final, rewardPerLot);
    else rewardCalcRes = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, tp_final, rewardPerLot);
    double riskMoneyAtOpen = MathAbs(profitPerLot) * lots;
    double rewardMoneyAtOpen = rewardCalcRes ? MathAbs(rewardPerLot) * lots : 0;

    string cmt = StringFormat("%s|FTMOv3|R:%.2f", session, riskAmount);
    bool res = false;

    if(dir == 1) res = trade.Buy(lots, _Symbol, 0.0, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, _Symbol, 0.0, sl_final, tp_final, cmt);

    bool filled = res && (trade.ResultRetcode() == TRADE_RETCODE_DONE ||
                          trade.ResultRetcode() == TRADE_RETCODE_DONE_PARTIAL) && trade.ResultDeal() > 0;
    if(filled) {
        g_tradesToday++;
        g_lastOpenedSession = session;

        long posId = 0;
        double actualFillPrice = trade.ResultPrice();
        if(HistoryDealSelect(trade.ResultDeal()))
            posId = (long)HistoryDealGetInteger(trade.ResultDeal(), DEAL_POSITION_ID);
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong t = PositionGetTicket(i);
            if(t == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
            if(posId > 0 && (long)PositionGetInteger(POSITION_IDENTIFIER) != posId) continue;
            posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
            actualFillPrice = PositionGetDouble(POSITION_PRICE_OPEN);
            lots = PositionGetDouble(POSITION_VOLUME);
            break;
        }
        if(posId > 0) RegisterPosition(posId, session, actualFillPrice, MathAbs(actualFillPrice - sl_final), dir,
                                        tp_final, entry, actualFillPrice, lots, riskMoneyAtOpen, rewardMoneyAtOpen);

        if(session == "LONDON") g_totalLon++;
        else if(session == "NY") g_totalNY++;
        else if(session == "ASIA") g_totalAsia++;
        else if(session == "NY_FADE") g_totalNYFade++;
        else if(session == "ASIA_FADE") g_totalAsiaFade++;
        else if(session == "NY_REVERSAL") g_totalNYRev++;

        Print("TRADE OPENED: ", session, " | PosID: ", posId, " | Risk: ", DoubleToString(riskAmount,2), " | TP: ", DoubleToString(tp_final, digits));
    } else {
        PrintFormat("TRADE REJECTED %s retcode %u %s", session, trade.ResultRetcode(), trade.ResultRetcodeDescription());
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

    if(dealEntry == DEAL_ENTRY_IN && FindSessionMapIndex(posId) < 0) {
        string sessionIn = HistoryDealGetString(trans.deal, DEAL_COMMENT);
        int separator = StringFind(sessionIn, "|");
        if(separator >= 0) sessionIn = StringSubstr(sessionIn, 0, separator);
        if(sessionIn == "") sessionIn = "UNKNOWN";
        double openPrice = HistoryDealGetDouble(trans.deal, DEAL_PRICE);
        double stopPrice = 0.0, takePrice = 0.0;
        double volume = HistoryDealGetDouble(trans.deal, DEAL_VOLUME);
        for(int p=PositionsTotal()-1; p>=0; p--) {
            if(PositionGetTicket(p) > 0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) {
                openPrice = PositionGetDouble(POSITION_PRICE_OPEN);
                stopPrice = PositionGetDouble(POSITION_SL);
                takePrice = PositionGetDouble(POSITION_TP);
                volume = PositionGetDouble(POSITION_VOLUME);
                break;
            }
        }
        int dirIn = HistoryDealGetInteger(trans.deal, DEAL_TYPE) == DEAL_TYPE_BUY ? 1 : -1;
        RegisterPosition(posId, sessionIn, openPrice, MathAbs(openPrice-stopPrice), dirIn,
                         takePrice, openPrice, openPrice, volume, 0, 0);
        return;
    }

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
        for(int p=PositionsTotal()-1; p>=0; p--) {
            if(PositionGetTicket(p) > 0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
        }
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
