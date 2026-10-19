// prova173: Come prova166 con cambio orario server USA, per confronto della contabilita CE(S)T
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova166: Long e vendite sotto EMA20 D1; overnight solo con stop in profitto, buffer 150
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova159: Long e short sotto EMA20 D1 come prova155, completamento 4 giorni con rischio massimo 1
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova155: Come prova151, vendite permesse soltanto sotto EMA20 D1 chiusa
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova151: Solo long prova146, chiusura venerdi e controllo rollover, buffer 100
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova146: Solo long, range e rientri come prova138, ATR ridotto, rischio 2
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova138: Prova134 con controllo giornaliero esportato e conversione oraria efficiente
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova134: Range esatti, breakout con 2 rientri per sessione, massimo 3 posizioni
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova132: Piu segnali: ATR Asia 0.15, NY/Londra 0.25, rischio 1.8
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova131: Prova130 con limiti FTMO, rischio originale, target 10 e arresto al superamento
// Conti mensili indipendenti 10000 EUR; target 10%; perdite FTMO 500/1000 EUR.
// prova130: Sessioni come prova117, rischio base 3.05 e incremento 1
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova117: Rischio differenziato: Asia 2.835, NY 2.16, London 2.97
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova104: Senza parziale, rischio 2.7 e incremento 0.75, stop risk aperto massimo 4
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova89: Senza parziale, rischio base 1.8 e incremento protetto 0.5
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova85: Prova73 con magic dedicato, statistiche incrementi e CSV per posizione
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova73: Rischio base 2.7 percento e incremento protetto 1 percento, stop risk aperto 2.5 percento
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova55: Asia 0.25 ATR, rischio base 2 percento e incremento protetto 0.75 percento
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova36: Filtro Asia 0.25 ATR e finestra 360 minuti
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
// prova25: Filtro ATR Asia 0.25, NY 0.35, Londra attiva, rischio 1.7 percento
// Test XAUUSD M5; 10000 EUR; leva 1:33; 2025.01.01-2026.09.27.
//+------------------------------------------------------------------+
//|                              Tri_Session_Master_v5.5_Fade.mq5    |
//|                   Logic: London/NY/Asia ORB - Sniper Filters     |
//|                   + FADE dei falsi breakout + gestione dinamica  |
//|                                                                    |
//|   v5.5 NOVITA':                                                   |
//|    - LONDON_FADE RIMOSSA: se il breakout su Londra ha corpo       |
//|      debole, non si apre nessun trade fade. La continuazione     |
//|      resta invariata. NY e Asia mantengono la fade come prima.   |
//+------------------------------------------------------------------+
#property copyright "Tri-Session Master v5.5 Fade"
#property version   "5.50"
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
input bool     InpLon_IgnoreBias    = true;

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
input int      InpMaxEntryDelayMin  = 360;
input double   InpBodyFilterPct     = 25.0;

input group "Fade dei Falsi Breakout (NON si applica a LONDON)"
input bool     InpFadeEnabled       = false;
input double   InpFadeRiskPct       = 50.0;
input double   InpFadeSLBufferPoints= 30;

input double InpATRMinRangeMultAsia = 0.1;
input double InpATRMinRangeMultNY = 0.15;

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;
input double   InpATRMinRangeMult   = 0.15;

input group "Break-Even & Partial Close"
input bool     InpUseBreakEven      = true;
input double   InpBreakEvenTriggerR = 1.0;
input double   InpBreakEvenLockR    = 0.1;
input bool     InpUsePartialClose   = false;
input double   InpPartialClosePct   = 50.0;
input double   InpPartialCloseAtR   = 1.0;

input group "Risk & Money Management"
input double   InpFixedRiskAmount   = 80.0;
input bool     InpUseEquityRisk     = true;
input double   InpRiskPercent       = 2.0;
input double   InpRR_Target         = 2;
input int      InpMaxLossesDay      = 4;
input int      InpMaxTradesDay      = 12;
input ulong    InpMagicNumber       = 2026101073;
input bool     InpShowComments      = true;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = false;
input double   InpMaxDailyLossPct    = 3.0;

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ══════════════════════════════════════════
input double InpPyramidAtR=1.2;
input double InpPyramidRiskPercent=0.4;
input double InpPyramidExtraTargetR=1.0;
input double InpTotalOpenStopRiskPct=4.0;
double g_profitPyramid=0; int g_winsPyramid=0,g_totalPyramid=0;
input double InpAsiaRiskMultiplier=1.05;
input double InpNYRiskMultiplier=0.8;
input double InpLondonRiskMultiplier=1.1;
// FTMO 2-Step: test mensile con target configurabile; nessun reset di saldo in reale.
enum ENUM_FTMO_SERVER_DST { FTMO_DST_EU=0, FTMO_DST_US=1, FTMO_DST_NONE=2 };
input group "FTMO 2-Step - protezione del conto"
input double InpFTMOInitialCapital=10000.0;
input double InpFTMOTargetPercent=10.0;
input int InpFTMOMinTradingDays=4;
input double InpFTMODailyLossEUR=500.0;
input double InpFTMOTotalLossEUR=1000.0;
input double InpFTMOBufferEUR=150.0;
input double InpFTMOReservePerTradeEUR=5.0;
input double InpFTMOQualificationRiskPct=1.0;
input int InpFTMOServerUTCWinter=2;
input int InpFTMOServerUTCSummer=3;
input ENUM_FTMO_SERVER_DST InpFTMOServerDST=FTMO_DST_US;
input bool InpFTMOBrokerClockConfirmed=false;
input group "Solo tester: periodo della prova indipendente"
input int InpResearchMonth=0;
input long InpResearchStopServer=0;
bool g_ftmoReady=false,g_ftmoFailed=false,g_ftmoPassed=false,g_ftmoTotalStopped=false;
int g_ftmoDay=0,g_ftmoLockedDay=0,g_ftmoDays[],g_ftmoMonth=0;
double g_ftmoMidnight=10000,g_ftmoMinEquity=10000,g_ftmoMinBalance=10000,g_ftmoWorstDaily=0;
datetime g_ftmoStartCET=0,g_ftmoEndCET=0,g_ftmoPassedServer=0,g_ftmoFirstEntry=0;
string g_ftmoFailureReason="";

input int InpMaximumPositions=3;
input int InpSessionMaxEntries=2;
input int InpReentryCooldownMin=20;
int g_lonEntries=0,g_nyEntries=0,g_asiaEntries=0;
datetime g_lonLastEntry=0,g_nyLastEntry=0,g_asiaLastEntry=0;
int g_auditKeys[],g_auditIndex=-1;double g_auditMidnight[],g_auditMinimum[];
input bool InpAllowLong=true;
input bool InpAllowShort=true;
input group "Rischio di fine giornata e fine settimana"
input bool InpCloseFriday=true;
input int InpRolloverCheckServerHour=22;
input int InpRolloverCheckServerMinute=45;
input group "Direzione giornaliera - dati chiusi"
input int InpDailyDirectionPeriod=20;
input bool InpDailyFilterLong=false;
input bool InpDailyFilterShort=true;
int g_dailyDirectionMA=INVALID_HANDLE;
input bool InpOvernightOnlyProtected=true;
input bool InpDailyFlat=false;
CTrade trade;
bool g_lonLoaded=false,g_nyLoaded=false,g_asiaLoaded=false;
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

struct SessionMap { long posId; string session; double entryPrice; double slDistance; int dir; bool beDone; bool partialDone; bool pyramidDone; };
SessionMap g_posSessionMap[];

// Stats continuazione
double g_profitLon = 0, g_profitNY = 0, g_profitAsia = 0;
int    g_winsLon = 0, g_winsNY = 0, g_winsAsia = 0;
int    g_totalLon = 0, g_totalNY = 0, g_totalAsia = 0;

// Stats fade (NY/Asia soltanto)
double g_profitNYFade = 0, g_profitAsiaFade = 0;
int    g_winsNYFade = 0, g_winsAsiaFade = 0;
int    g_totalNYFade = 0, g_totalAsiaFade = 0;

// Stats reversal - doppio fakeout (NY soltanto, Londra non ha piu' fade quindi non arma mai il reversal)
double g_profitNYRev = 0;
int    g_winsNYRev = 0;
int    g_totalNYRev = 0;

// Stato per il rientro reversal dopo stop della fade (NY)
int      g_nyFadeDir = 0;    double g_nyFadeSL = 0;
bool     g_nyReArmActive = false;  datetime g_nyReArmDeadline = 0;

// ═══════════════════════════════════════════
// 🚀 INIT & DEINIT
// ══════════════════════════════════════════
int OnInit() {
 if(InpDailyDirectionPeriod<=0)return INIT_PARAMETERS_INCORRECT;
 g_dailyDirectionMA=iMA(_Symbol,PERIOD_D1,InpDailyDirectionPeriod,0,MODE_EMA,PRICE_CLOSE);
 if(g_dailyDirectionMA==INVALID_HANDLE)return INIT_FAILED;

    if(InpFTMOInitialCapital<=0 || InpFTMOTargetPercent<=0 || InpFTMOMinTradingDays<4 || InpFTMOBufferEUR<0 || InpFTMODailyLossEUR!=500 || InpFTMOTotalLossEUR!=1000)return INIT_PARAMETERS_INCORRECT;
    if(!MQLInfoInteger(MQL_TESTER) && !InpFTMOBrokerClockConfirmed){Print("Verificare orario/DST del broker FTMO e impostare InpFTMOBrokerClockConfirmed. Nessun ordine inviato.");return INIT_PARAMETERS_INCORRECT;}

    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    ArrayResize(g_posSessionMap, 0);

    g_atrHandle = iATR(_Symbol, PERIOD_D1, InpATRPeriod);
    if(g_atrHandle == INVALID_HANDLE) Print("ATTENZIONE: impossibile creare handle ATR, filtro volatilità disattivato di fatto.");

    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

    RestoreDailyStats();
    Print("Tri-Session Master v5.5 (Londra senza fade) Ready.");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
 if(g_dailyDirectionMA!=INVALID_HANDLE)IndicatorRelease(g_dailyDirectionMA);
    PrintFinalReport();
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

void RegisterPosition(long posId, string session, double entryPrice, double slDistance, int dir) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) if(g_posSessionMap[i].posId == posId) ArrayRemove(g_posSessionMap, i, 1);
    int s = ArraySize(g_posSessionMap); ArrayResize(g_posSessionMap, s+1);
    g_posSessionMap[s].posId = posId;
    g_posSessionMap[s].session = session;
    g_posSessionMap[s].entryPrice = entryPrice;
    g_posSessionMap[s].slDistance = slDistance;
    g_posSessionMap[s].dir = dir;
    g_posSessionMap[s].beDone = false;
    g_posSessionMap[s].partialDone = false;
    g_posSessionMap[s].pyramidDone = false;
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
g_lonEntries=0;g_nyEntries=0;g_asiaEntries=0;g_lonLastEntry=0;g_nyLastEntry=0;g_asiaLastEntry=0;
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
            for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetTicket(p)>0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
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
    Print("TRI-SESSION MASTER v5.5 (LONDON NO-FADE) - FINAL REPORT");
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
    PrintSessionStats("PYRAMID",g_profitPyramid,g_winsPyramid,g_totalPyramid);
    Print("--- Totale ---");
    double totProfit = g_profitLon + g_profitNY + g_profitAsia + g_profitNYFade + g_profitAsiaFade + g_profitNYRev + g_profitPyramid;
    int totWins = g_winsLon + g_winsNY + g_winsAsia + g_winsNYFade + g_winsAsiaFade + g_winsNYRev + g_winsPyramid;
    int totTrades = g_totalLon + g_totalNY + g_totalAsia + g_totalNYFade + g_totalAsiaFade + g_totalNYRev + g_totalPyramid;
    PrintSessionStats("TOTALE      ", totProfit, totWins, totTrades);
    Print("--------------------------------------------------");
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

        double volume    = PositionGetDouble(POSITION_VOLUME);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);

        // --- CHIUSURA PARZIALE ---
        if(InpUsePartialClose && !g_posSessionMap[idx].partialDone && rMultiple >= InpPartialCloseAtR) {
            double volMin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            double closeVol = MathFloor((volume * (InpPartialClosePct/100.0)) / volStep) * volStep;
            if(closeVol >= volMin && closeVol < volume) {
                if(trade.PositionClosePartial(ticket, closeVol) && (trade.ResultRetcode()==TRADE_RETCODE_DONE || trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL)) {
                    g_posSessionMap[idx].partialDone = true;
                    Print("PARTIAL CLOSE: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Vol: ", DoubleToString(closeVol,2), " @ R=", DoubleToString(rMultiple,2));
                }
            } else {
                g_posSessionMap[idx].partialDone = true;
            }
        }

        // --- BREAK-EVEN ---
        if(InpUseBreakEven && !g_posSessionMap[idx].beDone && rMultiple >= InpBreakEvenTriggerR) {
            double beSL = (dir == 1) ? entry + (InpBreakEvenLockR * slDist) : entry - (InpBreakEvenLockR * slDist);
            beSL = NormalizeDouble(beSL, digits);
            bool improves = (dir == 1) ? (beSL > currentSL) : (currentSL == 0 || beSL < currentSL);
            if(improves) {
                if(trade.PositionModify(ticket, beSL, currentTP) && trade.ResultRetcode()==TRADE_RETCODE_DONE) {
                    g_posSessionMap[idx].beDone = true;
                    Print("BREAK-EVEN: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Nuovo SL: ", DoubleToString(beSL,digits));
                }
            }
        }
    }
}

// ══════════════════════════════════════════
// 🔄 MAIN LOGIC
// ═══════════════════════════════════════════
// Un solo incremento su una posizione originale gia protetta; nessuna media in perdita.
void ManagePyramids() {
    static datetime checkedBar=0;
    datetime bar=iTime(_Symbol,PERIOD_M1,0);
    if(bar==checkedBar) return;
    checkedBar=bar;
    if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING || g_dayLocked || g_tradesToday>=InpMaxTradesDay) return;
    MqlTick tick; if(!SymbolInfoTick(_Symbol,tick)) return;
    int ownCount=0; double existingStopRisk=0;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        ownCount++;
        int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
        double stop=PositionGetDouble(POSITION_SL),price=dir==1?tick.bid:tick.ask,pnl=0;
        if(stop<=0) return;
        if(!OrderCalcProfit(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,PositionGetDouble(POSITION_VOLUME),price,stop,pnl)) return;
        existingStopRisk+=MathMax(0,-pnl);
    }
    if(ownCount!=1) return;
    double equity=AccountInfoDouble(ACCOUNT_EQUITY);
    double budget=MathMin(equity*InpPyramidRiskPercent/100.0,equity*InpTotalOpenStopRiskPct/100.0-existingStopRisk);
    if(budget<=0) return;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        long id=PositionGetInteger(POSITION_IDENTIFIER);
        int idx=FindSessionMapIndex(id);
        if(idx<0 || g_posSessionMap[idx].pyramidDone || !g_posSessionMap[idx].beDone || StringFind(g_posSessionMap[idx].session,"FADE")>=0 || g_posSessionMap[idx].session=="PYRAMID") continue;
        int dir=g_posSessionMap[idx].dir;
        double entry=g_posSessionMap[idx].entryPrice,distance=g_posSessionMap[idx].slDistance;
        double price=dir==1?tick.bid:tick.ask;
        double achieved=dir==1?price-entry:entry-price;
        double stop=PositionGetDouble(POSITION_SL),target=PositionGetDouble(POSITION_TP);
        if(distance<=0 || achieved<InpPyramidAtR*distance || (dir==1 && stop<=entry) || (dir==-1 && stop>=entry)) continue;
        target+=dir*distance*InpPyramidExtraTargetR;
        int previous=g_tradesToday;
        ExecuteTrade(dir,dir==1?tick.ask:tick.bid,stop,"PYRAMID",budget,target);
        idx=FindSessionMapIndex(id);
        if(idx>=0 && g_tradesToday>previous) g_posSessionMap[idx].pyramidDone=true;
        return;
    }
}

int FTMOLastSunday(int year,int month) {
   MqlDateTime d={};d.year=year;d.mon=month+1;d.day=1;
   if(month==12){d.year++;d.mon=1;}
   datetime t=StructToTime(d)-86400;TimeToStruct(t,d);return d.day-d.day_of_week;
}
datetime FTMOEUChange(int year,int month) {
   MqlDateTime d={};d.year=year;d.mon=month;d.day=FTMOLastSunday(year,month);d.hour=1;return StructToTime(d);
}
datetime FTMOUSChange(int year,int month) {
   MqlDateTime d={};d.year=year;d.mon=month;d.day=1;TimeToStruct(StructToTime(d),d);
   d.day=1+(7-d.day_of_week)%7+(month==3?7:0);d.hour=month==3?7:6;return StructToTime(d);
}
int FTMOServerOffset(datetime server) {
   if(InpFTMOServerDST==FTMO_DST_NONE)return InpFTMOServerUTCWinter;
   MqlDateTime d;TimeToStruct(server,d);
   datetime a=InpFTMOServerDST==FTMO_DST_US?FTMOUSChange(d.year,3):FTMOEUChange(d.year,3);
   datetime b=InpFTMOServerDST==FTMO_DST_US?FTMOUSChange(d.year,11):FTMOEUChange(d.year,10);
   a+=InpFTMOServerUTCWinter*3600;b+=InpFTMOServerUTCSummer*3600;
   return server>=a && server<b?InpFTMOServerUTCSummer:InpFTMOServerUTCWinter;
}
datetime FTMOClock(datetime server) {
   static datetime hourStart=0;static int correction=0;
   datetime currentHour=(datetime)(((long)server/3600)*3600);
   if(currentHour!=hourStart){
      hourStart=currentHour;datetime utc=server-FTMOServerOffset(server)*3600;MqlDateTime d;TimeToStruct(utc,d);
      int offset=utc>=FTMOEUChange(d.year,3)&&utc<FTMOEUChange(d.year,10)?2:1;correction=offset*3600-FTMOServerOffset(server)*3600;
   }
   return server+correction;
}
int FTMODayKey(datetime server){
   static datetime dayStart=0;static int key=0;datetime local=FTMOClock(server);
   if(local<dayStart || local>=dayStart+86400){MqlDateTime d;TimeToStruct(local,d);key=d.year*10000+d.mon*100+d.day;d.hour=0;d.min=0;d.sec=0;dayStart=StructToTime(d);}
   return key;
}
datetime FTMODayStart(datetime cet){MqlDateTime d;TimeToStruct(cet,d);d.hour=0;d.min=0;d.sec=0;return StructToTime(d);}
string FTMOStateKey(string suffix){return StringFormat("F10_%I64d_%I64u_%s",AccountInfoInteger(ACCOUNT_LOGIN),InpMagicNumber,suffix);}
void FTMOStoreFailure(){if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("FAILED"),1);}
double FTMODealDelta(ulong ticket){return HistoryDealGetDouble(ticket,DEAL_PROFIT)+HistoryDealGetDouble(ticket,DEAL_SWAP)+HistoryDealGetDouble(ticket,DEAL_COMMISSION)+HistoryDealGetDouble(ticket,DEAL_FEE);}
bool FTMOInEvaluation(datetime server) {
   datetime cet=FTMOClock(server);
   if(cet<g_ftmoStartCET || (g_ftmoEndCET>0 && cet>=g_ftmoEndCET))return false;
   if(MQLInfoInteger(MQL_TESTER) && InpResearchStopServer>0 && (long)server>=InpResearchStopServer)return false;
   return true;
}
void FTMOAddTradingDay(int key,datetime entered) {
   if(g_ftmoFirstEntry==0 || entered<g_ftmoFirstEntry)g_ftmoFirstEntry=entered;
   for(int i=0;i<ArraySize(g_ftmoDays);i++)if(g_ftmoDays[i]==key)return;
   int n=ArraySize(g_ftmoDays);ArrayResize(g_ftmoDays,n+1);g_ftmoDays[n]=key;
}
bool FTMORebuildDay() {
   datetime now=TimeCurrent();if(!HistorySelect(0,now))return false;
   int today=FTMODayKey(now);double delta=0,allDelta=0;
   for(int i=0;i<HistoryDealsTotal();i++) {
      ulong deal=HistoryDealGetTicket(i);datetime t=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);
      if(!FTMOInEvaluation(t))continue;
      long type=HistoryDealGetInteger(deal,DEAL_TYPE);
      if(type==DEAL_TYPE_BALANCE || type==DEAL_TYPE_CREDIT)continue;
      double pnl=FTMODealDelta(deal);allDelta+=pnl;
      if(FTMODayKey(t)==today)delta+=pnl;
      long entry=HistoryDealGetInteger(deal,DEAL_ENTRY);
      if((type==DEAL_TYPE_BUY || type==DEAL_TYPE_SELL) && (entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT))FTMOAddTradingDay(FTMODayKey(t),t);
   }
   double balance=AccountInfoDouble(ACCOUNT_BALANCE);
   if(!MQLInfoInteger(MQL_TESTER) && MathAbs(balance-allDelta-InpFTMOInitialCapital)>.03){Print("FTMO: storico incompleto o capitale iniziale diverso; ingressi bloccati.");return false;}
   g_ftmoMidnight=balance-delta;g_ftmoDay=today;
   if(!MQLInfoInteger(MQL_TESTER)) {
      if(GlobalVariableCheck(FTMOStateKey("DAY")) && (int)GlobalVariableGet(FTMOStateKey("DAY"))==today && GlobalVariableCheck(FTMOStateKey("MIDNIGHT")))g_ftmoMidnight=MathMax(g_ftmoMidnight,GlobalVariableGet(FTMOStateKey("MIDNIGHT")));
      GlobalVariableSet(FTMOStateKey("DAY"),today);GlobalVariableSet(FTMOStateKey("MIDNIGHT"),g_ftmoMidnight);
      if(GlobalVariableCheck(FTMOStateKey("LOCKDAY")))g_ftmoLockedDay=(int)GlobalVariableGet(FTMOStateKey("LOCKDAY"));
   }
   return true;
}
bool FTMOEnsureReady() {
   if(g_ftmoReady)return true;
   datetime now=TimeCurrent();MqlDateTime d;TimeToStruct(FTMOClock(now),d);
   g_ftmoMonth=MQLInfoInteger(MQL_TESTER)&&InpResearchMonth>0?InpResearchMonth:d.year*100+d.mon;
   MqlDateTime a={};a.year=g_ftmoMonth/100;a.mon=g_ftmoMonth%100;a.day=1;g_ftmoStartCET=StructToTime(a);
   if(MQLInfoInteger(MQL_TESTER)&&InpResearchMonth>0){a.mon++;if(a.mon>12){a.year++;a.mon=1;}g_ftmoEndCET=StructToTime(a);}
   else g_ftmoStartCET=0; // In reale ricostruisce la fase dall'intero storico del conto dedicato.
   if(!FTMORebuildDay())return false;
   if(!MQLInfoInteger(MQL_TESTER)) {
      g_ftmoFailed=GlobalVariableCheck(FTMOStateKey("FAILED"));
      if(GlobalVariableCheck(FTMOStateKey("PASSED"))){g_ftmoPassed=true;g_ftmoPassedServer=(datetime)GlobalVariableGet(FTMOStateKey("PASSED"));}
      g_ftmoTotalStopped=GlobalVariableCheck(FTMOStateKey("TOTAL_STOP"));
   }
   g_ftmoMinEquity=AccountInfoDouble(ACCOUNT_EQUITY);g_ftmoMinBalance=AccountInfoDouble(ACCOUNT_BALANCE);g_ftmoReady=true;return true;
}
double FTMOTotalFloor(){return InpFTMOInitialCapital-InpFTMOTotalLossEUR;}
double FTMODailyFloor(){return g_ftmoMidnight-InpFTMODailyLossEUR;}
double FTMOFloor(){return MathMax(FTMOTotalFloor(),FTMODailyFloor());}
double FTMOTarget(){return InpFTMOInitialCapital*(1+InpFTMOTargetPercent/100.0);}
void FTMOCloseOwn() {
   for(int p=PositionsTotal()-1;p>=0;p--) {
      ulong t=PositionGetTicket(p);if(t==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
      bool sent=trade.PositionClose(t);
      if(!sent || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))PrintFormat("FTMO: chiusura non completata %I64u, retcode %u",t,trade.ResultRetcode());
   }
}
void FTMOAuditObserve() {
   if(g_auditIndex<0 || g_auditKeys[g_auditIndex]!=g_ftmoDay){
      g_auditIndex=ArraySize(g_auditKeys);ArrayResize(g_auditKeys,g_auditIndex+1);ArrayResize(g_auditMidnight,g_auditIndex+1);ArrayResize(g_auditMinimum,g_auditIndex+1);
      g_auditKeys[g_auditIndex]=g_ftmoDay;g_auditMidnight[g_auditIndex]=g_ftmoMidnight;g_auditMinimum[g_auditIndex]=AccountInfoDouble(ACCOUNT_EQUITY);
   }
   g_auditMinimum[g_auditIndex]=MathMin(g_auditMinimum[g_auditIndex],AccountInfoDouble(ACCOUNT_EQUITY));
}
void FTMOExportDaily() {
   string name=MQLInfoString(MQL_PROGRAM_NAME)+"_FTMO_"+IntegerToString(g_ftmoMonth)+"_daily.csv";
   int h=FileOpen(name,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE)return;
   FileWrite(h,"day","midnight_balance","minimum_equity","daily_loss","total_floor","daily_floor");
   for(int i=0;i<ArraySize(g_auditKeys);i++)FileWrite(h,g_auditKeys[i],DoubleToString(g_auditMidnight[i],2),DoubleToString(g_auditMinimum[i],2),DoubleToString(MathMax(0,g_auditMidnight[i]-g_auditMinimum[i]),2),DoubleToString(FTMOTotalFloor(),2),DoubleToString(g_auditMidnight[i]-InpFTMODailyLossEUR,2));
   FileClose(h);
}
void FTMOObserve() {
   if(!g_ftmoReady)return;
   double eq=AccountInfoDouble(ACCOUNT_EQUITY),bal=AccountInfoDouble(ACCOUNT_BALANCE);
   g_ftmoMinEquity=MathMin(g_ftmoMinEquity,eq);g_ftmoMinBalance=MathMin(g_ftmoMinBalance,bal);
   g_ftmoWorstDaily=MathMax(g_ftmoWorstDaily,MathMax(0,g_ftmoMidnight-eq));
   FTMOAuditObserve();
   if(eq<FTMOTotalFloor() || eq<FTMODailyFloor()) {
      if(!g_ftmoFailed){g_ftmoFailureReason=eq<FTMOTotalFloor()?"limite_totale":"limite_giornaliero";PrintFormat("FTMO VIOLAZIONE: equity %.2f, limite %.2f",eq,FTMOFloor());}
      g_ftmoFailed=true;g_ftmoPassed=false;FTMOStoreFailure();
   }
}
void FTMOMarkPassed() {
   if(g_ftmoFailed || g_ftmoPassed || !FTMOInEvaluation(TimeCurrent()))return;
   if(AccountInfoDouble(ACCOUNT_BALANCE)+.00001<FTMOTarget() || ArraySize(g_ftmoDays)<InpFTMOMinTradingDays || PositionsTotal()>0 || OrdersTotal()>0)return;
   g_ftmoPassed=true;g_ftmoPassedServer=TimeCurrent();
   if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("PASSED"),(double)g_ftmoPassedServer);
   PrintFormat("FTMO SUPERATA: saldo %.2f, giorni trading %d",AccountInfoDouble(ACCOUNT_BALANCE),ArraySize(g_ftmoDays));
}
bool FTMOPreRollover(){
 MqlDateTime now;TimeToStruct(TimeCurrent(),now);
 if(now.hour*60+now.min<InpRolloverCheckServerHour*60+InpRolloverCheckServerMinute)return false;
 if(InpCloseFriday && now.day_of_week==5){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();return true;}
 if(InpDailyFlat){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();return true;}
 if(InpOvernightOnlyProtected){
   for(int p=PositionsTotal()-1;p>=0;p--){
     ulong ticket=PositionGetTicket(p);if(ticket==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
     bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
     double entry=PositionGetDouble(POSITION_PRICE_OPEN),sl=PositionGetDouble(POSITION_SL);
     if(sl<=0 || (buy && sl<=entry) || (!buy && sl>=entry)){
       bool sent=trade.PositionClose(ticket);
       if(!sent || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))PrintFormat("FTMO rollover: chiusura non completata %I64u",ticket);
     }
   }
   FTMOObserve();FTMOMarkPassed();
 }

 double used=0;
 for(int p=PositionsTotal()-1;p>=0;p--){
   ulong ticket=PositionGetTicket(p);if(ticket==0)continue;
   string symbol=PositionGetString(POSITION_SYMBOL);double sl=PositionGetDouble(POSITION_SL),pnl=0;MqlTick quote;
   bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
   if(sl<=0 || !SymbolInfoTick(symbol,quote) || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,sl,pnl)){FTMOCloseOwn();return true;}
   used+=MathMax(0,-pnl)+InpFTMOReservePerTradeEUR;
 }
 double futureFloor=MathMax(FTMOTotalFloor(),AccountInfoDouble(ACCOUNT_BALANCE)-InpFTMODailyLossEUR);
 if(AccountInfoDouble(ACCOUNT_EQUITY)-used<=futureFloor+InpFTMOBufferEUR){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();}
 return true; // Nessun nuovo ingresso nella finestra di rollover.
}
bool FTMOUpdate() {
   if(!FTMOEnsureReady())return false;
   if(FTMODayKey(TimeCurrent())!=g_ftmoDay && !FTMORebuildDay())return false;
   FTMOObserve();
   if(g_ftmoFailed || g_ftmoTotalStopped || g_ftmoPassed){FTMOCloseOwn();return false;}
   if(!FTMOInEvaluation(TimeCurrent())){if(FTMOClock(TimeCurrent())>=g_ftmoStartCET)FTMOCloseOwn();return false;}
   double eq=AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq<=FTMOTotalFloor()+InpFTMOBufferEUR){g_ftmoTotalStopped=true;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("TOTAL_STOP"),1);}
   if(eq<=FTMOFloor()+InpFTMOBufferEUR){g_ftmoLockedDay=g_ftmoDay;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("LOCKDAY"),g_ftmoLockedDay);FTMOCloseOwn();return false;}
   if(eq>=FTMOTarget()+InpFTMOReservePerTradeEUR && ArraySize(g_ftmoDays)>=InpFTMOMinTradingDays){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();return false;}
   FTMOMarkPassed();
   if(FTMOPreRollover())return false;
   return !g_ftmoPassed && !g_ftmoTotalStopped && g_ftmoLockedDay!=g_ftmoDay;
}
double FTMORiskCapacity() {
   if(!g_ftmoReady || g_ftmoFailed || g_ftmoPassed || g_ftmoTotalStopped || g_ftmoLockedDay==g_ftmoDay)return 0;
   MqlTick tick;if(!SymbolInfoTick(_Symbol,tick))return 0;
   double used=0;int positions=0;
   for(int p=PositionsTotal()-1;p>=0;p--) {
      ulong t=PositionGetTicket(p);if(t==0)continue;
      string symbol=PositionGetString(POSITION_SYMBOL);bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
      double sl=PositionGetDouble(POSITION_SL),pnl=0;MqlTick quote;
      if(sl<=0 || !SymbolInfoTick(symbol,quote) || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,sl,pnl))return 0;
      used+=MathMax(0,-pnl);positions++;
   }
   if(OrdersTotal()>0)return 0;
   return MathMax(0,AccountInfoDouble(ACCOUNT_EQUITY)-FTMOFloor()-InpFTMOBufferEUR-used-InpFTMOReservePerTradeEUR*(positions+1));
}
double FTMOEntryRisk(double requested) {
   if(AccountInfoDouble(ACCOUNT_BALANCE)>=FTMOTarget() && ArraySize(g_ftmoDays)<InpFTMOMinTradingDays)requested=MathMin(requested,InpFTMOInitialCapital*InpFTMOQualificationRiskPct/100.0);
   return MathMin(requested,FTMORiskCapacity());
}
void FTMOOnDeal(const MqlTradeTransaction &trans) {
   if(trans.type!=TRADE_TRANSACTION_DEAL_ADD || !g_ftmoReady || !HistoryDealSelect(trans.deal))return;
   datetime t=(datetime)HistoryDealGetInteger(trans.deal,DEAL_TIME);long type=HistoryDealGetInteger(trans.deal,DEAL_TYPE),entry=HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
   if(FTMODayKey(t)!=g_ftmoDay && !FTMORebuildDay())return;
   if(FTMOInEvaluation(t) && (type==DEAL_TYPE_BUY || type==DEAL_TYPE_SELL) && (entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT))FTMOAddTradingDay(FTMODayKey(t),t);
   FTMOObserve();FTMOMarkPassed();
}
void FTMOExport() {
   if(!g_ftmoReady)FTMOEnsureReady();
   FTMOObserve();FTMOMarkPassed();
   FTMOExportDaily();
   double actualMinimum=TesterStatistics(STAT_EQUITYMIN);
   if(actualMinimum<FTMOTotalFloor()){g_ftmoFailed=true;g_ftmoPassed=false;g_ftmoFailureReason="limite_totale_statistiche_MT5";}
   string status=g_ftmoFailed?"violazione":g_ftmoPassed?"superato":g_ftmoTotalStopped?"arresto_protezione_totale":"non_superato";
   int calendarDays=g_ftmoPassed?1+(int)((FTMODayStart(FTMOClock(g_ftmoPassedServer))-g_ftmoStartCET)/86400):0;
   string filename=MQLInfoString(MQL_PROGRAM_NAME)+"_FTMO_"+IntegerToString(g_ftmoMonth)+"_result.csv";
   int h=FileOpen(filename,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE){Print("FTMO CSV fallito: ",GetLastError());return;}
   FileWrite(h,"month","status","balance","equity","min_balance","min_equity","max_daily_loss","trading_days","calendar_days_to_pass","passed_server","first_entry_server","failure_reason","target_percent","daily_loss_limit","total_floor","buffer","clock_dst");
   FileWrite(h,g_ftmoMonth,status,DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2),DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2),DoubleToString(TesterStatistics(STAT_BALANCEMIN),2),DoubleToString(actualMinimum,2),DoubleToString(g_ftmoWorstDaily,2),ArraySize(g_ftmoDays),calendarDays,g_ftmoPassed?TimeToString(g_ftmoPassedServer,TIME_DATE|TIME_SECONDS):"",g_ftmoFirstEntry>0?TimeToString(g_ftmoFirstEntry,TIME_DATE|TIME_SECONDS):"",g_ftmoFailureReason,InpFTMOTargetPercent,InpFTMODailyLossEUR,FTMOTotalFloor(),InpFTMOBufferEUR,(int)InpFTMOServerDST);
   FileClose(h);
}

void LoadCompletedORB(datetime start,int duration,bool &loaded,bool &isDefined,double &high,double &low){
 if(loaded || duration<=0 || TimeCurrent()<start+duration*60)return;
 MqlRates rates[];int copied=CopyRates(_Symbol,PERIOD_M1,start,start+duration*60-1,rates);if(copied<duration)return;
 double h=-DBL_MAX,l=DBL_MAX;for(int i=0;i<copied;i++){h=MathMax(h,rates[i].high);l=MathMin(l,rates[i].low);}
 if(h<=l || l<=0)return;high=h;low=l;isDefined=true;loaded=true;
}

int ResearchOwnPositionCount() {
    int count=0;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t>0 && PositionGetString(POSITION_SYMBOL)==_Symbol && PositionGetInteger(POSITION_MAGIC)==(long)InpMagicNumber) count++;
    }
    return count;
}
double ResearchOpenStopRisk() {
    double risk=0;
    MqlTick tick;if(!SymbolInfoTick(_Symbol,tick)) return DBL_MAX;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
        double stop=PositionGetDouble(POSITION_SL),pnl=0;
        if(stop<=0 || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,PositionGetDouble(POSITION_VOLUME),buy?tick.bid:tick.ask,stop,pnl)) return DBL_MAX;
        risk+=MathMax(0,-pnl);
    }
    return risk;
}

void OnTick() {
    if(!FTMOUpdate()) return;
    ManageOpenPositions();
    ManagePyramids();

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
g_lonEntries=0;g_nyEntries=0;g_asiaEntries=0;g_lonLastEntry=0;g_nyLastEntry=0;g_asiaLastEntry=0;
        g_lonLoaded=false;g_nyLoaded=false;g_asiaLoaded=false;
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

    LoadCompletedORB(today+InpLon_StartHour*3600+InpLon_StartMin*60,InpLon_DurationMin,g_lonLoaded,g_lonRangeDefined,g_lonHigh,g_lonLow);
 LoadCompletedORB(today+InpNY_StartHour*3600+InpNY_StartMin*60,InpNY_DurationMin,g_nyLoaded,g_nyRangeDefined,g_nyHigh,g_nyLow);
 if(InpUseAsia)LoadCompletedORB(today+InpAsia_StartHour*3600+InpAsia_StartMin*60,InpAsia_DurationMin,g_asiaLoaded,g_asiaRangeDefined,g_asiaHigh,g_asiaLow);
 if(g_dayLocked) { if(InpShowComments) Comment("STOP TRADING: Max Losses / Daily Loss Limit Reached"); return; }
    if(g_tradesToday >= InpMaxTradesDay) return;
    if(ResearchOwnPositionCount()>=InpMaximumPositions) return;

    double rearmClose=iClose(_Symbol,InpEntryTF,1);datetime rearmTime=iTime(_Symbol,InpEntryTF,1);
if(g_lonExecutedToday && g_lonEntries>0 && g_lonEntries<InpSessionMaxEntries && rearmTime>g_lonLastEntry+InpReentryCooldownMin*60 && rearmClose>g_lonLow && rearmClose<g_lonHigh)g_lonExecutedToday=false;
if(g_nyExecutedToday && g_nyEntries>0 && g_nyEntries<InpSessionMaxEntries && rearmTime>g_nyLastEntry+InpReentryCooldownMin*60 && rearmClose>g_nyLow && rearmClose<g_nyHigh)g_nyExecutedToday=false;
if(g_asiaExecutedToday && g_asiaEntries>0 && g_asiaEntries<InpSessionMaxEntries && rearmTime>g_asiaLastEntry+InpReentryCooldownMin*60 && rearmClose>g_asiaLow && rearmClose<g_asiaHigh)g_asiaExecutedToday=false;
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

    // Gestione rientro REVERSAL dopo stop di una fade (solo NY; Londra non apre piu' fade)
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

    // --- LONDON SESSION LOGIC (v5.5: SENZA FADE) ---
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
                    if(bull && g_dailyBias != 1) bull = false;
                    if(bear && g_dailyBias != -1) bear = false;
                }

                if(bull || bear) {
                    int beforeEntries=g_tradesToday;
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                        // Continuazione: unica logica attiva su Londra
                        if(bull) ExecuteTrade(1, m5_close, g_lonLow, "LONDON", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_lonHigh, "LONDON", riskAmt, 0);
                    }
                    // v5.5: RIMOSSO il ramo "else if(InpFadeEnabled)" -> Londra non apre mai
                    // un LONDON_FADE. Se il corpo e' debole, non si fa nulla e la scansione
                    // prosegue sulle candele successive (g_lonExecutedToday resta false,
                    // dato che sotto viene impostato solo se una posizione risulta aperta).
                    if(g_tradesToday>beforeEntries) g_lonExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= lonScanLimit) {
             g_lonExecutedToday = true;
        }
    }

    // --- NEW YORK SESSION LOGIC (invariata, fade attiva) ---
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
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMultNY) atrOkNY = false;
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
                    int beforeEntries=g_tradesToday;
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
                    if(g_tradesToday>beforeEntries) g_nyExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= nyScanLimit) {
             g_nyExecutedToday = true;
        }
    }

    // --- ASIAN SESSION LOGIC (invariata, fade attiva) ---
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
                    if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMultAsia) atrOkAsia = false;
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
                        int beforeEntries=g_tradesToday;
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
                        if(g_tradesToday>beforeEntries) g_asiaExecutedToday = true;
                    }
                }
            } else if(TimeCurrent() >= asiaScanLimit) {
                g_asiaExecutedToday = true;
            }
        }
    }

    if(InpShowComments) {
        Comment(StringFormat("Tri-Session v5.5 (Lon no-fade) | Bias: %s | Lon H/L: %.2f/%.2f | NY H/L: %.2f/%.2f | Asia H/L: %.2f/%.2f | Trades: %d",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")),
              g_lonHigh, g_lonLow, g_nyHigh, g_nyLow, g_asiaHigh, g_asiaLow, g_tradesToday));
    }
}

// ══════════════════════════════════════════
// 💼 EXECUTION
// ══════════════════════════════════════════
bool FTMODailyAllows(int dir){
 if((dir==1 && !InpDailyFilterLong) || (dir==-1 && !InpDailyFilterShort))return true;
 double ema[];if(CopyBuffer(g_dailyDirectionMA,0,1,1,ema)!=1)return false;
 double closed=iClose(_Symbol,PERIOD_D1,1);
 return dir==1?closed>ema[0]:closed<ema[0];
}
void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
    if(!FTMODailyAllows(dir))return;
    if((dir==1 && !InpAllowLong) || (dir==-1 && !InpAllowShort))return;
    if(ResearchOwnPositionCount()>=InpMaximumPositions) return;
    double availableStopRisk=AccountInfoDouble(ACCOUNT_EQUITY)*InpTotalOpenStopRiskPct/100.0-ResearchOpenStopRisk();
    riskAmount=MathMin(riskAmount,availableStopRisk);
    if(riskAmount<=0) return;

    if(StringFind(session,"ASIA")==0) riskAmount*=InpAsiaRiskMultiplier;
    else if(StringFind(session,"NY")==0) riskAmount*=InpNYRiskMultiplier;
    else if(session=="LONDON") riskAmount*=InpLondonRiskMultiplier;

    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // Rischio e target calcolati sul prezzo negoziabile corrente.
    MqlTick quote;
    if(!SymbolInfoTick(_Symbol,quote)) return;
    entry=dir==1?quote.ask:quote.bid;
    double tickSize=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
    riskAmount=FTMOEntryRisk(riskAmount);
    if(tickSize<=0 || riskAmount<=0 || entry<=0) return;
    double sl_final=NormalizeDouble(MathRound(sl_ref/tickSize)*tickSize,digits);
    double minStop=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
    if((dir==1 && (sl_final>=quote.bid-minStop)) || (dir==-1 && sl_final<=quote.ask+minStop)) return;

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
    // Nessun arrotondamento verso l'alto che superi il rischio scelto.
    if(lots < minLot || stepLot<=0) return;
    lots=MathMin(lots,MathFloor(SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX)/stepLot)*stepLot);
    double margin=0;
    if(!OrderCalcMargin(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,lots,entry,margin) || margin<=0) return;
    double availableMargin=AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.80;
    if(margin>availableMargin) lots=MathFloor(lots*availableMargin/margin/stepLot)*stepLot;
    if(lots<minLot) return;

    string cmt = StringFormat("%s|v5.5|R:%.2f", session, riskAmount);
    bool res = false;

    if(dir == 1) res = trade.Buy(lots, _Symbol, 0.0, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, _Symbol, 0.0, sl_final, tp_final, cmt);

    if(res && (trade.ResultRetcode()==TRADE_RETCODE_DONE || trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL)) {
        g_tradesToday++;
if(session=="LONDON"){g_lonEntries++;g_lonLastEntry=TimeCurrent();}
else if(StringFind(session,"NY")==0){g_nyEntries++;g_nyLastEntry=TimeCurrent();}
else if(StringFind(session,"ASIA")==0){g_asiaEntries++;g_asiaLastEntry=TimeCurrent();}

        long posId=0; double filledEntry=entry;
        ulong filledDeal=trade.ResultDeal();
        if(filledDeal>0 && HistoryDealSelect(filledDeal)) {
            posId=HistoryDealGetInteger(filledDeal,DEAL_POSITION_ID);
            filledEntry=HistoryDealGetDouble(filledDeal,DEAL_PRICE);
        }
        if(posId > 0) RegisterPosition(posId, session, filledEntry, MathAbs(filledEntry - sl_final), dir);

        if(session == "LONDON") g_totalLon++;
        else if(session == "NY") g_totalNY++;
        else if(session == "ASIA") g_totalAsia++;
        else if(session == "NY_FADE") g_totalNYFade++;
        else if(session == "ASIA_FADE") g_totalAsiaFade++;
        else if(session == "NY_REVERSAL") g_totalNYRev++;
        else if(session == "PYRAMID") g_totalPyramid++;

        Print("TRADE OPENED: ", session, " | PosID: ", posId, " | Risk: ", DoubleToString(riskAmount,2), " | TP: ", DoubleToString(tp_final, digits));
    }
}

// ══════════════════════════════════════════
// 📊 STATS TRACKING
// ══════════════════════════════════════════
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    FTMOOnDeal(trans);
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
        int dealsCount = HistoryDealsTotal();
        for(int i=0; i<dealsCount; i++) {
            ulong dticket = HistoryDealGetTicket(i);
            if(HistoryDealGetInteger(dticket, DEAL_POSITION_ID) == posId &&
               HistoryDealGetInteger(dticket, DEAL_MAGIC) == (long)InpMagicNumber &&
               HistoryDealGetInteger(dticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
                totalPnL += HistoryDealGetDouble(dticket, DEAL_PROFIT) +
                            HistoryDealGetDouble(dticket, DEAL_SWAP) +
                            HistoryDealGetDouble(dticket, DEAL_COMMISSION);
            }
        }

        bool stillOpen = false;
        for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetTicket(p)>0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
        if(stillOpen) return;

        if(!IsIdAlreadyCounted(posId)) {
            if(session == "LONDON") { g_profitLon += totalPnL; if(totalPnL > 0) g_winsLon++; }
            else if(session == "NY") { g_profitNY += totalPnL; if(totalPnL > 0) g_winsNY++; }
            else if(session == "ASIA") { g_profitAsia += totalPnL; if(totalPnL > 0) g_winsAsia++; }
            else if(session == "NY_FADE") { g_profitNYFade += totalPnL; if(totalPnL > 0) g_winsNYFade++; }
            else if(session == "ASIA_FADE") { g_profitAsiaFade += totalPnL; if(totalPnL > 0) g_winsAsiaFade++; }
            else if(session == "NY_REVERSAL") { g_profitNYRev += totalPnL; if(totalPnL > 0) g_winsNYRev++; }

            if(session=="PYRAMID") {g_profitPyramid+=totalPnL;if(totalPnL>0) g_winsPyramid++;}
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
//+------------------------------------------------------------------+
// Esporta una riga per posizione, unendo tutti i parziali e includendo costi e swap.
double OnTester() {
    FTMOExport();
    if(!HistorySelect(0,TimeCurrent())) return TesterStatistics(STAT_PROFIT);
    long ids[];
    for(int i=0;i<HistoryDealsTotal();i++) {
        ulong deal=HistoryDealGetTicket(i);
        if(HistoryDealGetString(deal,DEAL_SYMBOL)!=_Symbol || HistoryDealGetInteger(deal,DEAL_MAGIC)!=(long)InpMagicNumber) continue;
        long id=HistoryDealGetInteger(deal,DEAL_POSITION_ID);
        bool found=false;
        for(int j=0;j<ArraySize(ids);j++) if(ids[j]==id) {found=true;break;}
        if(id>0 && !found) {int n=ArraySize(ids);ArrayResize(ids,n+1);ids[n]=id;}
    }
    string filename=MQLInfoString(MQL_PROGRAM_NAME)+"_"+IntegerToString(g_ftmoMonth)+"_trades.csv";
    int handle=FileOpen(filename,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);
    if(handle==INVALID_HANDLE) {Print("CSV non creato: ",GetLastError());return TesterStatistics(STAT_PROFIT);}
    FileWrite(handle,"position_id","session","open_time","close_time","initial_volume","entry_price","initial_sl","initial_tp","net_profit");
    for(int i=0;i<ArraySize(ids);i++) {
        if(!HistorySelectByPosition(ids[i])) continue;
        string session="";datetime opened=0,closed=0;double volume=0,entry=0,sl=0,tp=0,net=0;
        for(int j=0;j<HistoryDealsTotal();j++) {
            ulong deal=HistoryDealGetTicket(j);
            net+=HistoryDealGetDouble(deal,DEAL_PROFIT)+HistoryDealGetDouble(deal,DEAL_COMMISSION)+HistoryDealGetDouble(deal,DEAL_SWAP)+HistoryDealGetDouble(deal,DEAL_FEE);
            if(HistoryDealGetInteger(deal,DEAL_ENTRY)==DEAL_ENTRY_IN && opened==0) {
                opened=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);
                session=HistoryDealGetString(deal,DEAL_COMMENT);int sep=StringFind(session,"|");if(sep>=0) session=StringSubstr(session,0,sep);
                volume=HistoryDealGetDouble(deal,DEAL_VOLUME);entry=HistoryDealGetDouble(deal,DEAL_PRICE);
                sl=HistoryDealGetDouble(deal,DEAL_SL);tp=HistoryDealGetDouble(deal,DEAL_TP);
            }
            if(HistoryDealGetInteger(deal,DEAL_ENTRY)==DEAL_ENTRY_OUT) closed=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);
        }
        FileWrite(handle,ids[i],session,TimeToString(opened,TIME_DATE|TIME_SECONDS),TimeToString(closed,TIME_DATE|TIME_SECONDS),DoubleToString(volume,4),DoubleToString(entry,5),DoubleToString(sl,5),DoubleToString(tp,5),DoubleToString(net,2));
    }
    FileClose(handle);
    Print("CSV creato: ",TerminalInfoString(TERMINAL_COMMONDATA_PATH),"\\Files\\",filename);
    return TesterStatistics(STAT_PROFIT);
}
