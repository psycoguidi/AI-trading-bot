// prova185_20k_FTMO: derivata da prova183, FTMO CFD 1-Step EUR 20.000.
// Regole consultate il 14/10/2026; Consistency sul profitto netto totale.
// Modalita Challenge e Funded Standard; nessun reset automatico del capitale.
// Compilazione e controlli tecnici non garantiscono risultati o conformita FTMO.
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
#property copyright "prova185 - ORB FTMO 1-Step 20k"
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
input double   InpFixedRiskAmount   = 150.0;
input bool     InpUseEquityRisk     = true;
input double   InpRiskPercent       = 0.75;
input double   InpRR_Target         = 2;
input int      InpMaxLossesDay      = 4;
input int      InpMaxTradesDay      = 12;
input ulong    InpMagicNumber       = 2026101485;
input bool     InpShowComments      = true;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = false;
input double   InpMaxDailyLossPct    = 3.0;

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ══════════════════════════════════════════
input double InpPyramidAtR=1.2;
input double InpPyramidRiskPercent=0.20;
input double InpPyramidExtraTargetR=1.0;
input double InpTotalOpenStopRiskPct=1.5;
double g_profitPyramid=0; int g_winsPyramid=0,g_totalPyramid=0;
input double InpAsiaRiskMultiplier=1.05;
input double InpNYRiskMultiplier=0.8;
input double InpLondonRiskMultiplier=1.1;
// FTMO 1-Step: le regole valgono sull'intero conto, non sul solo Magic.
enum ENUM_FTMO_SERVER_DST { FTMO_DST_EU=0, FTMO_DST_US=1, FTMO_DST_NONE=2 };
enum ENUM_FTMO_PHASE { FTMO_CHALLENGE=0, FTMO_FUNDED_STANDARD=1 };
input group "FTMO 1-Step EUR 20k - fase e rischio"
input ENUM_FTMO_PHASE InpFTMOPhase=FTMO_CHALLENGE;
input double InpFTMOInitialCapital=20000.0;
input double InpFTMOTargetPercent=10.0;
input double InpFTMODailyLossEUR=600.0;
input double InpFTMOTotalLossEUR=2000.0;
input double InpFTMOBufferEUR=150.0;
input double InpFTMOReservePerTradeEUR=10.0;
input double InpFTMOExtraCostPerLotEUR=12.0; // Riserva aggiuntiva per commissioni/slippage.
input double InpFTMOQualificationRiskPct=0.50;
input double InpFTMOChallengeDayCapPct=45.0; // % del target; 0 disattiva il tetto prudenziale.
input double InpFTMOFundedDayCapEUR=0.0; // Nel funded continua, senza un target fisso.
input datetime InpFTMOPhaseStartServer=0; // 0 = intero storico del conto dedicato.
input int InpFTMOServerUTCWinter=2;
input int InpFTMOServerUTCSummer=3;
input ENUM_FTMO_SERVER_DST InpFTMOServerDST=FTMO_DST_EU;
input bool InpFTMOBrokerClockConfirmed=false; // Confermare solo dopo verifica sul server FTMO.
input group "Funded Standard - notizie e chiusure"
input int InpFTMONewsCloseAheadMin=15;
input int InpFTMONewsResumeAfterMin=3;
input int InpFTMONewsRefreshSec=300;
input int InpFTMOMarketCloseAheadMin=15;
input bool InpFTMOFlatBeforeAllBreaks=true; // Include anche le pause <=2h, per prudenza.
input string InpFTMOExtraClosuresServer=""; // Chiusure straordinarie: date server separate da ;.
input string InpFTMONewsTesterCSV=""; // FILE_COMMON: coverage;UTC_da;UTC_a, poi event;UTC_data;nome.
input group "Solo tester: periodo della prova indipendente"
input int InpResearchMonth=0;
input long InpResearchStopServer=0;
bool g_ftmoReady=false,g_ftmoFailed=false,g_ftmoPassed=false,g_ftmoTotalStopped=false;
bool g_ftmoHistoryDirty=true,g_ftmoHistoryOK=false,g_ftmoExecuting=false;
int g_ftmoDay=0,g_ftmoLockedDay=0,g_ftmoDays[],g_ftmoMonth=0;
double g_ftmoMidnight=20000,g_ftmoHighestMidnight=20000;
double g_ftmoMinEquity=20000,g_ftmoMinBalance=20000,g_ftmoWorstDaily=0;
double g_ftmoNetClosed=0,g_ftmoBestDay=0,g_ftmoTodayClosed=0;
datetime g_ftmoStartCET=0,g_ftmoEndCET=0,g_ftmoPassedServer=0,g_ftmoFirstEntry=0;
datetime g_ftmoNewsLoaded=0,g_ftmoCSVFrom=0,g_ftmoCSVTo=0,g_ftmoLastSync=0;
datetime g_ftmoNewsUTC[],g_ftmoExtraClosures[];
bool g_ftmoNewsReady=false,g_ftmoCSVLoaded=false;
string g_ftmoFailureReason="",g_ftmoReason="",g_ftmoLastReason="";
int g_ftmoPnlKeys[]; double g_ftmoPnl[];
double g_auditTotalFloor[],g_auditDailyFloor[],g_auditBestDay[],g_auditNetClosed[];
bool g_ftmoInCriticalNews=false;

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
input double InpBelowInitialRiskPercent=0.50;
input bool InpOvernightOnlyProtected=true;
input bool InpDailyFlat=false;
datetime g_ftmoElapsedStartCET=0;
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

    if(!FTMOValidateInputs())return INIT_PARAMETERS_INCORRECT;
    if(!MQLInfoInteger(MQL_TESTER) && !InpFTMOBrokerClockConfirmed){Print("Verificare orario/DST del broker FTMO e impostare InpFTMOBrokerClockConfirmed. Nessun ordine inviato.");return INIT_PARAMETERS_INCORRECT;}

    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFillingBySymbol(_Symbol);

    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    ArrayResize(g_posSessionMap, 0);

    g_atrHandle = iATR(_Symbol, PERIOD_D1, InpATRPeriod);
    if(InpUseATRFilter && g_atrHandle == INVALID_HANDLE)return INIT_FAILED;

    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

    RestoreDailyStats();
    FTMOLoadExtraClosures();
    FTMORestorePositions();
    EventSetTimer(5);
    Print("prova185_20k_FTMO pronta: ORB prova183, capitale 20000 EUR, fase ",EnumToString(InpFTMOPhase));
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    EventKillTimer();
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
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) {
            g_tradesToday++;
            string s=HistoryDealGetString(ticket,DEAL_COMMENT);datetime t=(datetime)HistoryDealGetInteger(ticket,DEAL_TIME);
            if(StringFind(s,"LONDON|")==0){g_lonEntries++;g_lonLastEntry=MathMax(g_lonLastEntry,t);}
            else if(StringFind(s,"NY")==0){g_nyEntries++;g_nyLastEntry=MathMax(g_nyLastEntry,t);}
            else if(StringFind(s,"ASIA")==0){g_asiaEntries++;g_asiaLastEntry=MathMax(g_asiaLastEntry,t);}
        }
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
            double ts=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
            if(ts<=0)continue;
            beSL=NormalizeDouble(MathRound(beSL/ts)*ts,digits);
            double minDistance=MathMax(SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL))*SymbolInfoDouble(_Symbol,SYMBOL_POINT);
            if((dir==1 && beSL>=currentPrice-minDistance) || (dir==-1 && beSL<=currentPrice+minDistance))continue;
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
        if(idx>=0 && g_tradesToday>previous) {g_posSessionMap[idx].pyramidDone=true;
         if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOPositionKey(id,"PYR"),1);}
        return;
    }
}

// Motore conto FTMO 1-Step; nessun azzeramento mensile o dopo un riavvio.
bool FTMOChallenge(){return InpFTMOPhase==FTMO_CHALLENGE;}
datetime FTMONow(){return MQLInfoInteger(MQL_TESTER)?TimeCurrent():TimeTradeServer();}
bool FTMOValidateInputs(){
 if(AccountInfoString(ACCOUNT_CURRENCY)!="EUR") {Print("prova185 richiede un conto in EUR.");return false;}
 if(InpFTMOInitialCapital!=20000 || InpFTMODailyLossEUR!=600 || InpFTMOTotalLossEUR!=2000 || InpFTMOTargetPercent!=10){Print("Profilo richiesto: 20000 EUR, target 10%, perdita giornaliera 600, totale 2000.");return false;}
 if(InpFTMOPhase!=FTMO_CHALLENGE && InpFTMOPhase!=FTMO_FUNDED_STANDARD)return false;
 if(InpFTMOBufferEUR<50 || InpFTMOBufferEUR>=600 || InpFTMOReservePerTradeEUR<0 || InpFTMOExtraCostPerLotEUR<0 || InpFTMOQualificationRiskPct<=0)return false;
 if(InpRiskPercent<=0 || InpFixedRiskAmount<=0 || InpBelowInitialRiskPercent<=0 || InpTotalOpenStopRiskPct<=0 || InpRR_Target<=0)return false;
 if(InpFTMOChallengeDayCapPct<0 || InpFTMOChallengeDayCapPct>50 || InpFTMOFundedDayCapEUR<0)return false;
 if(InpFTMONewsCloseAheadMin<5 || InpFTMONewsResumeAfterMin<3 || InpFTMONewsRefreshSec<30 || InpFTMONewsRefreshSec>900 || InpFTMOMarketCloseAheadMin<5)return false;
 if(InpMaximumPositions<1 || InpMaxTradesDay<1 || InpSessionMaxEntries<1 || InpATRPeriod<1 || InpReentryCooldownMin<0)return false;
 if(InpFTMOPhaseStartServer>0 && InpResearchMonth>0)return false;
 if(InpFTMOServerUTCWinter<-12 || InpFTMOServerUTCWinter>14 || InpFTMOServerUTCSummer<-12 || InpFTMOServerUTCSummer>14)return false;
 if(InpResearchMonth>0 && (InpResearchMonth%100<1 || InpResearchMonth%100>12 || InpResearchMonth/100<2020))return false;
 if(InpFTMOExtraClosuresServer!=""){
  string dates[];int count=StringSplit(InpFTMOExtraClosuresServer,';',dates);
  for(int i=0;i<count;i++)if(StringToTime(dates[i])<=0){Print("Chiusura straordinaria non valida: ",dates[i]);return false;}
 }
 return true;
}
void FTMOExplain(string reason){
 g_ftmoReason=reason;
 if(reason!=g_ftmoLastReason){Print("FTMO185: ",reason);g_ftmoLastReason=reason;}
 if(InpShowComments)Comment(StringFormat("prova185 | %s\n%s\nSaldo %.2f | Equity %.2f | Soglia giorno %.2f | Soglia totale %.2f\nProfitto netto chiuso %.2f | Miglior giorno %.2f | Consistency %.2f%%",
    EnumToString(InpFTMOPhase),reason,AccountInfoDouble(ACCOUNT_BALANCE),AccountInfoDouble(ACCOUNT_EQUITY),FTMODailyFloor(),FTMOTotalFloor(),g_ftmoNetClosed,g_ftmoBestDay,FTMOConsistencyPercent()));
}
int FTMOLastSunday(int year,int month){
 MqlDateTime d={};d.year=year;d.mon=month+1;d.day=1;if(month==12){d.year++;d.mon=1;}
 datetime t=StructToTime(d)-86400;TimeToStruct(t,d);return d.day-d.day_of_week;
}
datetime FTMOEUChange(int year,int month){
 MqlDateTime d={};d.year=year;d.mon=month;d.day=FTMOLastSunday(year,month);d.hour=1;return StructToTime(d);
}
datetime FTMOUSChange(int year,int month){
 MqlDateTime d={};d.year=year;d.mon=month;d.day=1;TimeToStruct(StructToTime(d),d);
 d.day=1+(7-d.day_of_week)%7+(month==3?7:0);d.hour=month==3?7:6;return StructToTime(d);
}
int FTMOServerOffset(datetime server){
 if(InpFTMOServerDST==FTMO_DST_NONE)return InpFTMOServerUTCWinter;
 MqlDateTime d;TimeToStruct(server,d);
 datetime a=InpFTMOServerDST==FTMO_DST_US?FTMOUSChange(d.year,3):FTMOEUChange(d.year,3);
 datetime b=InpFTMOServerDST==FTMO_DST_US?FTMOUSChange(d.year,11):FTMOEUChange(d.year,10);
 a+=InpFTMOServerUTCWinter*3600;b+=InpFTMOServerUTCSummer*3600;
 return server>=a && server<b?InpFTMOServerUTCSummer:InpFTMOServerUTCWinter;
}
datetime FTMOClock(datetime server){
 datetime utc=server-FTMOServerOffset(server)*3600;MqlDateTime d;TimeToStruct(utc,d);
 int cet=utc>=FTMOEUChange(d.year,3) && utc<FTMOEUChange(d.year,10)?2:1;
 return utc+cet*3600;
}
int FTMODayKey(datetime server){MqlDateTime d;TimeToStruct(FTMOClock(server),d);return d.year*10000+d.mon*100+d.day;}
datetime FTMODayStart(datetime cet){MqlDateTime d;TimeToStruct(cet,d);d.hour=0;d.min=0;d.sec=0;return StructToTime(d);}
uint FTMOServerHash(){
 string name=AccountInfoString(ACCOUNT_SERVER);uint hash=2166136261;
 for(int i=0;i<StringLen(name);i++){hash^=(uint)StringGetCharacter(name,i);hash*=16777619;}
 return hash;
}
// Il Magic non fa parte dello stato del conto: cambiarlo non azzera i limiti.
string FTMOStateKey(string suffix){return StringFormat("P185_%I64d_%u_%d_%I64d_%s",AccountInfoInteger(ACCOUNT_LOGIN),FTMOServerHash(),(int)InpFTMOPhase,(long)InpFTMOPhaseStartServer,suffix);}
string FTMOPositionKey(long id,string suffix){return StringFormat("P185P_%I64d_%I64d_%s",AccountInfoInteger(ACCOUNT_LOGIN),id,suffix);}
void FTMOPersistPosition(long id,double distance){if(id>0 && distance>0 && !MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOPositionKey(id,"R"),distance);}
double FTMODealDelta(ulong t){return HistoryDealGetDouble(t,DEAL_PROFIT)+HistoryDealGetDouble(t,DEAL_SWAP)+HistoryDealGetDouble(t,DEAL_COMMISSION)+HistoryDealGetDouble(t,DEAL_FEE);}
bool FTMOInEvaluation(datetime server){
 if(InpFTMOPhaseStartServer>0 && server<InpFTMOPhaseStartServer)return false;
 datetime cet=FTMOClock(server);
 if(cet<g_ftmoStartCET || (g_ftmoEndCET>0 && cet>=g_ftmoEndCET))return false;
 if(MQLInfoInteger(MQL_TESTER) && InpResearchStopServer>0 && (long)server>=InpResearchStopServer)return false;
 return true;
}
void FTMOAddTradingDay(int key,datetime entered){
 if(g_ftmoFirstEntry==0 || entered<g_ftmoFirstEntry)g_ftmoFirstEntry=entered;
 for(int i=0;i<ArraySize(g_ftmoDays);i++)if(g_ftmoDays[i]==key)return;
 int n=ArraySize(g_ftmoDays);ArrayResize(g_ftmoDays,n+1);g_ftmoDays[n]=key;
}
void FTMOAddClosedDay(int key,double pnl){
 int n=ArraySize(g_ftmoPnlKeys);
 if(n==0 || g_ftmoPnlKeys[n-1]!=key){ArrayResize(g_ftmoPnlKeys,n+1);ArrayResize(g_ftmoPnl,n+1);g_ftmoPnlKeys[n]=key;g_ftmoPnl[n]=0;n++;}
 g_ftmoPnl[n-1]+=pnl;
}
double FTMOTrailingFloor(double highestMidnight){return MathMax(InpFTMOInitialCapital,highestMidnight)-InpFTMOTotalLossEUR;}
double FTMOTotalFloor(){return FTMOTrailingFloor(g_ftmoHighestMidnight);}
double FTMODailyFloor(){return g_ftmoMidnight-InpFTMODailyLossEUR;}
double FTMOFloor(){return MathMax(FTMOTotalFloor(),FTMODailyFloor());}
double FTMOTarget(){return InpFTMOInitialCapital*(1+InpFTMOTargetPercent/100.0);}
bool FTMOConsistencyOK(double net,double best){return net>0 && best<=0.5*net+0.0000001;}
double FTMOConsistencyPercent(){return g_ftmoNetClosed>0?100*g_ftmoBestDay/g_ftmoNetClosed:0;}
// Ricostruzione cronologica account-wide: il picco INTRADAY non aumenta il trailing.
bool FTMORebuildDay(){
 datetime now=FTMONow();if(now<=0 || !HistorySelect(0,now))return false;
 int today=FTMODayKey(now),previous=0;double running=InpFTMOInitialCapital,midnight=running,highest=running;
 long lastMillis=-1;
 long ids[];double remainingVol[],remainingCost[];
 ArrayResize(g_ftmoDays,0);ArrayResize(g_ftmoPnlKeys,0);ArrayResize(g_ftmoPnl,0);g_ftmoFirstEntry=0;
 int nDeals=HistoryDealsTotal();
 for(int i=0;i<nDeals;i++){
  ulong deal=HistoryDealGetTicket(i);if(deal==0)return false;
  datetime t=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);long ms=HistoryDealGetInteger(deal,DEAL_TIME_MSC);
  if(ms<lastMillis){FTMOExplain("Storico fuori ordine cronologico: ingressi sospesi.");return false;}lastMillis=ms;
  if(!FTMOInEvaluation(t))continue;
  long type=HistoryDealGetInteger(deal,DEAL_TYPE);
  if(type==DEAL_TYPE_BALANCE || type==DEAL_TYPE_CREDIT)continue;
  int key=FTMODayKey(t);if(key>today)continue;
  if(key!=previous){if(previous>0)highest=MathMax(highest,running);previous=key;if(key==today)midnight=running;}
  double delta=FTMODealDelta(deal);running+=delta;
  if(type==DEAL_TYPE_BUY || type==DEAL_TYPE_SELL){
   long entry=HistoryDealGetInteger(deal,DEAL_ENTRY),id=HistoryDealGetInteger(deal,DEAL_POSITION_ID);
   int idx=-1;for(int j=0;j<ArraySize(ids);j++)if(ids[j]==id){idx=j;break;}
   if(idx<0){idx=ArraySize(ids);ArrayResize(ids,idx+1);ArrayResize(remainingVol,idx+1);ArrayResize(remainingCost,idx+1);ids[idx]=id;remainingVol[idx]=0;remainingCost[idx]=0;}
   double volume=HistoryDealGetDouble(deal,DEAL_VOLUME);
   if(entry==DEAL_ENTRY_IN){remainingVol[idx]+=volume;remainingCost[idx]+=delta;FTMOAddTradingDay(key,t);}
   else if(entry==DEAL_ENTRY_OUT || entry==DEAL_ENTRY_OUT_BY){
    // Attribuisce al giorno di chiusura anche il costo d'ingresso, pro quota nei parziali.
    if(remainingVol[idx]+0.0000001<volume){FTMOExplain("Storico d'ingresso incompleto: impossibile ricostruire i costi dei parziali.");return false;}
    double allocated=remainingVol[idx]>0?remainingCost[idx]*volume/remainingVol[idx]:0;
    remainingCost[idx]-=allocated;remainingVol[idx]=MathMax(0,remainingVol[idx]-volume);
    FTMOAddClosedDay(key,delta+allocated);
   }else if(entry==DEAL_ENTRY_INOUT){
    FTMOExplain("Inversione netting nello storico: usare un conto dedicato hedging per questa versione.");return false;
   }
  }else FTMOAddClosedDay(key,delta); // Commissioni giornaliere e altri costi contabilizzati.
 }
 if(previous!=today){highest=MathMax(highest,running);midnight=running;}
 double balance=AccountInfoDouble(ACCOUNT_BALANCE);
 if(MathAbs(balance-running)>0.05 || MathAbs(AccountInfoDouble(ACCOUNT_CREDIT))>0.01){
  FTMOExplain(StringFormat("Storico/capitale non coerente: saldo %.2f, ricostruito %.2f. Richiesto conto dedicato EUR 20000, senza depositi o prelievi intermedi.",balance,running));return false;
 }
 g_ftmoMidnight=midnight;g_ftmoHighestMidnight=highest;g_ftmoDay=today;
 g_ftmoNetClosed=0;g_ftmoBestDay=0;g_ftmoTodayClosed=0;
 for(int i=0;i<ArraySize(g_ftmoPnl);i++){
  g_ftmoNetClosed+=g_ftmoPnl[i];g_ftmoBestDay=MathMax(g_ftmoBestDay,g_ftmoPnl[i]);
  if(g_ftmoPnlKeys[i]==today)g_ftmoTodayClosed=g_ftmoPnl[i];
 }
 if(!MQLInfoInteger(MQL_TESTER)){
  if(GlobalVariableCheck(FTMOStateKey("HWM")))g_ftmoHighestMidnight=MathMax(g_ftmoHighestMidnight,GlobalVariableGet(FTMOStateKey("HWM")));
  if(GlobalVariableCheck(FTMOStateKey("DAY")) && (int)GlobalVariableGet(FTMOStateKey("DAY"))==today && GlobalVariableCheck(FTMOStateKey("MID")))g_ftmoMidnight=MathMax(g_ftmoMidnight,GlobalVariableGet(FTMOStateKey("MID")));
  GlobalVariableSet(FTMOStateKey("HWM"),g_ftmoHighestMidnight);GlobalVariableSet(FTMOStateKey("DAY"),today);GlobalVariableSet(FTMOStateKey("MID"),g_ftmoMidnight);
  if(GlobalVariableCheck(FTMOStateKey("LOCK")))g_ftmoLockedDay=(int)GlobalVariableGet(FTMOStateKey("LOCK"));
 }
 g_ftmoHistoryDirty=false;g_ftmoHistoryOK=true;g_ftmoLastSync=now;return true;
}
bool FTMOEnsureReady(){
 if(g_ftmoReady)return true;
 datetime now=FTMONow();if(now<=0)return false;MqlDateTime d;TimeToStruct(FTMOClock(now),d);
 g_ftmoMonth=MQLInfoInteger(MQL_TESTER) && InpResearchMonth>0?InpResearchMonth:d.year*100+d.mon;
 if(MQLInfoInteger(MQL_TESTER) && InpResearchMonth>0){MqlDateTime a={};a.year=g_ftmoMonth/100;a.mon=g_ftmoMonth%100;a.day=1;g_ftmoStartCET=StructToTime(a);a.mon++;if(a.mon>12){a.year++;a.mon=1;}g_ftmoEndCET=StructToTime(a);}
 if(!FTMORebuildDay())return false;
 if(!MQLInfoInteger(MQL_TESTER)){
  g_ftmoFailed=GlobalVariableCheck(FTMOStateKey("FAIL"));g_ftmoTotalStopped=GlobalVariableCheck(FTMOStateKey("STOP"));
  if(FTMOChallenge() && GlobalVariableCheck(FTMOStateKey("PASS"))){g_ftmoPassed=true;g_ftmoPassedServer=(datetime)GlobalVariableGet(FTMOStateKey("PASS"));}
 }
 g_ftmoElapsedStartCET=FTMODayStart(FTMOClock(g_ftmoFirstEntry>0?g_ftmoFirstEntry:now));
 g_ftmoMinEquity=AccountInfoDouble(ACCOUNT_EQUITY);g_ftmoMinBalance=AccountInfoDouble(ACCOUNT_BALANCE);g_ftmoReady=true;
 return true;
}
void FTMOStoreFailure(){if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("FAIL"),1);}
void FTMOAuditObserve(){
 if(g_auditIndex<0 || g_auditKeys[g_auditIndex]!=g_ftmoDay){
  g_auditIndex=ArraySize(g_auditKeys);int n=g_auditIndex+1;
  ArrayResize(g_auditKeys,n);ArrayResize(g_auditMidnight,n);ArrayResize(g_auditMinimum,n);ArrayResize(g_auditTotalFloor,n);ArrayResize(g_auditDailyFloor,n);ArrayResize(g_auditBestDay,n);ArrayResize(g_auditNetClosed,n);
  g_auditKeys[g_auditIndex]=g_ftmoDay;g_auditMidnight[g_auditIndex]=g_ftmoMidnight;g_auditMinimum[g_auditIndex]=AccountInfoDouble(ACCOUNT_EQUITY);
  g_auditTotalFloor[g_auditIndex]=FTMOTotalFloor();g_auditDailyFloor[g_auditIndex]=FTMODailyFloor();
 }
 g_auditMinimum[g_auditIndex]=MathMin(g_auditMinimum[g_auditIndex],AccountInfoDouble(ACCOUNT_EQUITY));
 g_auditBestDay[g_auditIndex]=g_ftmoBestDay;g_auditNetClosed[g_auditIndex]=g_ftmoNetClosed;
}
void FTMOObserve(){
 if(!g_ftmoReady || !g_ftmoHistoryOK)return;
 double eq=AccountInfoDouble(ACCOUNT_EQUITY),bal=AccountInfoDouble(ACCOUNT_BALANCE);
 g_ftmoMinEquity=MathMin(g_ftmoMinEquity,eq);g_ftmoMinBalance=MathMin(g_ftmoMinBalance,bal);
 g_ftmoWorstDaily=MathMax(g_ftmoWorstDaily,MathMax(0,g_ftmoMidnight-eq));FTMOAuditObserve();
 if(eq<FTMOFloor()){
  if(!g_ftmoFailed){g_ftmoFailureReason=eq<FTMOTotalFloor()?"limite_totale":"limite_giornaliero";PrintFormat("FTMO185 VIOLAZIONE OSSERVATA: equity %.2f, limite %.2f",eq,FTMOFloor());}
  g_ftmoFailed=true;g_ftmoPassed=false;FTMOStoreFailure();
 }
}
bool FTMOTradeResultOK(){uint code=trade.ResultRetcode();return code==TRADE_RETCODE_DONE || code==TRADE_RETCODE_DONE_PARTIAL;}
void FTMOCloseOwn(){
 // Durante +/-2 minuti o con calendario sconosciuto non invia ordini di chiusura.
 if(!FTMOChallenge() && g_ftmoInCriticalNews)return;
 if(g_ftmoExecuting)return;g_ftmoExecuting=true;
 for(int p=OrdersTotal()-1;p>=0;p--){ulong t=OrderGetTicket(p);if(t==0 || OrderGetInteger(ORDER_MAGIC)!=(long)InpMagicNumber || OrderGetString(ORDER_SYMBOL)!=_Symbol)continue;
  trade.SetTypeFillingBySymbol(_Symbol);if(trade.OrderDelete(t) && FTMOTradeResultOK())g_ftmoHistoryDirty=true;}
 for(int p=PositionsTotal()-1;p>=0;p--){ulong t=PositionGetTicket(p);if(t==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
  trade.SetTypeFillingBySymbol(_Symbol);bool sent=trade.PositionClose(t);
  if(!sent || !FTMOTradeResultOK())FTMOExplain(StringFormat("Chiusura da ritentare, ticket %I64u, codice %u: %s",t,trade.ResultRetcode(),trade.ResultRetcodeDescription()));else g_ftmoHistoryDirty=true;
 }
 g_ftmoExecuting=false;
}
void FTMOMarkPassed(){
 if(!FTMOChallenge() || !g_ftmoHistoryOK || g_ftmoHistoryDirty || g_ftmoFailed || g_ftmoPassed || !FTMOInEvaluation(FTMONow()))return;
 if(AccountInfoDouble(ACCOUNT_BALANCE)<FTMOTarget() || !FTMOConsistencyOK(g_ftmoNetClosed,g_ftmoBestDay) || PositionsTotal()>0 || OrdersTotal()>0)return;
 g_ftmoPassed=true;g_ftmoPassedServer=FTMONow();
 if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("PASS"),(double)g_ftmoPassedServer);
 FTMOExplain("Obiettivi calcolati raggiunti: EA fermo. Attendere la conferma ufficiale di Account MetriX.");
}
double FTMOOpenStopRisk(){
 double used=0;
 for(int p=PositionsTotal()-1;p>=0;p--){ulong t=PositionGetTicket(p);if(t==0)return DBL_MAX;
  string symbol=PositionGetString(POSITION_SYMBOL);bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
  double sl=PositionGetDouble(POSITION_SL),pnl=0;MqlTick quote;
  if(sl<=0 || !SymbolInfoTick(symbol,quote) || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,sl,pnl))return DBL_MAX;
  used+=MathMax(0,-pnl)+PositionGetDouble(POSITION_VOLUME)*InpFTMOExtraCostPerLotEUR;
 }
 return used;
}
double FTMOCapacity(double equity,double floor,double used,int count){return MathMax(0,equity-floor-InpFTMOBufferEUR-used-InpFTMOReservePerTradeEUR*(count+1));}
double FTMORiskCapacity(){
 if(!g_ftmoReady || !g_ftmoHistoryOK || g_ftmoHistoryDirty || g_ftmoFailed || g_ftmoPassed || g_ftmoTotalStopped || g_ftmoLockedDay==g_ftmoDay || OrdersTotal()>0)return 0;
 double used=FTMOOpenStopRisk();if(used==DBL_MAX)return 0;
 return FTMOCapacity(AccountInfoDouble(ACCOUNT_EQUITY),FTMOFloor(),used,PositionsTotal());
}
double FTMOEntryRisk(double requested){
 if(MathMin(AccountInfoDouble(ACCOUNT_BALANCE),AccountInfoDouble(ACCOUNT_EQUITY))<InpFTMOInitialCapital)requested=MathMin(requested,InpFTMOInitialCapital*InpBelowInitialRiskPercent/100.0);
 if(FTMOChallenge() && AccountInfoDouble(ACCOUNT_BALANCE)>=FTMOTarget() && !FTMOConsistencyOK(g_ftmoNetClosed,g_ftmoBestDay))requested=MathMin(requested,InpFTMOInitialCapital*InpFTMOQualificationRiskPct/100.0);
 return MathMin(requested,FTMORiskCapacity());
}
bool FTMOPreRollover(){
 MqlDateTime now;TimeToStruct(FTMONow(),now);
 if(now.hour*60+now.min<InpRolloverCheckServerHour*60+InpRolloverCheckServerMinute)return false;
 if((InpCloseFriday && now.day_of_week==5) || InpDailyFlat){FTMOCloseOwn();return true;}
 if(InpOvernightOnlyProtected){
  for(int p=PositionsTotal()-1;p>=0;p--){ulong t=PositionGetTicket(p);if(t==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
   bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;double entry=PositionGetDouble(POSITION_PRICE_OPEN),sl=PositionGetDouble(POSITION_SL);
   if(sl<=0 || (buy && sl<=entry) || (!buy && sl>=entry)){trade.SetTypeFillingBySymbol(_Symbol);if(trade.PositionClose(t) && FTMOTradeResultOK())g_ftmoHistoryDirty=true;}
  }
 }
 double futureHighest=MathMax(g_ftmoHighestMidnight,AccountInfoDouble(ACCOUNT_BALANCE));
 double futureFloor=MathMax(FTMOTrailingFloor(futureHighest),AccountInfoDouble(ACCOUNT_BALANCE)-InpFTMODailyLossEUR);
 double used=FTMOOpenStopRisk()+InpFTMOReservePerTradeEUR*PositionsTotal();
 if(AccountInfoDouble(ACCOUNT_EQUITY)-used<=futureFloor+InpFTMOBufferEUR)FTMOCloseOwn();
 return true;
}
bool FTMORestrictedEvent(MqlCalendarEvent &event){
 // Include tutti gli eventi US/USD ad alta importanza: scelta prudenziale piu' ampia della lista FTMO.
 string c=event.event_code;
 return event.importance==CALENDAR_IMPORTANCE_HIGH || c=="fed-interest-rate-decision" || c=="fomc-meeting-statement" || c=="fomc-minutes" ||
 c=="nonfarm-payrolls" || c=="unemployment-rate" || c=="average-hourly-earnings-mm" || c=="average-hourly-earnings-yy" ||
 c=="consumer-price-index-yy" || c=="gross-domestic-product-qq";
}
bool FTMOLoadNewsCSV(){
 if(g_ftmoCSVLoaded)return g_ftmoNewsReady;g_ftmoCSVLoaded=true;
 if(InpFTMONewsTesterCSV==""){FTMOExplain("Funded nel tester: serve un CSV storico notizie con copertura UTC. Nessun filtro viene disattivato automaticamente.");return false;}
 int h=FileOpen(InpFTMONewsTesterCSV,FILE_COMMON|FILE_READ|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE)return false;
 string tag=FileReadString(h);string a=FileReadString(h),b=FileReadString(h);
 if(tag!="coverage"){FileClose(h);return false;}
 g_ftmoCSVFrom=StringToTime(a);g_ftmoCSVTo=StringToTime(b);ArrayResize(g_ftmoNewsUTC,0);
 while(!FileIsEnding(h)){
  tag=FileReadString(h);if(tag=="" && FileIsEnding(h))break;
  string date=FileReadString(h),label=FileReadString(h);datetime t=StringToTime(date);
  if(tag!="event" || t<g_ftmoCSVFrom || t>=g_ftmoCSVTo){FileClose(h);return false;}
  int n=ArraySize(g_ftmoNewsUTC);ArrayResize(g_ftmoNewsUTC,n+1);g_ftmoNewsUTC[n]=t;
 }
 FileClose(h);ArraySort(g_ftmoNewsUTC);g_ftmoNewsReady=g_ftmoCSVFrom>0 && g_ftmoCSVTo>g_ftmoCSVFrom && ArraySize(g_ftmoNewsUTC)>0;
 return g_ftmoNewsReady;
}
bool FTMOLoadNewsLive(){
 datetime now=FTMONow();if(g_ftmoNewsReady && now>=g_ftmoNewsLoaded && now-g_ftmoNewsLoaded<InpFTMONewsRefreshSec)return true;
 if(!g_ftmoNewsReady && g_ftmoNewsLoaded>0 && now-g_ftmoNewsLoaded<30)return false;
 g_ftmoNewsReady=false;g_ftmoNewsLoaded=now;MqlCalendarValue values[];ResetLastError();
 int count=CalendarValueHistory(values,now-86400,now+7*86400,"US","USD");
 if(count<=0 || GetLastError()!=0)return false;
 datetime events[];
 for(int i=0;i<count;i++){
  MqlCalendarEvent event;if(!CalendarEventById(values[i].event_id,event))return false;
  if(!FTMORestrictedEvent(event))continue;
  if(event.time_mode!=CALENDAR_TIMEMODE_DATETIME)return false;
  int n=ArraySize(events);ArrayResize(events,n+1);events[n]=values[i].time-FTMOServerOffset(values[i].time)*3600;
 }
 if(ArraySize(events)==0)return false;
 ArraySort(events);ArrayCopy(g_ftmoNewsUTC,events);g_ftmoNewsReady=true;return true;
}
bool FTMOInsideNewsWindow(datetime utc,datetime event,int before,int after){return utc>=event-before*60 && utc<=event+after*60;}
bool FTMONewsAllowed(){
 g_ftmoInCriticalNews=false;if(FTMOChallenge())return true;
 datetime now=FTMONow(),utc=now-FTMOServerOffset(now)*3600;
 bool ready=MQLInfoInteger(MQL_TESTER)?FTMOLoadNewsCSV():FTMOLoadNewsLive();
 if(!ready || (MQLInfoInteger(MQL_TESTER) && (utc<g_ftmoCSVFrom || utc>=g_ftmoCSVTo))){g_ftmoInCriticalNews=true;FTMOExplain("Calendario non disponibile o fuori copertura: ordini EA sospesi; verificare eventuali posizioni gia aperte.");return false;}
 int left=0,right=ArraySize(g_ftmoNewsUTC);
 while(left<right){int mid=(left+right)/2;if(g_ftmoNewsUTC[mid]<utc-InpFTMONewsResumeAfterMin*60)left=mid+1;else right=mid;}
 bool blocked=false;
 for(int i=left;i<ArraySize(g_ftmoNewsUTC);i++){
  datetime t=g_ftmoNewsUTC[i];if(t>utc+InpFTMONewsCloseAheadMin*60)break;
  if(FTMOInsideNewsWindow(utc,t,InpFTMONewsCloseAheadMin,InpFTMONewsResumeAfterMin))blocked=true;
  if(FTMOInsideNewsWindow(utc,t,2,2))g_ftmoInCriticalNews=true;
 }
 if(blocked){
  if(!g_ftmoInCriticalNews)FTMOCloseOwn();
  FTMOExplain(g_ftmoInCriticalNews?"Finestra notizie +/-2 minuti: nessun ordine EA; SL/TP preesistenti restano sotto controllo del server.":"Pausa notizie: chiusura anticipata e ingressi sospesi.");return false;
 }
 return true;
}
void FTMOLoadExtraClosures(){
 if(InpFTMOExtraClosuresServer=="")return;string dates[];int count=StringSplit(InpFTMOExtraClosuresServer,';',dates);
 for(int i=0;i<count;i++){datetime t=StringToTime(dates[i]);if(t<=0){Print("Data chiusura straordinaria non valida: ",dates[i]);continue;}int n=ArraySize(g_ftmoExtraClosures);ArrayResize(g_ftmoExtraClosures,n+1);g_ftmoExtraClosures[n]=t;}
}
int FTMOSessionSecond(datetime t){MqlDateTime d;TimeToStruct(t,d);return d.hour*3600+d.min*60+d.sec;}
bool FTMONextSession(string symbol,datetime after,datetime &next){
 datetime day=StringToTime(TimeToString(after,TIME_DATE));next=0;
 for(int offset=0;offset<=8;offset++){
  datetime base=day+offset*86400;MqlDateTime d;TimeToStruct(base,d);
  for(uint idx=0;idx<30;idx++){datetime a,b;if(!SymbolInfoSessionTrade(symbol,(ENUM_DAY_OF_WEEK)d.day_of_week,idx,a,b))break;
   datetime start=base+FTMOSessionSecond(a);if(start>=after && (next==0 || start<next))next=start;
  }
  if(next>0)return true;
 }
 return false;
}
bool FTMOMarketAllowed(){
 if(FTMOChallenge())return true;datetime now=FTMONow();
 for(int i=0;i<ArraySize(g_ftmoExtraClosures);i++)if(now>=g_ftmoExtraClosures[i]-InpFTMOMarketCloseAheadMin*60 && now<=g_ftmoExtraClosures[i]){FTMOCloseOwn();FTMOExplain("Chiusura straordinaria programmata: conto del bot in chiusura.");return false;}
 datetime day=StringToTime(TimeToString(now,TIME_DATE)),sessionEnd=0;bool known=false;
 // Include sessioni che attraversano la mezzanotte.
 for(int offset=-1;offset<=0;offset++){
  datetime base=day+offset*86400;MqlDateTime d;TimeToStruct(base,d);
  for(uint idx=0;idx<30;idx++){datetime a,b;if(!SymbolInfoSessionTrade(_Symbol,(ENUM_DAY_OF_WEEK)d.day_of_week,idx,a,b))break;
   known=true;int startSec=FTMOSessionSecond(a),endSec=FTMOSessionSecond(b);
   datetime begin=base+startSec,end=base+endSec;if(endSec<=startSec)end+=86400;
   if(now>=begin && now<end)sessionEnd=MathMax(sessionEnd,end);
  }
 }
 if(sessionEnd==0){FTMOExplain(known?"Fuori dalla sessione di negoziazione: ingressi sospesi.":"Sessioni del simbolo non disponibili: funded sospeso.");return false;}
 if(now>=sessionEnd-InpFTMOMarketCloseAheadMin*60){
  datetime next=0;bool found=FTMONextSession(_Symbol,sessionEnd,next);
  bool mustClose=InpFTMOFlatBeforeAllBreaks || !found || next-sessionEnd>7200;
  if(mustClose){FTMOCloseOwn();FTMOExplain("Chiusura anticipata prima della pausa di mercato.");return false;}
 }
 return true;
}
bool FTMOLiveReady(){
 if(MQLInfoInteger(MQL_TESTER))return true;
 if(!TerminalInfoInteger(TERMINAL_CONNECTED)){FTMOExplain("Terminale scollegato: ingressi sospesi.");return false;}
 if(!InpFTMOBrokerClockConfirmed || MathAbs((double)(FTMONow()-TimeGMT())-FTMOServerOffset(FTMONow())*3600)>120){FTMOExplain("Orario/DST del server da verificare: ingressi sospesi.");return false;}
 return true;
}
bool FTMOUpdate(){
 if(!FTMOEnsureReady())return false;
 datetime now=FTMONow();
 if(g_ftmoHistoryDirty || FTMODayKey(now)!=g_ftmoDay || now-g_ftmoLastSync>=60){
  g_ftmoHistoryOK=false;if(!FTMORebuildDay())return false;
 }
 FTMOObserve();
 bool live=FTMOLiveReady();bool news=live?FTMONewsAllowed():false;
 if(!live && !FTMOChallenge())g_ftmoInCriticalNews=true;
 if(g_ftmoFailed || g_ftmoTotalStopped || g_ftmoPassed){FTMOCloseOwn();FTMOExplain(g_ftmoFailed?"Violazione osservata: EA fermo.":g_ftmoPassed?"Obiettivi raggiunti: EA fermo in attesa di FTMO.":"Protezione totale: EA fermo.");return false;}
 if(!FTMOInEvaluation(now)){if(FTMOClock(now)>=g_ftmoStartCET)FTMOCloseOwn();FTMOExplain("Fuori dal periodo di prova selezionato.");return false;}
 double eq=AccountInfoDouble(ACCOUNT_EQUITY);
 if(eq<=FTMOTotalFloor()+InpFTMOBufferEUR){g_ftmoTotalStopped=true;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("STOP"),1);}
 if(eq<=FTMOFloor()+InpFTMOBufferEUR){g_ftmoLockedDay=g_ftmoDay;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("LOCK"),g_ftmoDay);FTMOCloseOwn();FTMOExplain("Buffer di perdita raggiunto: chiusura e blocco del giorno.");return false;}
 FTMOMarkPassed();if(g_ftmoPassed)return false;
 if(!live || !news)return false;
 if(!FTMOMarketAllowed())return false;
 double openRisk=FTMOOpenStopRisk();
 if(openRisk==DBL_MAX){FTMOCloseOwn();FTMOExplain("Rischio degli SL non calcolabile: chiusura prudenziale delle posizioni del bot e ingressi sospesi.");return false;}
 if(openRisk!=DBL_MAX && PositionsTotal()>0 && eq-openRisk-InpFTMOReservePerTradeEUR*PositionsTotal()<=FTMOFloor()+InpFTMOBufferEUR){
  FTMOCloseOwn();FTMOExplain("Rischio effettivo degli SL oltre la capienza residua: chiusura prudenziale delle posizioni del bot.");return false;
 }
 // Include i costi d'ingresso ancora non attribuiti a posizioni chiuse.
 double unsettled=AccountInfoDouble(ACCOUNT_BALANCE)-InpFTMOInitialCapital-g_ftmoNetClosed;
 double projectedToday=g_ftmoTodayClosed+eq-AccountInfoDouble(ACCOUNT_BALANCE)+unsettled;
 double projectedBest=MathMax(g_ftmoBestDay,projectedToday);
 double projectedNet=eq-InpFTMOInitialCapital;
 double requiredNet=MathMax(FTMOTarget()-InpFTMOInitialCapital,2*projectedBest);
 if(FTMOChallenge() && PositionsTotal()>0 && projectedNet>=requiredNet+InpFTMOReservePerTradeEUR*MathMax(1,PositionsTotal())){
  FTMOCloseOwn();FTMOExplain("Chiusura al target con Consistency stimata: verifica sul saldo dopo l'esecuzione.");return false;
 }
 double dayCap=FTMOChallenge()?((FTMOTarget()-InpFTMOInitialCapital)*InpFTMOChallengeDayCapPct/100.0):InpFTMOFundedDayCapEUR;
 if(dayCap>0 && projectedToday>=dayCap){
  FTMOCloseOwn();g_ftmoLockedDay=g_ftmoDay;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("LOCK"),g_ftmoDay);
  FTMOExplain("Tetto prudenziale di profitto giornaliero raggiunto: pausa fino al giorno successivo.");return false;
 }
 if(FTMOPreRollover()){FTMOExplain("Finestra di rollover: nessun nuovo ingresso.");return false;}
 if(g_ftmoLockedDay==g_ftmoDay){FTMOExplain("Giornata bloccata dalle protezioni.");return false;}
 if(FTMOOpenStopRisk()==DBL_MAX || OrdersTotal()>0){FTMOExplain("Posizione senza SL, quotazioni mancanti o ordini pendenti: nuovi ingressi sospesi.");return false;}
 MqlTick tick;if(!SymbolInfoTick(_Symbol,tick) || (!MQLInfoInteger(MQL_TESTER) && now-tick.time>60)){FTMOExplain("Quotazione non aggiornata: nessun nuovo ingresso.");return false;}
 FTMOExplain("Monitoraggio attivo; in attesa di un segnale ORB valido.");return true;
}
void FTMOOnDeal(const MqlTradeTransaction &trans){if(trans.type==TRADE_TRANSACTION_DEAL_ADD){g_ftmoHistoryDirty=true;g_ftmoHistoryOK=false;}}
void OnTimer(){FTMOUpdate();}
void FTMORestorePositions(){
 for(int p=PositionsTotal()-1;p>=0;p--){ulong ticket=PositionGetTicket(p);if(ticket==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=_Symbol)continue;
  long id=PositionGetInteger(POSITION_IDENTIFIER);double entry=PositionGetDouble(POSITION_PRICE_OPEN),stop=PositionGetDouble(POSITION_SL),volume=PositionGetDouble(POSITION_VOLUME);
  int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;string session=PositionGetString(POSITION_COMMENT);int sep=StringFind(session,"|");if(sep>=0)session=StringSubstr(session,0,sep);
  double distance=0,totalVolume=0;
  if(!MQLInfoInteger(MQL_TESTER) && GlobalVariableCheck(FTMOPositionKey(id,"R")))distance=GlobalVariableGet(FTMOPositionKey(id,"R"));
  if(HistorySelectByPosition(id))for(int j=0;j<HistoryDealsTotal();j++){ulong deal=HistoryDealGetTicket(j);if(HistoryDealGetInteger(deal,DEAL_ENTRY)!=DEAL_ENTRY_IN)continue;
   totalVolume+=HistoryDealGetDouble(deal,DEAL_VOLUME);if(distance<=0){double originalSL=HistoryDealGetDouble(deal,DEAL_SL);if(originalSL>0)distance=MathAbs(HistoryDealGetDouble(deal,DEAL_PRICE)-originalSL);}
  }
  // Se il rischio iniziale non e' ricostruibile, mantiene lo SL del server e disabilita la piramide.
  RegisterPosition(id,session,entry,distance,dir);int idx=FindSessionMapIndex(id);
  if(idx>=0){g_posSessionMap[idx].beDone=stop>0 && (dir==1?stop>=entry:stop<=entry);g_posSessionMap[idx].partialDone=totalVolume>volume+0.0000001;
   g_posSessionMap[idx].pyramidDone=distance<=0 || session=="PYRAMID" || (!MQLInfoInteger(MQL_TESTER) && GlobalVariableCheck(FTMOPositionKey(id,"PYR")));}
 }
}
void FTMOExportDaily(){
 string name=MQLInfoString(MQL_PROGRAM_NAME)+"_FTMO_"+IntegerToString(g_ftmoMonth)+"_daily.csv";
 int h=FileOpen(name,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE)return;
 FileWrite(h,"day","midnight_balance","minimum_equity_observed","daily_loss","total_floor","daily_floor","best_closed_day","net_closed_profit","consistency_pct");
 for(int i=0;i<ArraySize(g_auditKeys);i++)FileWrite(h,g_auditKeys[i],DoubleToString(g_auditMidnight[i],2),DoubleToString(g_auditMinimum[i],2),DoubleToString(MathMax(0,g_auditMidnight[i]-g_auditMinimum[i]),2),DoubleToString(g_auditTotalFloor[i],2),DoubleToString(g_auditDailyFloor[i],2),DoubleToString(g_auditBestDay[i],2),DoubleToString(g_auditNetClosed[i],2),g_auditNetClosed[i]>0?DoubleToString(100*g_auditBestDay[i]/g_auditNetClosed[i],2):"");
 FileClose(h);
}
void FTMOExport(){
 if(!g_ftmoReady)FTMOEnsureReady();if(g_ftmoHistoryDirty)FTMORebuildDay();FTMOObserve();FTMOMarkPassed();FTMOExportDaily();
 // Non confronta un minimo storico con la soglia FINALE: il trailing cambia nel tempo.
 string status=!g_ftmoReady || !g_ftmoHistoryOK?"storico_non_valido":g_ftmoFailed?"violazione_osservata":g_ftmoPassed?"obiettivi_calcolati_raggiunti":g_ftmoTotalStopped?"arresto_protezione_totale":FTMOChallenge()?"challenge_non_completata":"funded_continuo";
 int calendarDays=g_ftmoPassed?1+(int)((FTMODayStart(FTMOClock(g_ftmoPassedServer))-g_ftmoElapsedStartCET)/86400):0;
 string name=MQLInfoString(MQL_PROGRAM_NAME)+"_FTMO_"+IntegerToString(g_ftmoMonth)+"_result.csv";
 int h=FileOpen(name,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE)return;
 FileWrite(h,"phase","status","balance","equity","min_balance_mt5","min_equity_mt5","max_daily_loss_observed","trading_days","calendar_days_to_pass","passed_server","failure_reason","target_percent","daily_limit_eur","trailing_floor","highest_midnight_balance","buffer_eur","net_closed_profit","best_day","consistency_pct","last_diagnostic");
 FileWrite(h,EnumToString(InpFTMOPhase),status,DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2),DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2),DoubleToString(TesterStatistics(STAT_BALANCEMIN),2),DoubleToString(TesterStatistics(STAT_EQUITYMIN),2),DoubleToString(g_ftmoWorstDaily,2),ArraySize(g_ftmoDays),calendarDays,g_ftmoPassed?TimeToString(g_ftmoPassedServer,TIME_DATE|TIME_SECONDS):"",g_ftmoFailureReason,FTMOChallenge()?InpFTMOTargetPercent:0,InpFTMODailyLossEUR,DoubleToString(FTMOTotalFloor(),2),DoubleToString(g_ftmoHighestMidnight,2),InpFTMOBufferEUR,DoubleToString(g_ftmoNetClosed,2),DoubleToString(g_ftmoBestDay,2),DoubleToString(FTMOConsistencyPercent(),2),g_ftmoReason);
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

                bool signalAfterRange = iTime(_Symbol,InpEntryTF,1)>=lonEnd;
                bool bull = signalAfterRange && (m5_close > g_lonHigh);
                bool bear = signalAfterRange && (m5_close < g_lonLow);

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

                bool signalAfterRange = iTime(_Symbol,InpEntryTF,1)>=nyEnd;
                bool bull = signalAfterRange && (m5_close > g_nyHigh);
                bool bear = signalAfterRange && (m5_close < g_nyLow);

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

                    bool signalAfterRange = iTime(_Symbol,InpEntryTF,1)>=asiaEnd;
                bool bull = signalAfterRange && (m5_close > g_asiaHigh);
                    bool bear = signalAfterRange && (m5_close < g_asiaLow);

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
    if(!FTMOUpdate() || !FTMODailyAllows(dir))return;
    if((dir==1 && !InpAllowLong) || (dir==-1 && !InpAllowShort))return;
    if(ResearchOwnPositionCount()>=InpMaximumPositions) return;
    double availableStopRisk=AccountInfoDouble(ACCOUNT_EQUITY)*InpTotalOpenStopRiskPct/100.0-FTMOOpenStopRisk();
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

    if(stepLot<=0 || minLot<=0)return;
    double lots = MathFloor((riskAmount / (MathAbs(profitPerLot)+InpFTMOExtraCostPerLotEUR)) / stepLot) * stepLot;
    // Nessun arrotondamento verso l'alto che superi il rischio scelto.
    if(lots < minLot) return;
    lots=MathMin(lots,MathFloor(SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX)/stepLot)*stepLot);
    double margin=0;
    if(!OrderCalcMargin(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,lots,entry,margin) || margin<=0) return;
    double availableMargin=AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.80;
    if(margin>availableMargin) lots=MathFloor(lots*availableMargin/margin/stepLot)*stepLot;
    if(lots<minLot) return;

    string cmt = StringFormat("%s|185|R:%.2f", session, riskAmount);
    // Ricontrollo sul prezzo corrente prima dell'invio.
    double lossAtStop=0;
    if(!OrderCalcProfit(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,lots,entry,sl_final,lossAtStop) ||
       -lossAtStop+lots*InpFTMOExtraCostPerLotEUR>FTMOEntryRisk(riskAmount)+0.01)return;
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

        g_ftmoHistoryDirty=true;
        FTMOPersistPosition(posId,MathAbs(filledEntry-sl_final));
        FTMOUpdate();
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
