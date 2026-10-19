// prova990: NAS10030k DD10%: prova881, conferma candela ed estensione ORB {'InpNasMaxBreakoutExtension': 0.4, 'InpNasQualityStrongOnly': True, 'InpNasQualityCandleDirection': True, 'InpRiskPercent': 0.575, 'InpTotalOpenStopRiskPct': 0.575}; range e trend invariati, periodo2025-oggi riutilizzato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova881: NAS10030k DD10%: prova869 congelata, fino11/10/2026, tick reali disponibili; fallback broker dichiarato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova869: NAS10030k DD10%: base843 con filtroATR proporzionato al range, ORB NY16:30, durata15min, ora server; periodo aggiornato riutilizzato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova843: NAS10030k DD10%: prova833 congelata, fino11/10/2026, tick reali disponibili; fallback broker dichiarato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova833: NAS10030k DD10%: base824, target6R, durata240min, rientriFalse, rischio0.575%; periodo aggiornato riutilizzato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova824: NAS10030k DD10%: prova806 congelata, fino11/10/2026, tick reali disponibili; fallback broker dichiarato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova806: NAS10030k DD10%: trailing diprova792, rischio0.575%, confronto rischio/frequenza fino11/10/2026; storico riutilizzato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova792: NAS10030k DD10%: base755, stop profitto a scatti, rischio0.75%, TP3R, periodo aggiornato; storico riutilizzato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova755: NAS10030k DD10%: pesi sessione/direzione da745; rischio1.25%, tutte le direzioni, periodo fino11/10/2026; storico riutilizzato.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova745: NAS10030k DD10%: prova718 congelata, da01/01/2025 fino11/10/2026 incluso, Model4 tick reali disponibili; fine MT5 esclusiva12/10.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova718: NAS10030k DD10%: prova648 congelata rischio1%, modello4 completo, cache locale gia scaricata; fallback broker dichiarato, niente attivazione.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova648: NAS100 personale obiettivo30k: base642 rischio1%, Asia90 NY60min; periodo riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova642: NAS100 personale: prova640 congelata, periodo completo modello tick reali, dati riutilizzati e fallback broker da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova640: NAS100 personale: prova633 rischio1.52%, screening sul periodo riutilizzato, DD da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova633: NAS100 personale: base611 rischio1%, NY17:30 in tempo server; periodo riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova611: NAS100 personale: prova608 congelata, periodo completo modello tick reali, dati riutilizzati e fallback broker da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova608: NAS100 personale: prova605 rischio1.4%, selezione profitto/DD, periodo riutilizzato OHLC M1; DD da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova605: NAS100 personale: base583 rischio1%, rischio Asia1.5 e NY0.5; intero periodo riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova583: NAS100 personale: prova576 congelata dopo screening, periodo completo modello tick reali; dati riutilizzati, copertura da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova576: NAS100 personale: base571, rischio1.5% senza piramide; periodo intero riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova571: NAS100 personale: prova564 congelata dopo screening, intero periodo modello tick reali; dati riutilizzati, copertura da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova564: NAS100 personale: base543 rischio1%, corpo candela minimo50%; periodo intero riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova543: NAS100 personale: parametri542 invariati, trend H1 rischio1%, intero periodo modello tick reali; copertura da verificare.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova542: NAS100 personale: base538 trend H1, rischio1% al posto0.75%; intero periodo riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova538: NAS100 personale: base533, trendEMA20/100 H1 chiusa, stop originale; intero periodo riutilizzato OHLC M1.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova533: NAS100 personale: tre sessioni entrambe direzioni 2R, rischio0.75%, BE1R, max4h e chiusura23:45, corpo25%, rangeATR0.10/0.15.
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
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
input int      InpLon_StartHour     = 10;
input int      InpLon_StartMin      = 0;
input int      InpLon_DurationMin   = 60;
input bool     InpLon_UseDailyFilter= false;
input bool     InpLon_IgnoreBias    = true;

input group "New York Session (ORB)"
input int      InpNY_StartHour      = 16;
input int      InpNY_StartMin       = 30;
input int      InpNY_DurationMin    = 15;
input bool     InpNY_UseDailyFilter = false;

input group "Asian Session (ORB)"
input bool     InpUseAsia            = true;
input int      InpAsia_StartHour     = 3;
input int      InpAsia_StartMin      = 0;
input int      InpAsia_DurationMin   = 90;
input bool     InpAsia_UseDailyFilter= false;

input group "Sniper Filters (Crucial)"
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M5;
input int      InpMaxEntryDelayMin  = 360;
input double   InpBodyFilterPct     = 50;

input group "Fade dei Falsi Breakout (NON si applica a LONDON)"
input bool     InpFadeEnabled       = false;
input double   InpFadeRiskPct       = 50.0;
input double   InpFadeSLBufferPoints= 30;

input double InpATRMinRangeMultAsia = 0.1;
input double InpATRMinRangeMultNY = 0.0375;

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;
input double   InpATRMinRangeMult   = 0.15;

input group "Break-Even & Partial Close"
input bool     InpUseBreakEven      = false;
input double   InpBreakEvenTriggerR = 1.0;
input double   InpBreakEvenLockR    = 0.1;
input bool     InpUsePartialClose   = false;
input double   InpPartialClosePct   = 50.0;
input double   InpPartialCloseAtR   = 1.0;

input group "Risk & Money Management"
input double   InpFixedRiskAmount   = 80.0;
input bool     InpUseEquityRisk     = true;
input double   InpRiskPercent       = 0.575;
input double   InpRR_Target         = 6;
input int      InpMaxLossesDay      = 6;
input int      InpMaxTradesDay      = 9;
input ulong    InpMagicNumber       = 20261010990;
input bool     InpShowComments      = false;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = false;
input double   InpMaxDailyLossPct    = 3.0;

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ══════════════════════════════════════════
input double InpPyramidAtR=1.2;
input double InpPyramidRiskPercent=0;
input double InpPyramidExtraTargetR=1.0;
input double InpTotalOpenStopRiskPct=0.575;
double g_profitPyramid=0; int g_winsPyramid=0,g_totalPyramid=0;
input double InpAsiaRiskMultiplier=1.5;
input double InpNYRiskMultiplier=0.5;
input double InpLondonRiskMultiplier=1;
input group "NAS100: sessioni del broker e spread"
input bool InpTradeWeekend=false;
input double InpMaxSpreadToStopPct=10.0;
input bool InpRequireWeekendSessions=false;
bool g_lonLoaded=false,g_nyLoaded=false,g_asiaLoaded=false;
input group "NAS100: selezione della sessione"
input bool InpNasOnlyNY=false;
input bool InpNasCloseEndOfDay=true;
input int InpNasCloseHour=23;
input int InpNasCloseMinute=45;
input int InpNasMaxHoldMinutes=240;
input bool InpNasAllowLong=true;
input bool InpNasAllowShort=true;
input group "NAS100: trend su H1 chiusa e distanza stop"
input bool InpNasTrendFilter=true;
input int InpNasTrendFast=20;
input int InpNasTrendSlow=100;
input double InpNasStopFraction=1;
int g_nasTrendFast=INVALID_HANDLE,g_nasTrendSlow=INVALID_HANDLE;
input group "NAS100: rientri con nuova rottura del range"
input bool InpNasReentry=false;
input int InpNasMaxSessionEntries=2;
int g_nasLonEntries=0,g_nasNYEntries=0,g_nasAsiaEntries=0;
datetime g_nasEntryDay=0,g_nasRearmBar=0,g_nasLonFillBar=0,g_nasNYFillBar=0,g_nasAsiaFillBar=0;
input group "NAS100: rischio adattivo dal massimo di equity"
input bool InpNasAdaptiveRisk=false;
input double InpNasAdaptiveSoftDDPct=2;
input double InpNasAdaptiveHardDDPct=9.2;
input double InpNasAdaptiveCurve=2;
input double InpNasAdaptiveMinFraction=0.10;
input double InpNasAdaptiveRoomFraction=0.3;
double g_nasAdaptivePeak=0;
bool g_nasAdaptiveHalted=false;
// Il massimo si aggiorna con i tick gia osservati, senza reset periodici.
void NasAdaptiveObserve() {
    if(!InpNasAdaptiveRisk) return;
    double eq=AccountInfoDouble(ACCOUNT_EQUITY);
    g_nasAdaptivePeak=MathMax(g_nasAdaptivePeak,eq);
    if(g_nasAdaptivePeak<=0) return;
    double floorEquity=g_nasAdaptivePeak*(1-InpNasAdaptiveHardDDPct/100.0);
    if(eq<=floorEquity) g_nasAdaptiveHalted=true;
}
double NasAdaptiveBudget(double requested) {
    if(!InpNasAdaptiveRisk) return requested;
    if(g_nasAdaptiveHalted || g_nasAdaptivePeak<=0) return 0;
    double eq=AccountInfoDouble(ACCOUNT_EQUITY);
    double dd=MathMax(0,(g_nasAdaptivePeak-eq)/g_nasAdaptivePeak*100.0);
    double progress=MathMax(0,MathMin(1,(dd-InpNasAdaptiveSoftDDPct)/(InpNasAdaptiveHardDDPct-InpNasAdaptiveSoftDDPct)));
    double scale=InpNasAdaptiveMinFraction+(1-InpNasAdaptiveMinFraction)*MathPow(1-progress,InpNasAdaptiveCurve);
    double room=MathMax(0,eq-g_nasAdaptivePeak*(1-InpNasAdaptiveHardDDPct/100.0));
    return MathMin(requested*scale,room*InpNasAdaptiveRoomFraction);
}
input group "NAS100: distanza dal bordo ORB"
input double InpNasMaxBreakoutExtension=0.4;
input bool InpNasQualityStrongOnly=true;
input bool InpNasQualityCandleDirection=true;
input group "NAS100: pesi per sessione e direzione"
input double InpNasAsiaLongRisk=3;
input double InpNasAsiaShortRisk=0.35;
input double InpNasLondonLongRisk=0.35;
input double InpNasLondonShortRisk=3;
input double InpNasNYLongRisk=1.5;
input double InpNasNYShortRisk=0.35;
// Pesi costanti congelati prima del test. Nessun esito futuro usato all'ingresso.
double NasDirectionRisk(string session,int dir) {
    if(StringFind(session,"ASIA")==0) return dir==1?InpNasAsiaLongRisk:InpNasAsiaShortRisk;
    if(session=="LONDON") return dir==1?InpNasLondonLongRisk:InpNasLondonShortRisk;
    if(StringFind(session,"NY")==0) return dir==1?InpNasNYLongRisk:InpNasNYShortRisk;
    return 1;
}
input group "NAS100: stop a scatti in R"
input double InpNasTrailStartR=1;
input double InpNasTrailGapR=0.5;
input double InpNasTrailStepR=0.25;
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
    if(InpNasMaxBreakoutExtension<=0 || InpNasMaxBreakoutExtension>2) return INIT_PARAMETERS_INCORRECT;

    if(!MQLInfoInteger(MQL_TESTER)) {
        Print("NAS_RESEARCH_ONLY: questa variante si avvia solo nel tester.");
        return INIT_FAILED;
    }
    if(_Symbol!="NAS100") {Print("NAS_RESEARCH_ONLY: simbolo richiesto NAS100");return INIT_PARAMETERS_INCORRECT;}

    if(InpRequireWeekendSessions) {
        datetime from,to;
        if(!SymbolInfoSessionTrade(_Symbol,SATURDAY,0,from,to) || !SymbolInfoSessionTrade(_Symbol,SUNDAY,0,from,to)) {
            Print("CRYPTO_INIT_FAILED: il simbolo non ha sessioni sia sabato sia domenica: ",_Symbol);
            return INIT_PARAMETERS_INCORRECT;
        }
    }
    if(InpNasAdaptiveRisk && (InpNasAdaptiveSoftDDPct<0 || InpNasAdaptiveHardDDPct<=InpNasAdaptiveSoftDDPct || InpNasAdaptiveHardDDPct>=10 || InpNasAdaptiveCurve<=0 || InpNasAdaptiveMinFraction<=0 || InpNasAdaptiveMinFraction>1 || InpNasAdaptiveRoomFraction<=0 || InpNasAdaptiveRoomFraction>=1)) return INIT_PARAMETERS_INCORRECT;
    g_nasAdaptivePeak=AccountInfoDouble(ACCOUNT_EQUITY);
    if(InpNasAsiaLongRisk<=0 || InpNasAsiaShortRisk<=0 || InpNasLondonLongRisk<=0 || InpNasLondonShortRisk<=0 || InpNasNYLongRisk<=0 || InpNasNYShortRisk<=0) return INIT_PARAMETERS_INCORRECT;
    if(InpNasTrailStartR<=0 || InpNasTrailGapR<=0 || InpNasTrailGapR>InpNasTrailStartR || InpNasTrailStepR<=0) return INIT_PARAMETERS_INCORRECT;
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFillingBySymbol(_Symbol);

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
    if(InpNasStopFraction<=0 || InpNasStopFraction>1 || InpNasTrendFast<1 || InpNasTrendSlow<=InpNasTrendFast) return INIT_PARAMETERS_INCORRECT;
    if(InpNasTrendFilter) {
        g_nasTrendFast=iMA(_Symbol,PERIOD_H1,InpNasTrendFast,0,MODE_EMA,PRICE_CLOSE);
        g_nasTrendSlow=iMA(_Symbol,PERIOD_H1,InpNasTrendSlow,0,MODE_EMA,PRICE_CLOSE);
        if(g_nasTrendFast==INVALID_HANDLE || g_nasTrendSlow==INVALID_HANDLE) return INIT_FAILED;
    }
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    Print("NAS_ADAPTIVE_FINAL peak=",DoubleToString(g_nasAdaptivePeak,2)," halted=",g_nasAdaptiveHalted);
    if(g_nasTrendFast!=INVALID_HANDLE) IndicatorRelease(g_nasTrendFast);
    if(g_nasTrendSlow!=INVALID_HANDLE) IndicatorRelease(g_nasTrendSlow);

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
    MqlDateTime now;TimeToStruct(TimeCurrent(),now);
    bool closeDay=InpNasCloseEndOfDay && now.hour*60+now.min>=InpNasCloseHour*60+InpNasCloseMinute;

    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        datetime opened=(datetime)PositionGetInteger(POSITION_TIME);
        if(closeDay || (InpNasMaxHoldMinutes>0 && TimeCurrent()-opened>=InpNasMaxHoldMinutes*60)) {
            if(!trade.PositionClose(ticket) || trade.ResultRetcode()!=TRADE_RETCODE_DONE)
                Print("NAS_TIMED_CLOSE_RETRY: ",ticket," retcode ",trade.ResultRetcode());
            continue;
        }
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

        // Distanza di rischio originale: non cambia dopo ogni modifica dello stop.
        if(rMultiple>=InpNasTrailStartR) {
            double lockR=InpNasTrailStartR-InpNasTrailGapR+MathFloor((rMultiple-InpNasTrailStartR)/InpNasTrailStepR)*InpNasTrailStepR;
            double wanted=entry+dir*lockR*slDist;
            double grid=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
            MqlTick live;
            if(grid>0 && SymbolInfoTick(_Symbol,live)) {
                wanted=NormalizeDouble((dir==1?MathFloor(wanted/grid):MathCeil(wanted/grid))*grid,digits);
                double minGap=MathMax(SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL))*SymbolInfoDouble(_Symbol,SYMBOL_POINT);
                bool improves=dir==1?wanted>currentSL+grid/2:(currentSL==0 || wanted<currentSL-grid/2);
                bool allowed=dir==1?wanted<live.bid-minGap:wanted>live.ask+minGap;
                if(improves && allowed && trade.PositionModify(ticket,wanted,currentTP) && trade.ResultRetcode()==TRADE_RETCODE_DONE) {
                    currentSL=wanted;
                    Print("NAS_TRAIL_UPDATE position=",posId," R=",DoubleToString(rMultiple,4)," lockR=",DoubleToString(lockR,4)," stop=",DoubleToString(wanted,digits));
                }
            }
        }
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
            double grid=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
            if(grid<=0) continue;
            beSL=NormalizeDouble((dir==1?MathFloor(beSL/grid):MathCeil(beSL/grid))*grid,digits);
            MqlTick live; if(!SymbolInfoTick(_Symbol,live)) continue;
            double minGap=MathMax(SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL))*SymbolInfoDouble(_Symbol,SYMBOL_POINT);
            if((dir==1 && beSL>=live.bid-minGap) || (dir==-1 && beSL<=live.ask+minGap)) continue;
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

// Range esatto delle M1 concluse: non include la candela precedente alla finestra.
void LoadCryptoORB(datetime start,int minutes,bool &loaded,bool &rangeDefined,double &high,double &low) {
    if(loaded || minutes<=0 || TimeCurrent()<start+minutes*60) return;
    MqlRates bars[];
    int n=CopyRates(_Symbol,PERIOD_M1,start,start+minutes*60-1,bars);
    if(n<=0) return;
    double h=-DBL_MAX,l=DBL_MAX;
    for(int i=0;i<n;i++){h=MathMax(h,bars[i].high);l=MathMin(l,bars[i].low);}
    if(h<=l || l<=0) return;
    high=h;low=l;rangeDefined=true;loaded=true;
}

void NasResetEntryCounters() {
    datetime now=TimeCurrent(),day=now-now%86400;
    if(day!=g_nasEntryDay) {
        g_nasEntryDay=day;g_nasLonEntries=0;g_nasNYEntries=0;g_nasAsiaEntries=0;
        g_nasRearmBar=0;g_nasLonFillBar=0;g_nasNYFillBar=0;g_nasAsiaFillBar=0;
    }
}
void NasRearmBreakouts() {
    NasResetEntryCounters();
    if(!InpNasReentry || HasOpenPosition()) return;
    datetime bar=iTime(_Symbol,InpEntryTF,1);
    if(bar<=0 || bar==g_nasRearmBar) return;
    g_nasRearmBar=bar;
    double closed=iClose(_Symbol,InpEntryTF,1);
    if(g_nasLonEntries>0 && g_nasLonEntries<InpNasMaxSessionEntries && bar>g_nasLonFillBar && closed>=g_lonLow && closed<=g_lonHigh) g_lonExecutedToday=false;
    if(g_nasNYEntries>0 && g_nasNYEntries<InpNasMaxSessionEntries && bar>g_nasNYFillBar && closed>=g_nyLow && closed<=g_nyHigh) g_nyExecutedToday=false;
    if(g_nasAsiaEntries>0 && g_nasAsiaEntries<InpNasMaxSessionEntries && bar>g_nasAsiaFillBar && closed>=g_asiaLow && closed<=g_asiaHigh) g_asiaExecutedToday=false;
}

void OnTick() {
    NasAdaptiveObserve();
    if(g_nasAdaptiveHalted) {
        // Soglia sperimentale: esecuzioni e gap sono verificati nel report nativo.
        for(int i=PositionsTotal()-1;i>=0;i--) {
            ulong ticket=PositionGetTicket(i);
            if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
            if(!trade.PositionClose(ticket) || trade.ResultRetcode()!=TRADE_RETCODE_DONE)
                Print("NAS_ADAPTIVE_CLOSE_FAILED ",ticket," retcode=",trade.ResultRetcode());
        }
        return;
    }
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

    if(!InpTradeWeekend && (dt.day_of_week==0 || dt.day_of_week==6)) return;
    LoadCryptoORB(today+InpLon_StartHour*3600+InpLon_StartMin*60,InpLon_DurationMin,g_lonLoaded,g_lonRangeDefined,g_lonHigh,g_lonLow);
    LoadCryptoORB(today+InpNY_StartHour*3600+InpNY_StartMin*60,InpNY_DurationMin,g_nyLoaded,g_nyRangeDefined,g_nyHigh,g_nyLow);
    if(InpUseAsia) LoadCryptoORB(today+InpAsia_StartHour*3600+InpAsia_StartMin*60,InpAsia_DurationMin,g_asiaLoaded,g_asiaRangeDefined,g_asiaHigh,g_asiaLow);
    ManagePyramids();
    NasRearmBreakouts();
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
                    if(HasOpenPosition()) g_lonExecutedToday = true;
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
        Comment(StringFormat("Tri-Session v5.5 (Lon no-fade) | Bias: %s | Lon H/L: %.2f/%.2f | NY H/L: %.2f/%.2f | Asia H/L: %.2f/%.2f | Trades: %d",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")),
              g_lonHigh, g_lonLow, g_nyHigh, g_nyLow, g_asiaHigh, g_asiaLow, g_tradesToday));
    }
}

// ══════════════════════════════════════════
// 💼 EXECUTION
// ══════════════════════════════════════════
void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
    NasResetEntryCounters();
    if(InpNasReentry && ((session=="LONDON" && g_nasLonEntries>=InpNasMaxSessionEntries) || (session=="NY" && g_nasNYEntries>=InpNasMaxSessionEntries) || (session=="ASIA" && g_nasAsiaEntries>=InpNasMaxSessionEntries))) return;

    MqlDateTime now;TimeToStruct(TimeCurrent(),now);
    if(InpNasCloseEndOfDay && now.hour*60+now.min>=InpNasCloseHour*60+InpNasCloseMinute) return;

    if((dir==1 && !InpNasAllowLong) || (dir==-1 && !InpNasAllowShort)) return;
    // Nelle prove solo NY la piramide e disabilitata nei parametri.
    if(InpNasOnlyNY && session!="NY" && session!="NY_FADE" && session!="NY_REVERSAL") return;

    if(StringFind(session,"ASIA")==0) riskAmount*=InpAsiaRiskMultiplier;
    else if(StringFind(session,"NY")==0) riskAmount*=InpNYRiskMultiplier;
    else if(session=="LONDON") riskAmount*=InpLondonRiskMultiplier;
    riskAmount*=NasDirectionRisk(session,dir);
    riskAmount=NasAdaptiveBudget(riskAmount);
    if(riskAmount<=0) return;

    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // Rischio e target calcolati sul prezzo negoziabile corrente.
    MqlTick quote;
    if(!SymbolInfoTick(_Symbol,quote)) return;
    entry=dir==1?quote.ask:quote.bid;
    bool qualityApplied=!InpNasQualityStrongOnly || (session=="ASIA" && dir==1) || (session=="LONDON" && dir==-1);
    double qualityHigh=0,qualityLow=0,qualityExtension=0,qualityClose=0,qualityOpen=0;
    if(qualityApplied) {
        if(session=="ASIA") {qualityHigh=g_asiaHigh;qualityLow=g_asiaLow;}
        else if(session=="LONDON") {qualityHigh=g_lonHigh;qualityLow=g_lonLow;}
        else if(session=="NY") {qualityHigh=g_nyHigh;qualityLow=g_nyLow;}
        else return;
        if(qualityHigh<=qualityLow || qualityLow<=0) return;
        qualityClose=iClose(_Symbol,InpEntryTF,1);qualityOpen=iOpen(_Symbol,InpEntryTF,1);
        if(qualityClose<=0 || qualityOpen<=0) return;
        if(InpNasQualityCandleDirection && ((dir==1 && qualityClose<=qualityOpen) || (dir==-1 && qualityClose>=qualityOpen))) return;
        double edge=dir==1?qualityHigh:qualityLow;
        qualityExtension=dir*(entry-edge)/(qualityHigh-qualityLow);
        if(qualityExtension<0 || qualityExtension>InpNasMaxBreakoutExtension) return;
    }

    if(InpNasTrendFilter) {
        double fast[1],slow[1];
        if(CopyBuffer(g_nasTrendFast,0,1,1,fast)!=1 || CopyBuffer(g_nasTrendSlow,0,1,1,slow)!=1) return;
        double closedH1=iClose(_Symbol,PERIOD_H1,1);
        if(closedH1<=0 || fast[0]<=0 || slow[0]<=0) return;
        if(dir==1 && !(fast[0]>slow[0] && closedH1>fast[0])) return;
        if(dir==-1 && !(fast[0]<slow[0] && closedH1<fast[0])) return;
    }
    sl_ref=entry+(sl_ref-entry)*InpNasStopFraction;

    double tickSize=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
    if(tickSize<=0 || riskAmount<=0 || entry<=0) return;
    double sl_final=NormalizeDouble(MathRound(sl_ref/tickSize)*tickSize,digits);
    double minStop=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
    if((dir==1 && (sl_final>=quote.bid-minStop)) || (dir==-1 && sl_final<=quote.ask+minStop)) return;

    double stopDistance=MathAbs(entry-sl_final);
    if(InpMaxSpreadToStopPct>0 && quote.ask-quote.bid>stopDistance*InpMaxSpreadToStopPct/100.0) return;
    double tp_final = 0;
    if(customTP > 0) {
        tp_final = NormalizeDouble(customTP, digits);
    } else {
        double dist = MathAbs(entry - sl_final);
        if(dist == 0) return;
        tp_final = NormalizeDouble((dir == 1) ? entry + (dist * InpRR_Target) : entry - (dist * InpRR_Target), digits);
    }

    tp_final=NormalizeDouble(MathRound(tp_final/tickSize)*tickSize,digits);
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

    if(stepLot<=0) return;
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
        datetime fillBar=iTime(_Symbol,InpEntryTF,0);
        if(session=="LONDON") {g_nasLonEntries++;g_nasLonFillBar=fillBar;}
        else if(session=="NY") {g_nasNYEntries++;g_nasNYFillBar=fillBar;}
        else if(session=="ASIA") {g_nasAsiaEntries++;g_nasAsiaFillBar=fillBar;}


        long posId=0; double filledEntry=entry;
        ulong filledDeal=trade.ResultDeal();
        if(filledDeal>0 && HistoryDealSelect(filledDeal)) {
            posId=HistoryDealGetInteger(filledDeal,DEAL_POSITION_ID);
            filledEntry=HistoryDealGetDouble(filledDeal,DEAL_PRICE);
        }
        if(posId > 0) RegisterPosition(posId, session, filledEntry, MathAbs(filledEntry - sl_final), dir);
        if(qualityApplied) Print("NAS_QUALITY_ACCEPT position=",posId," session=",session," side=",dir,
          " high=",DoubleToString(qualityHigh,8)," low=",DoubleToString(qualityLow,8),
          " quoteEntry=",DoubleToString(entry,8)," extension=",DoubleToString(qualityExtension,12),
          " max=",DoubleToString(InpNasMaxBreakoutExtension,12),
          " closed=",DoubleToString(qualityClose,8)," candleOpen=",DoubleToString(qualityOpen,8),
          " directionRequired=",InpNasQualityCandleDirection);


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
    Print("NAS_EXACT_EQUITY_DD_PCT=",DoubleToString(TesterStatistics(STAT_EQUITY_DDREL_PERCENT),12));
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
    string filename=MQLInfoString(MQL_PROGRAM_NAME)+"_trades.csv";
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
