// prova370: MR_MACRO_PULLBACK; M30rientro2sigma/RSI35/65,macroH4EMA12/36pendenza3barre,2R,risk0.75%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova367: MR_TIMEFRAME; M30,rientrointrendEMA20/60 H1,ADX nonlimitato,target2R,risk0.75%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova293: MEAN_REVERT su barreM5 concluse; bande2sigma,RSI35/65,TPmedia,min0.5R,nessunfiltroH1
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova266: Controllo su tick reali della264, rischio weekend dimezzato e parametri invariati
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova264: ORB3 asimmetrica: rischio weekend dimezzato, max1 posizione
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova220: ORB3 sessioni asimmetrica, rischio1.25%, long dimezzato
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova209: Rischio long dimezzato, rischio short invariato
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova198: Maggiore frequenza: range minimo 0.05 ATR, rischio 0.75%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova188: ORB continuazione, uscite 3R, senza BE e senza piramide
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova186: Prova185 con break-even allineato alla griglia tick e verifica distanza minima del contratto crypto
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova185: Adattamento iniziale di prova130 alle crypto, sette giorni su sette
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
input double   InpBodyFilterPct     = 50.0;

input group "Fade dei Falsi Breakout (NON si applica a LONDON)"
input bool     InpFadeEnabled       = false;
input double   InpFadeRiskPct       = 50.0;
input double   InpFadeSLBufferPoints= 30;

input double InpATRMinRangeMultAsia = 0.05;
input double InpATRMinRangeMultNY = 0.05;

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;
input double   InpATRMinRangeMult   = 0.05;

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
input double   InpRiskPercent       = 0.75;
input double   InpRR_Target         = 2.0;
input int      InpMaxLossesDay      = 6;
input int      InpMaxTradesDay      = 12;
input ulong    InpMagicNumber       = 20261010370;
input bool     InpShowComments      = false;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = true;
input double   InpMaxDailyLossPct    = 2.0;

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ══════════════════════════════════════════
input double InpPyramidAtR=1.2;
input double InpPyramidRiskPercent=0.0;
input double InpPyramidExtraTargetR=1.0;
input double InpTotalOpenStopRiskPct=2.0;
double g_profitPyramid=0; int g_winsPyramid=0,g_totalPyramid=0;
input double InpAsiaRiskMultiplier=1.05;
input double InpNYRiskMultiplier=0.8;
input double InpLondonRiskMultiplier=1.1;
input group "Crypto: operativita e costo dello spread"
input bool InpTradeWeekend=true;
input double InpMaxSpreadToStopPct=10.0;
input bool InpRequireWeekendSessions=true;
bool g_lonLoaded=false,g_nyLoaded=false,g_asiaLoaded=false;
input group "Crypto: durata, direzione e rischio aggregato"
input double InpCryptoMaxHoldHours=0.0;
input bool InpCryptoAllowLong=true;
input bool InpCryptoAllowShort=true;
input double InpCryptoLongRiskMult=1.0;
input double InpCryptoShortRiskMult=1.0;
input double InpCryptoMaxStopATR=0.0;
input int InpCryptoMaxPositions=1;
input bool InpCryptoInitializeDailyBias=false;
input double InpCryptoWeekendRiskMult=1.0;
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

input group "Crypto: rientro dopo eccesso M5"
input int InpMRBandPeriod=20;
input double InpMRBandDeviation=2.0;
input int InpMRRSIPeriod=14;
input double InpMRRSILow=35.0;
input double InpMRRSIHigh=65.0;
input double InpMRStopATR=1.25;
input double InpMRStopBufferATR=0.2;
input double InpMRMaxStopATR=4.0;
input double InpMRMinBodyATR=0.0;
input double InpMRMaxGapATR=0.5;
input double InpMRMinRewardR=1.5;
input int InpMRTargetMode=1; //0: media delle bande;1: RR fisso
input int InpMRRegime=2; //0: nessuno;1: ADX basso H1;2: trendEMA H1
input double InpMRADXMax=0.0;
int g_mrBands=INVALID_HANDLE,g_mrATR=INVALID_HANDLE,g_mrRSI=INVALID_HANDLE;
int g_mrADX=INVALID_HANDLE,g_mrFast=INVALID_HANDLE,g_mrSlow=INVALID_HANDLE;
datetime g_mrSignalBar=0,g_mrRegimeBar=0;
bool g_mrLong=false,g_mrShort=false;
double g_mrProfit=0;int g_mrTrades=0,g_mrWins=0;

bool CryptoMRRegimeAllows(int direction){
 if(InpMRRegime==0)return true;
 datetime bar=iTime(_Symbol,InpMRMacroTF,0);
 if(bar<=0)return false;
 if(bar!=g_mrRegimeBar){
  double adx[];
  if(CopyBuffer(g_mrADX,0,1,1,adx)!=1)return false;
  bool allowed=InpMRADXMax<=0 || adx[0]<=InpMRADXMax;
  bool buy=allowed,sell=allowed;
  if(InpMRRegime==2){
   double fast[],slow[];int nf=InpMRMacroSlopeBars+1;
   if(CopyBuffer(g_mrFast,0,1,nf,fast)!=nf || CopyBuffer(g_mrSlow,0,1,1,slow)!=1)return false;
   if(fast[nf-1]<=0 || slow[0]<=0 || fast[nf-1]==EMPTY_VALUE || slow[0]==EMPTY_VALUE)return false;
   buy=allowed && fast[nf-1]>slow[0] && fast[nf-1]>fast[0];
   sell=allowed && fast[nf-1]<slow[0] && fast[nf-1]<fast[0];
  }
  g_mrLong=buy;g_mrShort=sell;g_mrRegimeBar=bar;
 }
 return direction==1?g_mrLong:g_mrShort;
}

input ENUM_TIMEFRAMES InpMRSignalTF=PERIOD_M30;

input group "Crypto: macrotrend per rientri locali"
input ENUM_TIMEFRAMES InpMRMacroTF=PERIOD_H4;
input int InpMRMacroFast=12;
input int InpMRMacroSlow=36;
input int InpMRMacroSlopeBars=3;

int OnInit() {
 if((InpMRMacroTF!=PERIOD_H1 && InpMRMacroTF!=PERIOD_H4) || InpMRMacroFast<2 ||
    InpMRMacroSlow<=InpMRMacroFast || InpMRMacroSlopeBars<1 || InpMRMacroSlopeBars>12)return INIT_PARAMETERS_INCORRECT;

 if(InpMRSignalTF!=PERIOD_M15 && InpMRSignalTF!=PERIOD_M30 && InpMRSignalTF!=PERIOD_H1)return INIT_PARAMETERS_INCORRECT;

 if(InpMRBandPeriod<5 || InpMRBandDeviation<=0 || InpMRRSIPeriod<2 ||
    InpMRRSILow<=0 || InpMRRSIHigh>=100 || InpMRRSILow>=InpMRRSIHigh ||
    InpMRStopATR<=0 || InpMRStopBufferATR<0 || InpMRMaxStopATR<InpMRStopATR ||
    InpMRMinBodyATR<0 || InpMRMaxGapATR<0 || InpMRMinRewardR<0 ||
    InpMRTargetMode<0 || InpMRTargetMode>1 || InpMRRegime<0 || InpMRRegime>2 ||
    InpMRADXMax<0 || (InpMRRegime==1 && InpMRADXMax<=0))return INIT_PARAMETERS_INCORRECT;
 g_mrBands=iBands(_Symbol,InpMRSignalTF,InpMRBandPeriod,0,InpMRBandDeviation,PRICE_CLOSE);
 g_mrATR=iATR(_Symbol,InpMRSignalTF,14);
 g_mrRSI=iRSI(_Symbol,InpMRSignalTF,InpMRRSIPeriod,PRICE_CLOSE);
 if(g_mrBands==INVALID_HANDLE || g_mrATR==INVALID_HANDLE || g_mrRSI==INVALID_HANDLE)return INIT_FAILED;
 if(InpMRRegime>0){
  g_mrADX=iADX(_Symbol,InpMRMacroTF,14);if(g_mrADX==INVALID_HANDLE)return INIT_FAILED;
 }
 if(InpMRRegime==2){
  g_mrFast=iMA(_Symbol,InpMRMacroTF,InpMRMacroFast,0,MODE_EMA,PRICE_CLOSE);
  g_mrSlow=iMA(_Symbol,InpMRMacroTF,InpMRMacroSlow,0,MODE_EMA,PRICE_CLOSE);
  if(g_mrFast==INVALID_HANDLE || g_mrSlow==INVALID_HANDLE)return INIT_FAILED;
 }

    if(InpCryptoWeekendRiskMult<0 || InpCryptoWeekendRiskMult>1)return INIT_PARAMETERS_INCORRECT;
    if(InpCryptoMaxHoldHours<0 || InpCryptoMaxPositions<1 || InpCryptoMaxStopATR<0 || InpCryptoLongRiskMult<0 || InpCryptoShortRiskMult<0)return INIT_PARAMETERS_INCORRECT;
    if(InpCryptoMaxPositions>1 && AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)return INIT_PARAMETERS_INCORRECT;
    if(InpRequireWeekendSessions) {
        datetime from,to;
        if(!SymbolInfoSessionTrade(_Symbol,SATURDAY,0,from,to) || !SymbolInfoSessionTrade(_Symbol,SUNDAY,0,from,to)) {
            Print("CRYPTO_INIT_FAILED: il simbolo non ha sessioni sia sabato sia domenica: ",_Symbol);
            return INIT_PARAMETERS_INCORRECT;
        }
    }
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
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
 if(g_mrBands!=INVALID_HANDLE)IndicatorRelease(g_mrBands);
 if(g_mrATR!=INVALID_HANDLE)IndicatorRelease(g_mrATR);
 if(g_mrRSI!=INVALID_HANDLE)IndicatorRelease(g_mrRSI);
 if(g_mrADX!=INVALID_HANDLE)IndicatorRelease(g_mrADX);
 if(g_mrFast!=INVALID_HANDLE)IndicatorRelease(g_mrFast);
 if(g_mrSlow!=INVALID_HANDLE)IndicatorRelease(g_mrSlow);

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

void PrintFinalReport(){
 PrintSessionStats("MEAN_REVERT",g_mrProfit,g_mrWins,g_mrTrades);
}

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


int CryptoPositionCount(){
 int n=0;for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket>0 && PositionGetString(POSITION_SYMBOL)==_Symbol && PositionGetInteger(POSITION_MAGIC)==(long)InpMagicNumber)n++;
 }return n;
}
bool CryptoSessionOpen(string session){
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket>0 && PositionGetString(POSITION_SYMBOL)==_Symbol && PositionGetInteger(POSITION_MAGIC)==(long)InpMagicNumber && StringFind(PositionGetString(POSITION_COMMENT),session+"|")==0)return true;
 }return false;
}
double CryptoOpenStopRisk(){
 MqlTick quote;if(!SymbolInfoTick(_Symbol,quote))return DBL_MAX;
 double total=0;
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
  double stop=PositionGetDouble(POSITION_SL),pnl=0;
  if(stop<=0 || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,stop,pnl))return DBL_MAX;
  total+=MathMax(0,-pnl);
 }return total;
}
void CryptoExpirePositions(){
 if(InpCryptoMaxHoldHours<=0)return;
 static datetime lastMinute=0;
 datetime minute=(datetime)(TimeCurrent()/60);if(minute==lastMinute)return;lastMinute=minute;
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  if(TimeCurrent()-(datetime)PositionGetInteger(POSITION_TIME)>=InpCryptoMaxHoldHours*3600){
   if(!trade.PositionClose(ticket) || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))Print("CRYPTO_CLOSE_FAILED ",ticket," retcode ",trade.ResultRetcode());
  }
 }
}
void CryptoWarmDailyBias(){
 if(!InpCryptoInitializeDailyBias)return;
 ArrayResize(g_daily_swing_highs,0);ArrayResize(g_daily_swing_lows,0);
 int oldest=MathMin(300,Bars(_Symbol,PERIOD_D1)-InpSwingLen_Daily-1);
 for(int shift=oldest;shift>=InpSwingLen_Daily+1;shift--){
  double high=GetPivot(PERIOD_D1,InpSwingLen_Daily,shift,true),low=GetPivot(PERIOD_D1,InpSwingLen_Daily,shift,false);
  if(high!=EMPTY_VALUE){AddToArray(g_daily_swing_highs,high);if(ArraySize(g_daily_swing_highs)>10)ShiftArray(g_daily_swing_highs);}
  if(low!=EMPTY_VALUE){AddToArray(g_daily_swing_lows,low);if(ArraySize(g_daily_swing_lows)>10)ShiftArray(g_daily_swing_lows);}
 }g_dailyBias=GetDailyBias();
}


void OnTick(){
 CryptoExpirePositions();
 ManageOpenPositions();
 static datetime lastMinute=0;
 datetime minute=iTime(_Symbol,PERIOD_M1,0);
 if(minute<=0 || minute==lastMinute)return;
 lastMinute=minute;
 MqlDateTime clock;TimeToStruct(TimeCurrent(),clock);
 datetime today=StringToTime(TimeToString(TimeCurrent(),TIME_DATE));
 if(today!=g_currentDay){
  g_currentDay=today;g_dayStartEquity=AccountInfoDouble(ACCOUNT_EQUITY);
  g_dayLocked=false;g_tradesToday=0;g_lossesToday=0;
  ArrayResize(g_countedLossIds,0);
 }
 if(InpUseEquityDailyLock && g_dayStartEquity>0 &&
    100.0*(g_dayStartEquity-AccountInfoDouble(ACCOUNT_EQUITY))/g_dayStartEquity>=InpMaxDailyLossPct)g_dayLocked=true;
 if(g_dayLocked || g_tradesToday>=InpMaxTradesDay || CryptoPositionCount()>=InpCryptoMaxPositions)return;
 if(!InpTradeWeekend && (clock.day_of_week==0 || clock.day_of_week==6))return;
 datetime bar=iTime(_Symbol,InpMRSignalTF,0);
 if(bar<=0 || bar==g_mrSignalBar)return;
 MqlRates rates[];ArraySetAsSeries(rates,true);
 if(CopyRates(_Symbol,InpMRSignalTF,1,2,rates)!=2)return;
 double mid[],upper[],lower[],atr[],rsi[];
 if(CopyBuffer(g_mrBands,0,1,2,mid)!=2 || CopyBuffer(g_mrBands,1,1,2,upper)!=2 ||
    CopyBuffer(g_mrBands,2,1,2,lower)!=2 || CopyBuffer(g_mrATR,0,1,1,atr)!=1 ||
    CopyBuffer(g_mrRSI,0,2,1,rsi)!=1 || atr[0]<=0)return;
 // Rates serie:0=barra1,1=barra2; buffer normali:0=barra2,1=barra1.
 // Si entra DOPO che barra1 ha confermato il rientro; mai valori barra0.
 g_mrSignalBar=bar;
 int direction=0;
 double body=rates[0].close-rates[0].open;
 if(rates[1].close<lower[0] && rsi[0]<=InpMRRSILow &&
    rates[0].close>lower[1] && rates[0].close<mid[1] && body>atr[0]*InpMRMinBodyATR)direction=1;
 if(rates[1].close>upper[0] && rsi[0]>=InpMRRSIHigh &&
    rates[0].close<upper[1] && rates[0].close>mid[1] && -body>atr[0]*InpMRMinBodyATR)direction=-1;
 if(direction==0 || !CryptoMRRegimeAllows(direction))return;
 MqlTick quote;if(!SymbolInfoTick(_Symbol,quote))return;
 double entry=direction==1?quote.ask:quote.bid;
 if(MathAbs(entry-rates[0].close)>atr[0]*InpMRMaxGapATR)return;
 double extreme=direction==1?MathMin(rates[0].low,rates[1].low):MathMax(rates[0].high,rates[1].high);
 double stop=extreme-direction*atr[0]*InpMRStopBufferATR;
 double distance=MathMax(MathAbs(entry-stop),atr[0]*InpMRStopATR);
 if(distance>atr[0]*InpMRMaxStopATR)return;
 stop=entry-direction*distance;
 double target=InpMRTargetMode==0?mid[1]:entry+direction*distance*InpRR_Target;
 if(direction*(target-entry)<distance*InpMRMinRewardR)return;
 ExecuteTrade(direction,entry,stop,"MEAN_REVERT",GetRiskAmount(),target);
}

void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
    if((dir==1 && !InpCryptoAllowLong) || (dir==-1 && !InpCryptoAllowShort))return;
    if(CryptoPositionCount()>=InpCryptoMaxPositions || CryptoSessionOpen(session))return;
    riskAmount*=dir==1?InpCryptoLongRiskMult:InpCryptoShortRiskMult;
    MqlDateTime riskClock;TimeToStruct(TimeCurrent(),riskClock);
    if(riskClock.day_of_week==0 || riskClock.day_of_week==6)riskAmount*=InpCryptoWeekendRiskMult;

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
    if(tickSize<=0 || riskAmount<=0 || entry<=0) return;
    double sl_final=NormalizeDouble(MathRound(sl_ref/tickSize)*tickSize,digits);
    double minStop=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
    if((dir==1 && (sl_final>=quote.bid-minStop)) || (dir==-1 && sl_final<=quote.ask+minStop)) return;

    double stopDistance=MathAbs(entry-sl_final);
    if(InpCryptoMaxStopATR>0){double atr=GetATRValue();if(atr<=0 || stopDistance>atr*InpCryptoMaxStopATR)return;}
    double aggregateBudget=AccountInfoDouble(ACCOUNT_EQUITY)*InpTotalOpenStopRiskPct/100.0-CryptoOpenStopRisk();
    riskAmount=MathMin(riskAmount,aggregateBudget);if(riskAmount<=0)return;

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

        long posId=0; double filledEntry=entry;
        ulong filledDeal=trade.ResultDeal();
        if(filledDeal>0 && HistoryDealSelect(filledDeal)) {
            posId=HistoryDealGetInteger(filledDeal,DEAL_POSITION_ID);
            filledEntry=HistoryDealGetDouble(filledDeal,DEAL_PRICE);
        }
        if(posId > 0) RegisterPosition(posId, session, filledEntry, MathAbs(filledEntry - sl_final), dir);

        if(session=="MEAN_REVERT")g_mrTrades++;
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
            if(session=="MEAN_REVERT"){g_mrProfit+=totalPnL;if(totalPnL>0)g_mrWins++;}
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
