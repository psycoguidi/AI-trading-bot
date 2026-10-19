//+------------------------------------------------------------------+
//|              Tri_Session_Master_v5.7_PerSymbolParams.mq5         |
//|         Logic: London/NY/Asia ORB - Sniper Filters               |
//|         + FADE falsi breakout + gestione dinamica                |
//|         + MULTI-SIMBOLO con PARAMETRI INDIPENDENTI (v5.7)        |
//+------------------------------------------------------------------+
#property copyright "Tri-Session Master v5.7 Per-Symbol Params"
#property version   "5.70"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS
// ═══════════════════════════════════════════
input group "Portafoglio Multi-Simbolo (NEW v5.7: parametri per-simbolo)"
// Formato: "SIMBOLO;chiave=valore;chiave=valore|SIMBOLO;chiave=valore|..."
// Chiavi disponibili: risk (% equity), body (% corpo candela), atrmult (moltiplicatore ATR),
//                      rr (RR target continuazione), faderiskpct (% rischio sulla fade), fadeatrfrac (frazione ATR per buffer SL fade)
// Chiavi omesse = usa il default della relativa sezione qui sotto.
input string   InpSymbolConfig      = "XAUUSD;risk=1.0|EURUSD;risk=1.0"; //|EURUSD;risk=1.0
input int      InpMaxConcurrentPositions = 3;

input group "Diagnostica / Esclusioni Combo"
// Formato: "SIMBOLO_SESSIONE,SIMBOLO_SESSIONE,..." Sessioni valide: LONDON, NY, ASIA
input string   InpDisabledCombos     = "";

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

input group "Sniper Filters - DEFAULT (sovrascrivibili per simbolo)"
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M5;
input int      InpMaxEntryDelayMin  = 60;
input double   InpBodyFilterPct     = 50.0;   // default "body="
input double   InpATRMinRangeMult   = 0.35;   // default "atrmult="
input double   InpRR_Target         = 2;      // default "rr="

input group "Fade dei Falsi Breakout - DEFAULT (sovrascrivibili per simbolo)"
input bool     InpFadeEnabled       = true;
input double   InpFadeRiskPct       = 50.0;   // default "faderiskpct="
input double   InpFadeSLBufferATRFrac = 0.05; // default "fadeatrfrac="

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;

input group "Break-Even & Partial Close"
input bool     InpUseBreakEven      = true;
input double   InpBreakEvenTriggerR = 1.0;
input double   InpBreakEvenLockR    = 0.1;
input bool     InpUsePartialClose   = true;
input double   InpPartialClosePct   = 50.0;
input double   InpPartialCloseAtR   = 1.0;

input group "Risk & Money Management - DEFAULT (sovrascrivibile per simbolo con 'risk=')"
input double   InpFixedRiskAmount   = 80.0;
input bool     InpUseEquityRisk     = true;   // v5.7: di default in % come nel tuo test
input double   InpRiskPercent       = 1.0;    // default "risk="
input int      InpMaxLossesDay      = 3;
input int      InpMaxTradesDay      = 6;
input ulong    InpMagicNumber       = 20260911;
input bool     InpShowComments      = true;

input group "Daily Loss Control (portafoglio)"
input bool     InpUseEquityDailyLock = true;
input double   InpMaxDailyLossPct    = 3.0;

// ═══════════════════════════════════════════
// 🧠 STRUTTURE & GLOBALS
// ══════════════════════════════════════════
CTrade trade;
int    InpSwingLen_Daily = 5;
string g_disabledCombos[];

struct SymbolParams {
    double riskPct;
    double bodyPct;
    double atrMult;
    double rrTarget;
    double fadeRiskPct;
    double fadeATRFrac;
};

struct SessionState {
    string   name;
    double   high, low;
    bool     rangeDefined;
    bool     executedToday;
    int      fadeDir;
    double   fadeSL;
    bool     reArmActive;
    datetime reArmDeadline;
    double   profitCont, profitFade, profitRev;
    int      winsCont, winsFade, winsRev;
    int      totalCont, totalFade, totalRev;
};

struct SymbolData {
    string   symbol;
    SymbolParams params;
    int      atrHandle;
    SessionState lon;
    SessionState ny;
    SessionState asia;
    int      dailyBias;
    double   swingHighs[];
    double   swingLows[];
    datetime lastD1;
    int      tradesToday;
    int      lossesToday;
    bool     dayLocked;
    long     countedLossIds[];
};
SymbolData g_symbols[];

struct SessionMap {
    long   posId;
    int    symbolIdx;
    int    sessionIdx;
    int    tradeType;
    double entryPrice;
    double slDistance;
    int    dir;
    bool   beDone;
    bool   partialDone;
};
SessionMap g_posMap[];

datetime g_currentDay = 0;
double   g_dayStartEquity = 0.0;
bool     g_portfolioLocked = false;

// ═══════════════════════════════════════════
// 🔧 PARSER CONFIG PER-SIMBOLO (NEW v5.7)
// ═══════════════════════════════════════════
int ParseSymbolConfig(string cfg, string &symbols[], SymbolParams &params[]) {
    string blocks[];
    int n = StringSplit(cfg, '|', blocks);
    ArrayResize(symbols, n);
    ArrayResize(params, n);
    for(int i=0; i<n; i++) {
        string fields[];
        int fn = StringSplit(blocks[i], ';', fields);
        if(fn <= 0) { symbols[i] = ""; continue; }
        string sym = fields[0]; StringTrimLeft(sym); StringTrimRight(sym);
        symbols[i] = sym;

        // Default = valori globali
        params[i].riskPct     = InpRiskPercent;
        params[i].bodyPct     = InpBodyFilterPct;
        params[i].atrMult     = InpATRMinRangeMult;
        params[i].rrTarget    = InpRR_Target;
        params[i].fadeRiskPct = InpFadeRiskPct;
        params[i].fadeATRFrac = InpFadeSLBufferATRFrac;

        for(int f=1; f<fn; f++) {
            string kv[];
            int kvn = StringSplit(fields[f], '=', kv);
            if(kvn != 2) continue;
            string key = kv[0]; StringTrimLeft(key); StringTrimRight(key);
            string valStr = kv[1]; StringTrimLeft(valStr); StringTrimRight(valStr);
            double val = StringToDouble(valStr);
            if(key == "risk") params[i].riskPct = val;
            else if(key == "body") params[i].bodyPct = val;
            else if(key == "atrmult") params[i].atrMult = val;
            else if(key == "rr") params[i].rrTarget = val;
            else if(key == "faderiskpct") params[i].fadeRiskPct = val;
            else if(key == "fadeatrfrac") params[i].fadeATRFrac = val;
        }
    }
    return n;
}

// ═══════════════════════════════════════════
// 🚀 INIT & DEINIT
// ══════════════════════════════════════════
int OnInit() {
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    string symNames[];
    SymbolParams symParams[];
    int n = ParseSymbolConfig(InpSymbolConfig, symNames, symParams);
    if(n <= 0) { Print("ERRORE: InpSymbolConfig vuota o non valida."); return(INIT_FAILED); }

    ArrayResize(g_symbols, n);
    for(int i=0; i<n; i++) {
        string sym = symNames[i];
        g_symbols[i].symbol = sym;
        g_symbols[i].params = symParams[i];
        if(!SymbolSelect(sym, true)) Print("ATTENZIONE: impossibile selezionare ", sym, " nel Market Watch.");
        g_symbols[i].atrHandle = iATR(sym, PERIOD_D1, InpATRPeriod);
        if(g_symbols[i].atrHandle == INVALID_HANDLE) Print("ATTENZIONE: ATR handle non creato per ", sym);
        g_symbols[i].lon.name  = "LONDON";
        g_symbols[i].ny.name   = "NY";
        g_symbols[i].asia.name = "ASIA";
        g_symbols[i].lastD1 = 0;
        g_symbols[i].dailyBias = 0;
        ArrayResize(g_symbols[i].swingHighs, 0);
        ArrayResize(g_symbols[i].swingLows, 0);
        ArrayResize(g_symbols[i].countedLossIds, 0);
        ResetSymbolDay(i);
        RestoreDailyStats(i);

        Print("SIMBOLO CARICATO: ", sym,
              " | risk=", DoubleToString(g_symbols[i].params.riskPct,2),
              " body=", DoubleToString(g_symbols[i].params.bodyPct,1),
              " atrmult=", DoubleToString(g_symbols[i].params.atrMult,2),
              " rr=", DoubleToString(g_symbols[i].params.rrTarget,2),
              " faderiskpct=", DoubleToString(g_symbols[i].params.fadeRiskPct,1),
              " fadeatrfrac=", DoubleToString(g_symbols[i].params.fadeATRFrac,3));
    }

    if(StringLen(InpDisabledCombos) > 0) {
        int dn = StringSplit(InpDisabledCombos, ',', g_disabledCombos);
        for(int i=0; i<dn; i++) { StringTrimLeft(g_disabledCombos[i]); StringTrimRight(g_disabledCombos[i]); }
    } else {
        ArrayResize(g_disabledCombos, 0);
    }

    ArrayResize(g_posMap, 0);

    MqlDateTime dtInit; TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
    g_portfolioLocked = false;

    Print("Tri-Session Master v5.7 Per-Symbol Ready | Config: ", InpSymbolConfig);
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    PrintFinalReport();
    for(int i=0; i<ArraySize(g_symbols); i++) if(g_symbols[i].atrHandle != INVALID_HANDLE) IndicatorRelease(g_symbols[i].atrHandle);
    Comment("");
}

// ═══════════════════════════════════════════
// 🔧 UTILS
// ═══════════════════════════════════════════
void AddToArray(double &arr[], double val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }
void ShiftArray(double &arr[]) {
    if(ArraySize(arr)<=1){ArrayResize(arr,0);return;}
    for(int i=0;i<ArraySize(arr)-1;i++) arr[i]=arr[i+1];
    ArrayResize(arr,ArraySize(arr)-1);
}
void AddToLongArray(long &arr[], long val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }

bool IsComboDisabled(string sym, string sessionName) {
    string key = sym + "_" + sessionName;
    for(int i=0; i<ArraySize(g_disabledCombos); i++) if(g_disabledCombos[i] == key) return true;
    return false;
}

double GetRiskAmount(int idx) {
    if(InpUseEquityRisk) return AccountInfoDouble(ACCOUNT_EQUITY) * (g_symbols[idx].params.riskPct/100.0);
    return InpFixedRiskAmount;
}

double GetATRValue(int idx) {
    if(g_symbols[idx].atrHandle == INVALID_HANDLE) return -1;
    double buf[];
    ArraySetAsSeries(buf, true);
    if(CopyBuffer(g_symbols[idx].atrHandle, 0, 1, 1, buf) <= 0) return -1;
    return buf[0];
}

int CountOpenPositions() {
    int c = 0;
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong t = PositionGetTicket(i);
        if(t == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) c++;
    }
    return c;
}

bool HasOpenPositionForSymbol(string sym) {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong t = PositionGetTicket(i);
        if(t == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) == sym && PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) return true;
    }
    return false;
}

void RegisterPosition(long posId, int symbolIdx, int sessionIdx, int tradeType, double entryPrice, double slDistance, int dir) {
    for(int i=ArraySize(g_posMap)-1; i>=0; i--) if(g_posMap[i].posId == posId) ArrayRemove(g_posMap, i, 1);
    int s = ArraySize(g_posMap); ArrayResize(g_posMap, s+1);
    g_posMap[s].posId = posId;
    g_posMap[s].symbolIdx = symbolIdx;
    g_posMap[s].sessionIdx = sessionIdx;
    g_posMap[s].tradeType = tradeType;
    g_posMap[s].entryPrice = entryPrice;
    g_posMap[s].slDistance = slDistance;
    g_posMap[s].dir = dir;
    g_posMap[s].beDone = false;
    g_posMap[s].partialDone = false;
}

int FindPosMapIndex(long posId) {
    for(int i=0; i<ArraySize(g_posMap); i++) if(g_posMap[i].posId == posId) return i;
    return -1;
}

void UnregisterPosition(long posId) {
    for(int i=ArraySize(g_posMap)-1; i>=0; i--) if(g_posMap[i].posId == posId) { ArrayRemove(g_posMap, i, 1); return; }
}

bool IsIdAlreadyCounted(int idx, long posId) {
    for(int i=0; i<ArraySize(g_symbols[idx].countedLossIds); i++) if(g_symbols[idx].countedLossIds[i] == posId) return true;
    return false;
}

double GetPivot(string sym, ENUM_TIMEFRAMES tf, int len, int shift, bool isHigh) {
    double val = isHigh ? iHigh(sym, tf, shift) : iLow(sym, tf, shift);
    for(int i=1; i<=len; i++) {
        double c1 = isHigh ? iHigh(sym, tf, shift+i) : iLow(sym, tf, shift+i);
        double c2 = isHigh ? iHigh(sym, tf, shift-i) : iLow(sym, tf, shift-i);
        if((isHigh && (c1 >= val || c2 >= val)) || (!isHigh && (c1 <= val || c2 <= val))) return EMPTY_VALUE;
    }
    return val;
}

int GetDailyBias(int idx, string sym) {
    if(ArraySize(g_symbols[idx].swingHighs) < 2 || ArraySize(g_symbols[idx].swingLows) < 2) return 0;
    double lastHH = g_symbols[idx].swingHighs[ArraySize(g_symbols[idx].swingHighs)-1];
    double prevHH = g_symbols[idx].swingHighs[ArraySize(g_symbols[idx].swingHighs)-2];
    double lastLL = g_symbols[idx].swingLows[ArraySize(g_symbols[idx].swingLows)-1];
    double prevLL = g_symbols[idx].swingLows[ArraySize(g_symbols[idx].swingLows)-2];
    double close = iClose(sym, PERIOD_D1, 1);
    if(close > lastHH || (lastHH > prevHH && lastLL > prevLL)) return 1;
    if(close < lastLL || (lastHH < prevHH && lastLL < prevLL)) return -1;
    return 0;
}

void UpdateDailyBias(int idx) {
    string sym = g_symbols[idx].symbol;
    datetime d1time = iTime(sym, PERIOD_D1, 0);
    if(d1time == g_symbols[idx].lastD1) return;
    g_symbols[idx].lastD1 = d1time;
    double ph = GetPivot(sym, PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, true);
    double pl = GetPivot(sym, PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, false);
    if(ph != EMPTY_VALUE) { AddToArray(g_symbols[idx].swingHighs, ph); if(ArraySize(g_symbols[idx].swingHighs)>10) ShiftArray(g_symbols[idx].swingHighs); }
    if(pl != EMPTY_VALUE) { AddToArray(g_symbols[idx].swingLows, pl); if(ArraySize(g_symbols[idx].swingLows)>10) ShiftArray(g_symbols[idx].swingLows); }
    g_symbols[idx].dailyBias = GetDailyBias(idx, sym);
}

void ResetSymbolDay(int idx) {
    g_symbols[idx].lon.high=0; g_symbols[idx].lon.low=999999.0; g_symbols[idx].lon.rangeDefined=false; g_symbols[idx].lon.executedToday=false; g_symbols[idx].lon.reArmActive=false;
    g_symbols[idx].ny.high=0; g_symbols[idx].ny.low=999999.0; g_symbols[idx].ny.rangeDefined=false; g_symbols[idx].ny.executedToday=false; g_symbols[idx].ny.reArmActive=false;
    g_symbols[idx].asia.high=0; g_symbols[idx].asia.low=999999.0; g_symbols[idx].asia.rangeDefined=false; g_symbols[idx].asia.executedToday=false; g_symbols[idx].asia.reArmActive=false;
    g_symbols[idx].tradesToday=0; g_symbols[idx].lossesToday=0; g_symbols[idx].dayLocked=false;
    ArrayResize(g_symbols[idx].countedLossIds, 0);
}

void RestoreDailyStats(int idx) {
    string sym = g_symbols[idx].symbol;
    datetime todayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    if(!HistorySelect(todayStart, TimeCurrent())) return;
    int totalDeals = HistoryDealsTotal();
    for(int i=0; i<totalDeals; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
        if(HistoryDealGetString(ticket, DEAL_SYMBOL) != sym) continue;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) g_symbols[idx].tradesToday++;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_OUT) {
            long posId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
            double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + HistoryDealGetDouble(ticket, DEAL_SWAP) + HistoryDealGetDouble(ticket, DEAL_COMMISSION);
            bool stillOpen = false;
            for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
            if(!stillOpen && profit < 0 && !IsIdAlreadyCounted(idx, posId)) { g_symbols[idx].lossesToday++; AddToLongArray(g_symbols[idx].countedLossIds, posId); }
        }
    }
    if(g_symbols[idx].lossesToday >= InpMaxLossesDay) g_symbols[idx].dayLocked = true;
}

double ReportSession(string label, SessionState &s, int &winsAcc, int &totalAcc) {
    double p = s.profitCont + s.profitFade + s.profitRev;
    int w = s.winsCont + s.winsFade + s.winsRev;
    int t = s.totalCont + s.totalFade + s.totalRev;
    winsAcc += w; totalAcc += t;
    double wr = (t > 0) ? (double)w/t*100.0 : 0.0;
    Print(label, " | Profit: ", DoubleToString(p,2), " | Cont:", s.totalCont, " Fade:", s.totalFade, " Rev:", s.totalRev,
          " | Wins/Total: ", w, "/", t, " | WR: ", DoubleToString(wr,1), "%");
    return p;
}

void PrintFinalReport() {
    Print("=====================================================");
    Print("TRI-SESSION MASTER v5.7 PER-SYMBOL PARAMS - FINAL REPORT");
    Print("=====================================================");
    double grandProfit = 0; int grandWins = 0, grandTotal = 0;
    for(int i=0; i<ArraySize(g_symbols); i++) {
        Print("--- ", g_symbols[i].symbol, " (risk=", DoubleToString(g_symbols[i].params.riskPct,2), "%) ---");
        grandProfit += ReportSession("  LONDON", g_symbols[i].lon, grandWins, grandTotal);
        grandProfit += ReportSession("  NY    ", g_symbols[i].ny, grandWins, grandTotal);
        grandProfit += ReportSession("  ASIA  ", g_symbols[i].asia, grandWins, grandTotal);
    }
    Print("=====================================================");
    Print("TOTALE PORTAFOGLIO | Profit: ", DoubleToString(grandProfit,2), " | Wins/Total: ", grandWins, "/", grandTotal);
    Print("=====================================================");
}

void ShowPortfolioComment() {
    int openPos = CountOpenPositions();
    int tradesToday = 0;
    for(int i=0; i<ArraySize(g_symbols); i++) tradesToday += g_symbols[i].tradesToday;
    Comment(StringFormat("Tri-Session v5.7 | Simboli: %d | Posizioni aperte: %d/%d | Trade oggi: %d | Portfolio Lock: %s",
          ArraySize(g_symbols), openPos, InpMaxConcurrentPositions, tradesToday, (g_portfolioLocked?"SI":"NO")));
}

// ══════════════════════════════════════════
// 🎯 GESTIONE DINAMICA POSIZIONI
// ══════════════════════════════════════════
void ApplyClose(SessionState &s, int tradeType, double pnl) {
    bool win = (pnl > 0);
    if(tradeType == 0) { s.profitCont += pnl; if(win) s.winsCont++; }
    else if(tradeType == 1) { s.profitFade += pnl; if(win) s.winsFade++; }
    else { s.profitRev += pnl; if(win) s.winsRev++; }
}

void ArmReversal(SessionState &s) {
    s.reArmActive = true;
    s.reArmDeadline = TimeCurrent() + 30*60;
}

void ManageOpenPositions() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        long posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
        int mIdx = FindPosMapIndex(posId);
        if(mIdx < 0) continue;

        string sym = PositionGetString(POSITION_SYMBOL);
        int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

        double entry  = g_posMap[mIdx].entryPrice;
        double slDist = g_posMap[mIdx].slDistance;
        int    dir    = g_posMap[mIdx].dir;
        if(slDist <= 0) continue;

        double currentPrice = (dir == 1) ? SymbolInfoDouble(sym, SYMBOL_BID) : SymbolInfoDouble(sym, SYMBOL_ASK);
        double rMultiple = (dir == 1) ? (currentPrice - entry) / slDist : (entry - currentPrice) / slDist;

        double volume    = PositionGetDouble(POSITION_VOLUME);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);

        if(InpUsePartialClose && !g_posMap[mIdx].partialDone && rMultiple >= InpPartialCloseAtR) {
            double volMin  = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
            double volStep = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
            double closeVol = MathFloor((volume * (InpPartialClosePct/100.0)) / volStep) * volStep;
            if(closeVol >= volMin && closeVol < volume) {
                if(trade.PositionClosePartial(ticket, closeVol)) {
                    g_posMap[mIdx].partialDone = true;
                    Print("PARTIAL CLOSE: ", sym, " | PosID: ", posId, " | Vol: ", DoubleToString(closeVol,2), " @ R=", DoubleToString(rMultiple,2));
                }
            } else {
                g_posMap[mIdx].partialDone = true;
            }
        }

        if(InpUseBreakEven && !g_posMap[mIdx].beDone && rMultiple >= InpBreakEvenTriggerR) {
            double beSL = (dir == 1) ? entry + (InpBreakEvenLockR * slDist) : entry - (InpBreakEvenLockR * slDist);
            beSL = NormalizeDouble(beSL, digits);
            bool improves = (dir == 1) ? (beSL > currentSL) : (currentSL == 0 || beSL < currentSL);
            if(improves) {
                if(trade.PositionModify(ticket, beSL, currentTP)) {
                    g_posMap[mIdx].beDone = true;
                    Print("BREAK-EVEN: ", sym, " | PosID: ", posId, " | Nuovo SL: ", DoubleToString(beSL,digits));
                }
            }
        }
    }
}

// ══════════════════════════════════════════
// 💼 EXECUTION
// ══════════════════════════════════════════
void ExecuteTrade(int symbolIdx, int sessionIdx, SessionState &s, int tradeType, int dir, double entry, double sl_ref, string label, double riskAmount, double customTP, double rrTarget) {
    string sym = g_symbols[symbolIdx].symbol;
    int digits = (int)SymbolInfoInteger(sym, SYMBOL_DIGITS);

    double sl_final = NormalizeDouble(sl_ref, digits);
    double tp_final;
    if(customTP > 0) {
        tp_final = NormalizeDouble(customTP, digits);
    } else {
        double dist = MathAbs(entry - sl_final);
        if(dist == 0) return;
        tp_final = NormalizeDouble((dir == 1) ? entry + (dist * rrTarget) : entry - (dist * rrTarget), digits);
    }
    if(dir == 1 && tp_final <= entry) return;
    if(dir == -1 && tp_final >= entry) return;

    double minLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(sym, SYMBOL_VOLUME_STEP);
    double profitPerLot = 0;
    bool calcRes = (dir == 1) ? OrderCalcProfit(ORDER_TYPE_BUY, sym, 1.0, entry, sl_final, profitPerLot)
                               : OrderCalcProfit(ORDER_TYPE_SELL, sym, 1.0, entry, sl_final, profitPerLot);
    if(!calcRes || profitPerLot >= 0) return;

    double lots = MathFloor((riskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
    if(lots < minLot) lots = minLot;

    string cmt = StringFormat("%s|v5.7|R:%.2f", label, riskAmount);
    bool res = (dir == 1) ? trade.Buy(lots, sym, entry, sl_final, tp_final, cmt)
                           : trade.Sell(lots, sym, entry, sl_final, tp_final, cmt);

    if(res) {
        g_symbols[symbolIdx].tradesToday++;

        long posId = 0;
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong t = PositionGetTicket(i);
            if(t == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != sym) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
            posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
            break;
        }
        if(posId > 0) RegisterPosition(posId, symbolIdx, sessionIdx, tradeType, entry, MathAbs(entry - sl_final), dir);

        if(tradeType == 0) s.totalCont++;
        else if(tradeType == 1) s.totalFade++;
        else s.totalRev++;

        Print("TRADE OPENED: ", label, " (", sym, ") | PosID: ", posId, " | Risk: ", DoubleToString(riskAmount,2), " | TP: ", DoubleToString(tp_final,digits));
    }
}

// ══════════════════════════════════════════
// 🔄 SESSION PROCESSOR
// ══════════════════════════════════════════
void ProcessSession(int idx, int sessionIdx, SessionState &s, int startHour, int startMin, int durationMin, bool useDailyFilter, bool strictBias, bool ignoreBias, MqlDateTime &dt) {
    string sym = g_symbols[idx].symbol;
    SymbolParams p = g_symbols[idx].params;

    if(IsComboDisabled(sym, s.name)) return;

    if(s.reArmActive) {
        if(TimeCurrent() > s.reArmDeadline) {
            s.reArmActive = false;
        } else if(!HasOpenPositionForSymbol(sym) && CountOpenPositions() < InpMaxConcurrentPositions) {
            double price = (s.fadeDir == -1) ? SymbolInfoDouble(sym, SYMBOL_ASK) : SymbolInfoDouble(sym, SYMBOL_BID);
            int revDir = -s.fadeDir;
            ExecuteTrade(idx, sessionIdx, s, 2, revDir, price, s.fadeSL, s.name + "_REVERSAL", GetRiskAmount(idx), 0, p.rrTarget);
            s.reArmActive = false;
        }
    }

    if(s.executedToday) return;

    datetime sessStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, startHour, startMin));
    datetime sessEnd    = sessStart + durationMin * 60;
    datetime scanLimit  = sessEnd + InpMaxEntryDelayMin * 60;

    if(TimeCurrent() < sessStart) return;

    if(TimeCurrent() < sessEnd) {
        double h = iHigh(sym, PERIOD_M1, 1);
        double l = iLow(sym, PERIOD_M1, 1);
        if(h > s.high) s.high = h;
        if(l < s.low) s.low = l;
        s.rangeDefined = true;
        return;
    }

    if(TimeCurrent() >= scanLimit) { s.executedToday = true; return; }
    if(!s.rangeDefined) return;

    if(HasOpenPositionForSymbol(sym)) return;
    if(CountOpenPositions() >= InpMaxConcurrentPositions) return;

    double atrVal = GetATRValue(idx);

    bool atrOk = true;
    if(InpUseATRFilter) {
        double sessionRange = s.high - s.low;
        if(atrVal <= 0 || sessionRange < atrVal * p.atrMult) atrOk = false;
    }
    if(!atrOk) { s.executedToday = true; return; }

    double m5_close = iClose(sym, InpEntryTF, 1);
    double m5_open  = iOpen(sym, InpEntryTF, 1);
    double m5_high  = iHigh(sym, InpEntryTF, 1);
    double m5_low   = iLow(sym, InpEntryTF, 1);
    double bodySize  = MathAbs(m5_close - m5_open);
    double totalSize = m5_high - m5_low;
    bool validBody = (totalSize > 0) && ((bodySize/totalSize)*100.0 > p.bodyPct);

    bool bull = (m5_close > s.high);
    bool bear = (m5_close < s.low);

    if(!ignoreBias && useDailyFilter) {
        if(strictBias) {
            if(bull && g_symbols[idx].dailyBias != 1) bull = false;
            if(bear && g_symbols[idx].dailyBias != -1) bear = false;
        } else {
            if(bull && g_symbols[idx].dailyBias == -1) bull = false;
            if(bear && g_symbols[idx].dailyBias == 1) bear = false;
        }
    }

    if(bull || bear) {
        double riskAmt = GetRiskAmount(idx);
        if(validBody) {
            if(bull) ExecuteTrade(idx, sessionIdx, s, 0, 1, m5_close, s.low, s.name, riskAmt, 0, p.rrTarget);
            else     ExecuteTrade(idx, sessionIdx, s, 0, -1, m5_close, s.high, s.name, riskAmt, 0, p.rrTarget);
        } else if(InpFadeEnabled) {
            double fadeRisk = riskAmt * (p.fadeRiskPct/100.0);
            double buffer = (atrVal > 0) ? (atrVal * p.fadeATRFrac) : (10 * SymbolInfoDouble(sym, SYMBOL_POINT));
            if(bull) {
                double sl = m5_high + buffer;
                s.fadeDir = -1; s.fadeSL = sl;
                ExecuteTrade(idx, sessionIdx, s, 1, -1, m5_close, sl, s.name+"_FADE", fadeRisk, s.low, p.rrTarget);
            } else {
                double sl = m5_low - buffer;
                s.fadeDir = 1; s.fadeSL = sl;
                ExecuteTrade(idx, sessionIdx, s, 1, 1, m5_close, sl, s.name+"_FADE", fadeRisk, s.high, p.rrTarget);
            }
        }
        if(HasOpenPositionForSymbol(sym)) s.executedToday = true;
    }
}

// ══════════════════════════════════════════
// 🔄 MAIN LOGIC
// ══════════════════════════════════════════
void OnTick() {
    ManageOpenPositions();

    static datetime lastMinute = 0;
    datetime nowMinute = (datetime)(((long)TimeCurrent()/60)*60);
    if(nowMinute == lastMinute) return;
    lastMinute = nowMinute;

    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(today != g_currentDay) {
        g_currentDay = today;
        g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        g_portfolioLocked = false;
        for(int i=0; i<ArraySize(g_symbols); i++) ResetSymbolDay(i);
    }

    if(InpUseEquityDailyLock && g_dayStartEquity > 0) {
        double eq = AccountInfoDouble(ACCOUNT_EQUITY);
        double lossPct = (g_dayStartEquity - eq) / g_dayStartEquity * 100.0;
        if(lossPct >= InpMaxDailyLossPct) g_portfolioLocked = true;
    }

    for(int i=0; i<ArraySize(g_symbols); i++) UpdateDailyBias(i);

    if(g_portfolioLocked) { if(InpShowComments) ShowPortfolioComment(); return; }

    for(int i=0; i<ArraySize(g_symbols); i++) {
        if(g_symbols[i].dayLocked) continue;
        if(g_symbols[i].tradesToday >= InpMaxTradesDay) continue;

        ProcessSession(i, 0, g_symbols[i].lon,  InpLon_StartHour,  InpLon_StartMin,  InpLon_DurationMin,  InpLon_UseDailyFilter,  true,  InpLon_IgnoreBias, dt);
        ProcessSession(i, 1, g_symbols[i].ny,   InpNY_StartHour,   InpNY_StartMin,   InpNY_DurationMin,   InpNY_UseDailyFilter,   false, false, dt);
        if(InpUseAsia)
            ProcessSession(i, 2, g_symbols[i].asia, InpAsia_StartHour, InpAsia_StartMin, InpAsia_DurationMin, InpAsia_UseDailyFilter, false, false, dt);
    }

    if(InpShowComments) ShowPortfolioComment();
}

// ══════════════════════════════════════════
// 📊 STATS TRACKING
// ══════════════════════════════════════════
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagicNumber) return;

    long posId = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
    int dealEntry = (int)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);
    if(dealEntry != DEAL_ENTRY_OUT) return;

    int mIdx = FindPosMapIndex(posId);
    if(mIdx < 0) return;

    int symbolIdx  = g_posMap[mIdx].symbolIdx;
    int sessionIdx = g_posMap[mIdx].sessionIdx;
    int tradeType  = g_posMap[mIdx].tradeType;

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
    for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
    if(stillOpen) return;

    if(!IsIdAlreadyCounted(symbolIdx, posId)) {
        if(sessionIdx == 0) ApplyClose(g_symbols[symbolIdx].lon, tradeType, totalPnL);
        else if(sessionIdx == 1) ApplyClose(g_symbols[symbolIdx].ny, tradeType, totalPnL);
        else ApplyClose(g_symbols[symbolIdx].asia, tradeType, totalPnL);

        AddToLongArray(g_symbols[symbolIdx].countedLossIds, posId);

        if(totalPnL < 0) {
            g_symbols[symbolIdx].lossesToday++;
            if(g_symbols[symbolIdx].lossesToday >= InpMaxLossesDay) g_symbols[symbolIdx].dayLocked = true;

            if(tradeType == 1) {
                if(sessionIdx == 0) ArmReversal(g_symbols[symbolIdx].lon);
                else if(sessionIdx == 1) ArmReversal(g_symbols[symbolIdx].ny);
                else ArmReversal(g_symbols[symbolIdx].asia);
            }
        }
        UnregisterPosition(posId);
    }
}
//+------------------------------------------------------------------+