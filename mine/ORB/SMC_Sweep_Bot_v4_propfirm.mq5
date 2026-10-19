//+------------------------------------------------------------------+
//|                                      ORB_NY_Open_v3_PropFirm.mq5 |
//|          Logic: 0.35% Risk + Displacement + Prem/Disc + Delay    |
//|      Conforme a: Prompt Analista SMC Personalizzato (v2.0)       |
//+------------------------------------------------------------------+
#property copyright "ORB Strategy v3.0 - Prop Firm Edition"
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS
// ═══════════════════════════════════════════
input group "🕒 Sessioni & Range (NY Time)"
input int      InpORB_StartHour   = 15;    // Ora Inizio Range (Server Time, es. 15 per 15:30 IT)
input int      InpORB_StartMin    = 30;    // Minuti Inizio
input int      InpORB_DurationMin = 15;    // Durata Range in Minuti (15 min)
input int      InpEntryDelayMin   = 5;     // Minuti di attesa dopo fine range (Anti-Noise)

input group "📈 Struttura & Bias (Daily/Premium-Discount)"
input bool     InpUsePremDiscFilter = true; // Filtra Long in Premium / Short in Discount
input int      InpSwingLen_Daily  = 5;     // Lookback Swing Daily (per bias strutturale opzionale)

input group "⚡ Entry & Execution (M5)"
input ENUM_TIMEFRAMES InpEntryTF  = PERIOD_M5;
input bool     InpRequireBodyClose= true;  // Richiedi chiusura corpo oltre livello
input double   InpMinBodyPct      = 0.60;  // FILTER: Min % Corpo Candela Breakout (Displacement)
input double   InpSLBuffer_Points = 30;    // Buffer SL sotto/sopra range (AfterPrime)

input group "⚖️ Risk & Money Management (PROP FIRM SAFE)"
input double   InpRiskPct         = 0.35;  // RISCHIO % FISSO (0.35% per Prop Firm Safety)
input double   InpRR_Target       = 2.0;   // Target Finale R:R (Abbassato a 2.0 per sicurezza)
input double   InpPartialCloseRR  = 1.0;   // Chiudi parziale a questo R:R
input double   InpPartialClosePct = 50.0;  // % da chiudere al TP1
input int      InpMaxLossesDay    = 2;     // MAX PERDITE GIORNALIERE -> STOP (Circuit Breaker)
input int      InpMaxTradesDay    = 2;     // Max Trade Totali/Giorno

input group "🔍 Debug"
input ulong    InpMagicNumber     = 20260908;
input bool     InpShowComments    = true;

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ═══════════════════════════════════════════
CTrade trade;

double g_orbHigh = 0.0;
double g_orbLow  = 999999.0;
bool   g_orbDefined = false;
datetime g_orbEndTime = 0;

// Dati Yesterday per Premium/Discount
double g_yestHigh = 0.0;
double g_yestLow  = 0.0;
double g_yestMid  = 0.0;
bool   g_yestDataReady = false;

int    g_dailyBias = 0; 
double g_daily_swing_highs[];
double g_daily_swing_lows[];

datetime g_currentDay = 0;
int      g_tradesToday = 0;
int      g_lossesToday = 0;
bool     g_dayLocked = false;

// ═══════════════════════════════════════════
// 🚀 INIT
// ═══════════════════════════════════════════
int OnInit() {
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20); // Slippage tolerance
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    
    Print("✅ ORB v3.0 Prop Firm Ready. Risk: ", InpRiskPct, "% | Filters: Displ + Prem/Disc");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { Comment(""); }

// ═══════════════════════════════════════════
// 🔧 UTILS
// ═══════════════════════════════════════════
void AddToArray(double &arr[], double val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }
void ShiftArray(double &arr[]) { if(ArraySize(arr)<=1){ArrayResize(arr,0);return;} for(int i=0;i<ArraySize(arr)-1;i++) arr[i]=arr[i+1]; ArrayResize(arr,ArraySize(arr)-1); }

double GetPivot(ENUM_TIMEFRAMES tf, int len, int shift, bool isHigh) {
    double val = isHigh ? iHigh(_Symbol, tf, shift) : iLow(_Symbol, tf, shift);
    for(int i=1; i<=len; i++) {
        double c1 = isHigh ? iHigh(_Symbol, tf, shift+i) : iLow(_Symbol, tf, shift+i);
        double c2 = isHigh ? iHigh(_Symbol, tf, shift-i) : iLow(_Symbol, tf, shift-i);
        if((isHigh && (c1 >= val || c2 >= val)) || (!isHigh && (c1 <= val || c2 <= val))) return EMPTY_VALUE;
    }
    return val;
}

// Calcola range Yesterday per Premium/Discount
void UpdateYesterdayRange() {
    // Prendiamo High/Low della barra Daily chiusa (shift 1)
    g_yestHigh = iHigh(_Symbol, PERIOD_D1, 1);
    g_yestLow  = iLow(_Symbol, PERIOD_D1, 1);
    g_yestMid  = (g_yestHigh + g_yestLow) / 2.0;
    g_yestDataReady = true;
}

bool HasOpenPosition() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && 
           PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) return true;
    }
    return false;
}

// ═══════════════════════════════════════════
// 🔄 MAIN LOGIC
// ═══════════════════════════════════════════
void OnTick() {
    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;
    
    ManagePositions(); // Partial Close + BE Logic
    
    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
    
    // Reset Giornaliero
    if(today != g_currentDay) {
        g_currentDay = today;
        g_tradesToday = 0;
        g_lossesToday = 0;
        g_dayLocked = false;
        g_orbDefined = false;
        g_orbHigh = 0.0;
        g_orbLow = 999999.0;
        g_yestDataReady = false;
        ArrayResize(g_daily_swing_highs, 0);
        ArrayResize(g_daily_swing_lows, 0);
    }
    
    // --- CIRCUIT BREAKER (Sez. Risk Management) ---
    if(g_dayLocked) {
        if(InpShowComments) Comment("🛑 STOP TRADING: Max Losses Reached (", g_lossesToday, "/", InpMaxLossesDay, ")");
        return;
    }
    if(g_lossesToday >= InpMaxLossesDay) {
        g_dayLocked = true;
        return;
    }
    if(g_tradesToday >= InpMaxTradesDay) return;
    if(HasOpenPosition()) return;
    
    // Aggiorna Dati Yesterday (una volta al giorno)
    if(!g_yestDataReady) UpdateYesterdayRange();
    
    // Aggiorna Struttura DAILY (Opzionale, per contesto)
    static datetime lastD1 = 0;
    if(iTime(_Symbol, PERIOD_D1, 0) != lastD1) {
        lastD1 = iTime(_Symbol, PERIOD_D1, 0);
        double ph = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, true);
        double pl = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, false);
        if(ph != EMPTY_VALUE) { AddToArray(g_daily_swing_highs, ph); if(ArraySize(g_daily_swing_highs)>10) ShiftArray(g_daily_swing_highs); }
        if(pl != EMPTY_VALUE) { AddToArray(g_daily_swing_lows, pl); if(ArraySize(g_daily_swing_lows)>10) ShiftArray(g_daily_swing_lows); }
    }
    
    // --- FASE 1: DEFINIZIONE RANGE ORB ---
    datetime orbStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpORB_StartHour, InpORB_StartMin));
    g_orbEndTime = orbStart + (InpORB_DurationMin * 60);
    
    if(TimeCurrent() >= orbStart && TimeCurrent() < g_orbEndTime) {
        double h = iHigh(_Symbol, PERIOD_M1, 1);
        double l = iLow(_Symbol, PERIOD_M1, 1);
        if(h > g_orbHigh) g_orbHigh = h;
        if(l < g_orbLow) g_orbLow = l;
        g_orbDefined = true;
    }
    
    // --- FASE 2: BREAKOUT ENTRY (Con Filtri) ---
    // Aspetta che sia passato il delay dopo la fine del range
    if(g_orbDefined && TimeCurrent() >= (g_orbEndTime + (InpEntryDelayMin * 60))) {
        
        double m5_close = iClose(_Symbol, InpEntryTF, 1);
        double m5_open = iOpen(_Symbol, InpEntryTF, 1);
        double m5_high = iHigh(_Symbol, InpEntryTF, 1);
        double m5_low = iLow(_Symbol, InpEntryTF, 1);
        
        // 1. Check Breakout Base
        bool bullBreakout = (m5_close > g_orbHigh) && (!InpRequireBodyClose || m5_close > m5_open);
        bool bearBreakout = (m5_close < g_orbLow) && (!InpRequireBodyClose || m5_close < m5_open);
        
        // 2. FILTER: Displacement (Corpo candela forte)
        double bodySize = MathAbs(m5_close - m5_open);
        double candleRange = m5_high - m5_low;
        bool isStrongCandle = (candleRange > 0) && ((bodySize / candleRange) >= InpMinBodyPct);
        
        if(!isStrongCandle) {
            bullBreakout = false;
            bearBreakout = false;
        }
        
        // 3. FILTER: Premium / Discount (Sez. Prompt)
        if(InpUsePremDiscFilter && g_yestDataReady) {
            // Long solo in Discount (Prezzo < Mid Yesterday)
            if(bullBreakout && m5_close > g_yestMid) bullBreakout = false;
            // Short solo in Premium (Prezzo > Mid Yesterday)
            if(bearBreakout && m5_close < g_yestMid) bearBreakout = false;
        }
        
        if(bullBreakout) {
            ExecuteTrade(1, m5_close, g_orbLow);
        }
        else if(bearBreakout) {
            ExecuteTrade(-1, m5_close, g_orbHigh);
        }
    }
    
    if(InpShowComments) {
        string zone = (g_yestDataReady) ? ((SymbolInfoDouble(_Symbol, SYMBOL_BID) > g_yestMid) ? "PREMIUM" : "DISCOUNT") : "CALC...";
        string orbStr = g_orbDefined ? StringFormat("H:%.2f L:%.2f", g_orbHigh, g_orbLow) : "Defining...";
        Comment(StringFormat("ORB v3.0 PF | Zone: %s | Range: %s | Trades: %d | Losses: %d/%d", 
              zone, orbStr, g_tradesToday, g_lossesToday, InpMaxLossesDay));
    }
}

// ═══════════════════════════════════════════
// 💼 EXECUTION & MANAGEMENT
// ═══════════════════════════════════════════
void ExecuteTrade(int dir, double entry, double sl_ref) {
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    double buffer = InpSLBuffer_Points * point;
    double sl, tp_final;
    
    if(dir == 1) {
        sl = sl_ref - buffer;
        double dist = entry - sl;
        tp_final = entry + (dist * InpRR_Target);
    } else {
        sl = sl_ref + buffer;
        double dist = sl - entry;
        tp_final = entry - (dist * InpRR_Target);
    }
    
    double distPrice = MathAbs(entry - sl);
    if(distPrice == 0) return;
    
    // --- CALCOLO LOTTO BASATO SU % FISSA (0.35%) ---
    double equity = AccountInfoDouble(ACCOUNT_EQUITY);
    double riskAmt = equity * (InpRiskPct / 100.0);
    
    double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSz = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    if(tickVal == 0 || tickSz == 0) return;
    
    double lots = riskAmt / ((distPrice / tickSz) * tickVal);
    
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    lots = MathFloor(lots / step) * step;
    if(lots < minLot) lots = minLot;
    
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    sl = NormalizeDouble(sl, digits);
    tp_final = NormalizeDouble(tp_final, digits);
    
    bool res = false;
    if(dir == 1) res = trade.Buy(lots, _Symbol, 0, sl, tp_final, "ORB v3 Long");
    else res = trade.Sell(lots, _Symbol, 0, sl, tp_final, "ORB v3 Short");
    
    if(res) {
        g_tradesToday++;
        Print("✅ ORB v3 TRADE: ", (dir==1?"BUY":"SELL"), " @ ", entry, " Lots:", lots, " Risk:", InpRiskPct, "%");
    }
}

void ManagePositions() {
    if(!PositionSelect(_Symbol) || PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) return;
    
    double openP = PositionGetDouble(POSITION_PRICE_OPEN);
    double sl = PositionGetDouble(POSITION_SL);
    double tp = PositionGetDouble(POSITION_TP);
    double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
    double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
    int type = (int)PositionGetInteger(POSITION_TYPE);
    double volume = PositionGetDouble(POSITION_VOLUME);
    
    double riskDist = MathAbs(openP - sl);
    double currentProfitDist = (type==POSITION_TYPE_BUY) ? (bid - openP) : (openP - ask);
    double currentRR = (riskDist > 0) ? (currentProfitDist / riskDist) : 0;
    
    // --- PARTIAL CLOSE + BE LOGIC ---
    if(currentRR >= InpPartialCloseRR && volume > 0) {
        double closeVol = volume * (InpPartialClosePct / 100.0);
        closeVol = MathFloor(closeVol / SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        
        if(closeVol >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) {
            if(type==POSITION_TYPE_BUY) trade.PositionClosePartial(_Symbol, closeVol, 0);
            else trade.PositionClosePartial(_Symbol, closeVol, 0);
            
            // Sposta SL a Break Even
            if(type==POSITION_TYPE_BUY && sl < openP) trade.PositionModify(_Symbol, openP, tp);
            if(type==POSITION_TYPE_SELL && sl > openP) trade.PositionModify(_Symbol, openP, tp);
        }
    }
}

// --- TRACKING PERDITE PER CIRCUIT BREAKER ---
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    long dealMagic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
    if(dealMagic != (long)InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;

    double dealProfit = HistoryDealGetDouble(trans.deal, DEAL_PROFIT) + HistoryDealGetDouble(trans.deal, DEAL_SWAP) + HistoryDealGetDouble(trans.deal, DEAL_COMMISSION);
    
    if(dealProfit < 0) {
        g_lossesToday++;
        if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;
    }
}
//+------------------------------------------------------------------+