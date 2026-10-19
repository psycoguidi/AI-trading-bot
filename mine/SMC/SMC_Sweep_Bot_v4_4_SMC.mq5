//+------------------------------------------------------------------+
//|                                      ORB_NY_Open_v2_Profit.mq5   |
//|          Logic: Fixed Lot + Daily Bias + Partial Close + Circuit |
//|      Conforme a: Prompt Analista SMC Personalizzato (v2.0)       |
//+------------------------------------------------------------------+
#property copyright "ORB Strategy v2.0 - Profit Focus"
#property version   "2.00"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS
// ═══════════════════════════════════════════
input group "🕒 Sessioni & Range (NY Time)"
input int      InpORB_StartHour   = 15;    // Ora Inizio Range (Server Time)
input int      InpORB_StartMin    = 30;    // Minuti Inizio
input int      InpORB_DurationMin = 15;    // Durata Range (15 min)

input group "📈 Struttura & Bias (DAILY)"
input bool     InpUseDailyFilter  = true;  // Filtra trade contro-trend Daily
input int      InpSwingLen_Daily  = 5;     // Lookback Swing Daily

input group "⚡ Entry & Execution (M5)"
input ENUM_TIMEFRAMES InpEntryTF  = PERIOD_M5;
input bool     InpRequireBodyClose= true;  // Chiusura corpo oltre livello
input double   InpSLBuffer_Points = 30;    // Buffer SL (AfterPrime)

input group "⚖️ Risk & Money Management (FIX CRITICO)"
input double   InpFixedRiskAmount = 5.0;   // RISCHIO FISSO IN VALUTA (es. 5$) - NON %
input double   InpRR_Target       = 2.5;   // Target Finale R:R
input double   InpPartialCloseRR  = 1.0;   // Chiudi parziale a questo R:R
input double   InpPartialClosePct = 50.0;  // % da chiudere al TP1
input int      InpMaxLossesDay    = 2;     // MAX PERDITE GIORNALIERE -> STOP
input int      InpMaxTradesDay    = 3;     // Max Trade Totali/Giorno

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

int    g_dailyBias = 0; // 1 Bull, -1 Bear, 0 Neutral
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
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    
    Print("✅ ORB v2.0 Ready. Fixed Risk: $", InpFixedRiskAmount, " | Max Loss/Day: ", InpMaxLossesDay);
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

int GetDailyBias() {
    if(ArraySize(g_daily_swing_highs) < 2 || ArraySize(g_daily_swing_lows) < 2) return 0;
    double lastHH = g_daily_swing_highs[ArraySize(g_daily_swing_highs)-1];
    double prevHH = g_daily_swing_highs[ArraySize(g_daily_swing_highs)-2];
    double lastLL = g_daily_swing_lows[ArraySize(g_daily_swing_lows)-1];
    double prevLL = g_daily_swing_lows[ArraySize(g_daily_swing_lows)-2];
    double close = iClose(_Symbol, PERIOD_D1, 1);
    
    // Bullish: HH + HL o Prezzo > Last HH
    if(close > lastHH || (lastHH > prevHH && lastLL > prevLL)) return 1;
    // Bearish: LH + LL o Prezzo < Last LL
    if(close < lastLL || (lastHH < prevHH && lastLL < prevLL)) return -1;
    return 0;
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
    
    // Aggiorna Struttura DAILY
    static datetime lastD1 = 0;
    if(iTime(_Symbol, PERIOD_D1, 0) != lastD1) {
        lastD1 = iTime(_Symbol, PERIOD_D1, 0);
        double ph = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, true);
        double pl = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, false);
        if(ph != EMPTY_VALUE) { AddToArray(g_daily_swing_highs, ph); if(ArraySize(g_daily_swing_highs)>10) ShiftArray(g_daily_swing_highs); }
        if(pl != EMPTY_VALUE) { AddToArray(g_daily_swing_lows, pl); if(ArraySize(g_daily_swing_lows)>10) ShiftArray(g_daily_swing_lows); }
        g_dailyBias = GetDailyBias();
    }
    
    // --- FASE 1: DEFINIZIONE RANGE ORB ---
    datetime orbStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpORB_StartHour, InpORB_StartMin));
    datetime orbEnd = orbStart + (InpORB_DurationMin * 60);
    
    if(TimeCurrent() >= orbStart && TimeCurrent() < orbEnd) {
        double h = iHigh(_Symbol, PERIOD_M1, 1);
        double l = iLow(_Symbol, PERIOD_M1, 1);
        if(h > g_orbHigh) g_orbHigh = h;
        if(l < g_orbLow) g_orbLow = l;
        g_orbDefined = true;
    }
    
    // --- FASE 2: BREAKOUT ENTRY ---
    if(g_orbDefined && TimeCurrent() >= orbEnd && g_tradesToday < InpMaxTradesDay) {
        double m5_close = iClose(_Symbol, InpEntryTF, 1);
        double m5_open = iOpen(_Symbol, InpEntryTF, 1);
        
        bool bullBreakout = (m5_close > g_orbHigh) && (!InpRequireBodyClose || m5_close > m5_open);
        bool bearBreakout = (m5_close < g_orbLow) && (!InpRequireBodyClose || m5_close < m5_open);
        
        // Filtro DAILY Bias (Sez. Multi-timeframe)
        if(InpUseDailyFilter) {
            if(bullBreakout && g_dailyBias == -1) bullBreakout = false; // No Long se Daily Bearish
            if(bearBreakout && g_dailyBias == 1) bearBreakout = false;  // No Short se Daily Bullish
        }
        
        if(bullBreakout) {
            ExecuteTrade(1, m5_close, g_orbLow);
        }
        else if(bearBreakout) {
            ExecuteTrade(-1, m5_close, g_orbHigh);
        }
    }
    
    if(InpShowComments) {
        string biasStr = (g_dailyBias==1)?"DAILY BULL":(g_dailyBias==-1)?"DAILY BEAR":"NEUTRAL";
        string orbStr = g_orbDefined ? StringFormat("H:%.2f L:%.2f", g_orbHigh, g_orbLow) : "Defining...";
        Comment(StringFormat("ORB v2.0 | %s | Range: %s | Trades: %d | Losses: %d/%d", 
              biasStr, orbStr, g_tradesToday, g_lossesToday, InpMaxLossesDay));
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
    
    // --- CALCOLO LOTTO FISSO IN VALUTA (Sez. Risk Management) ---
    double tickVal = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
    double tickSz = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
    if(tickVal == 0 || tickSz == 0) return;
    
    // Formula: Lotto = Rischio_Fisso_Valuta / (Distanza_SL_in_Tick * Valore_Tick)
    double lots = InpFixedRiskAmount / ((distPrice / tickSz) * tickVal);
    
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    lots = MathFloor(lots / step) * step;
    if(lots < minLot) lots = minLot;
    
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    sl = NormalizeDouble(sl, digits);
    tp_final = NormalizeDouble(tp_final, digits);
    
    bool res = false;
    if(dir == 1) res = trade.Buy(lots, _Symbol, 0, sl, tp_final, "ORB v2 Long");
    else res = trade.Sell(lots, _Symbol, 0, sl, tp_final, "ORB v2 Short");
    
    if(res) {
        g_tradesToday++;
        Print("✅ ORB TRADE: ", (dir==1?"BUY":"SELL"), " @ ", entry, " Lots:", lots, " Risk:$", InpFixedRiskAmount);
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
    
    double totalDist = MathAbs(tp - openP); // Distanza totale originale
    double currentProfitDist = (type==POSITION_TYPE_BUY) ? (bid - openP) : (openP - ask);
    double riskDist = MathAbs(openP - sl);
    
    // --- PARTIAL CLOSE + BE LOGIC (Sez. Liquidità/Risk) ---
    // Calcola R:R attuale raggiunto
    double currentRR = (riskDist > 0) ? (currentProfitDist / riskDist) : 0;
    
    // Se raggiungiamo il target di partial close (es. 1:1)
    if(currentRR >= InpPartialCloseRR && volume > 0) {
        double closeVol = volume * (InpPartialClosePct / 100.0);
        closeVol = MathFloor(closeVol / SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP)) * SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
        
        if(closeVol >= SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN)) {
            // Chiudi Parziale
            if(type==POSITION_TYPE_BUY) trade.PositionClosePartial(_Symbol, closeVol, 0);
            else trade.PositionClosePartial(_Symbol, closeVol, 0);
            
            // Sposta SL a Break Even sul resto
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
    } else {
        // Opzionale: resetta contatore loss se vinci? 
        // Per sicurezza massima, manteniamo il blocco giornaliero anche dopo una win se abbiamo già perso 2 volte.
        // g_lossesToday = 0; 
    }
}
//+------------------------------------------------------------------+