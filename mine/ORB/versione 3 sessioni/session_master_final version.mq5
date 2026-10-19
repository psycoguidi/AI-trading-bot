//+------------------------------------------------------------------+
//|                               Tri_Session_Master_v4.1.mq5         |
//|                   Logic: Asia MeanRev + London/NY ORB            |
//|                   Risk: Fixed Amount + Robust PnL Tracking       |
//|   v4.1 FIXES:                                                     |
//|    - OnTradeTransaction: distingue chiusura parziale da totale   |
//|    - London/NY: rimossa la gate "currentHour==StartHour" che     |
//|      uccideva il breakout quando la finestra attraversa l'ora    |
//|    - Asia SL_Points di default alzato 200 -> 450                 |
//|    - ExecuteTrade: cattura POSITION_IDENTIFIER senza affidarsi   |
//|      al parsing del commento (fragile con DoubleToString)        |
//+------------------------------------------------------------------+
#property copyright "Tri-Session Master v4.1"
#property version   "4.10"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
//  INPUTS
// ═══════════════════════════════════════════
input group " Asian Session (Mean Reversion)"
input int      InpAsia_StartHour    = 1;     // 01:00 Broker Time
input int      InpAsia_EndRangeHour = 4;     // 04:00 Broker Time (End of Range Def)
input int      InpAsia_ForceCloseH  = 7;     // 07:30/08:00 approx Force Close
input int      InpAsia_BB_Period    = 20;
input double   InpAsia_BB_Dev       = 2.0;
input int      InpAsia_RSI_Period   = 14;
input double   InpAsia_RSI_OverBuy  = 70.0;
input double   InpAsia_RSI_OverSell = 30.0;
input double   InpAsia_SL_Points    = 450;   // v4.1: era 200, troppo stretto per XAUUSD
input int      InpAsia_ADX_Period   = 14;    // Filtro Trend
input double   InpAsia_ADX_Limit    = 25.0;  // Sopra questo valore, NO MeanRev
input bool     InpAsia_Enabled      = false;  // v4.1: kill-switch per isolare le altre sessioni

input group "London Session (ORB)"
input int      InpLon_StartHour     = 8;
input int      InpLon_StartMin      = 0;
input int      InpLon_DurationMin   = 60;
input bool     InpLon_UseDailyFilter= true;
input bool     InpLon_IgnoreBias    = true;  // Sblocca Londra per test
input int      InpLon_SwingLen      = 5;

input group "New York Session (ORB)"
input int      InpNY_StartHour      = 15;    // 15:00 or 16:00 depending on DST/Broker
input int      InpNY_StartMin       = 30;
input int      InpNY_DurationMin    = 15;
input bool     InpNY_UseDailyFilter = true;

input group "Execution & Global Risk"
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M5;
input double   InpFixedRiskAmount   = 50.0;
input double   InpRR_Target         = 3.0;
input double   InpPartialCloseRR    = 99;
input double   InpPartialClosePct   = 50.0;
input int      InpMaxLossesDay      = 3;
input int      InpMaxTradesDay      = 10;
input ulong    InpMagicNumber       = 20260909;
input bool     InpShowComments      = true;

// ═══════════════════════════════════════════
//  GLOBALS
// ══════════════════════════════════════════
CTrade trade;

int h_bbands, h_rsi, h_adx;

double g_asiaHigh = 0.0, g_asiaLow = 999999.0;
bool   g_asiaRangeDefined = false;

double g_lonHigh = 0.0, g_lonLow = 999999.0;
bool   g_lonRangeDefined = false;
bool   g_lonExecutedToday = false;     // v4.1: sostituisce il gate su currentHour

double g_nyHigh = 0.0, g_nyLow = 999999.0;
bool   g_nyRangeDefined = false;
bool   g_nyExecutedToday = false;      // v4.1: idem per coerenza/robustezza

int    g_dailyBias = 0;
double g_daily_swing_highs[];
double g_daily_swing_lows[];

datetime g_currentDay = 0;
int      g_tradesToday = 0;
int      g_lossesToday = 0;
bool     g_dayLocked = false;
long     g_countedLossIds[];

// Mappa Sessione -> PositionID per tracciamento PnL robusto
struct SessionMap {
    long   posId;
    string session;
};
SessionMap g_posSessionMap[];

// Performance Tracking
double g_profitAsia = 0, g_profitLon = 0, g_profitNY = 0;
int    g_winsAsia = 0, g_winsLon = 0, g_winsNY = 0;
int    g_totalAsia = 0, g_totalLon = 0, g_totalNY = 0;

// ═══════════════════════════════════════════
//  INIT & DEINIT
// ══════════════════════════════════════════
int OnInit() {
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    h_bbands = iBands(_Symbol, PERIOD_M15, InpAsia_BB_Period, 0, InpAsia_BB_Dev, PRICE_CLOSE);
    h_rsi    = iRSI(_Symbol, PERIOD_M15, InpAsia_RSI_Period, PRICE_CLOSE);
    h_adx    = iADX(_Symbol, PERIOD_H1, InpAsia_ADX_Period);

    if(h_bbands == INVALID_HANDLE || h_rsi == INVALID_HANDLE || h_adx == INVALID_HANDLE) {
        Print("Errore inizializzazione indicatori");
        return(INIT_FAILED);
    }

    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    ArrayResize(g_posSessionMap, 0);

    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));

    RestoreDailyStats();
    Print("Tri-Session Master v4.1 Ready.");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
    PrintFinalReport();
    Comment("");
    if(h_bbands != INVALID_HANDLE) IndicatorRelease(h_bbands);
    if(h_rsi != INVALID_HANDLE) IndicatorRelease(h_rsi);
    if(h_adx != INVALID_HANDLE) IndicatorRelease(h_adx);
}

// ═══════════════════════════════════════════
//  UTILS & STATS
// ═══════════════════════════════════════════
void AddToArray(double &arr[], double val) {
    int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val;
}
void ShiftArray(double &arr[]) {
    if(ArraySize(arr)<=1){ArrayResize(arr,0);return;}
    for(int i=0;i<ArraySize(arr)-1;i++) arr[i]=arr[i+1];
    ArrayResize(arr,ArraySize(arr)-1);
}
void AddToLongArray(long &arr[], long val) {
    int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val;
}

void RegisterPosition(long posId, string session) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) {
        if(g_posSessionMap[i].posId == posId) ArrayRemove(g_posSessionMap, i, 1);
    }
    int s = ArraySize(g_posSessionMap);
    ArrayResize(g_posSessionMap, s+1);
    g_posSessionMap[s].posId = posId;
    g_posSessionMap[s].session = session;
    Print("DEBUG RegisterPosition: posId=", posId, " session=", session, " mapSize=", ArraySize(g_posSessionMap));
}

string GetSessionByPosId(long posId) {
    for(int i=0; i<ArraySize(g_posSessionMap); i++)
        if(g_posSessionMap[i].posId == posId) return g_posSessionMap[i].session;
    return "UNKNOWN";
}

void UnregisterPosition(long posId) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--)
        if(g_posSessionMap[i].posId == posId) { ArrayRemove(g_posSessionMap, i, 1); return; }
}

// v4.1: la posizione esiste ancora "live" tra quelle aperte? (serve a distinguere partial da full close)
bool IsPositionStillOpen(long posId) {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong t = PositionGetTicket(i);
        if(t > 0 && (long)PositionGetInteger(POSITION_IDENTIFIER) == posId) return true;
    }
    return false;
}

bool IsIdAlreadyCounted(long posId) {
    for(int i=0; i<ArraySize(g_countedLossIds); i++) if(g_countedLossIds[i] == posId) return true;
    return false;
}

bool HasOpenPosition() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol &&
           PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) return true;
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
            bool stillOpen = IsPositionStillOpen(posId);
            if(!stillOpen && profit < 0 && !IsIdAlreadyCounted(posId)) {
                g_lossesToday++;
                AddToLongArray(g_countedLossIds, posId);
            }
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
    Print("TRI-SESSION MASTER v4.1 FINAL REPORT");
    Print("--------------------------------------------------");
    PrintSessionStats("ASIA  ", g_profitAsia, g_winsAsia, g_totalAsia);
    PrintSessionStats("LONDON", g_profitLon, g_winsLon, g_totalLon);
    PrintSessionStats("NY    ", g_profitNY, g_winsNY, g_totalNY);
    Print("--------------------------------------------------");
}

// ══════════════════════════════════════════
//  MAIN LOGIC
// ═══════════════════════════════════════════
void OnTick() {
    ManagePositions();

    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;

    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(today != g_currentDay) {
        g_currentDay = today;
        g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
        g_asiaRangeDefined = false; g_lonRangeDefined = false; g_nyRangeDefined = false;
        g_lonExecutedToday = false; g_nyExecutedToday = false;   // v4.1
        g_asiaHigh = 0.0; g_asiaLow = 999999.0;
        g_lonHigh = 0.0; g_lonLow = 999999.0;
        g_nyHigh = 0.0; g_nyLow = 999999.0;
        ArrayResize(g_daily_swing_highs, 0); ArrayResize(g_daily_swing_lows, 0);
        ArrayResize(g_countedLossIds, 0);
        // v4.2 FIX CRITICO: NON svuotare g_posSessionMap qui. Una posizione può restare
        // aperta a cavallo di mezzanotte (specialmente London/NY con RR 3.0): svuotare la
        // mappa ogni giorno cancellava il collegamento posizione->sessione per qualsiasi
        // trade ancora vivo, causando "SESSIONE NON TROVATA" alla chiusura reale.
        // La mappa si autogestisce già con RegisterPosition/UnregisterPosition.
    }

    if(g_dayLocked) { if(InpShowComments) Comment("STOP TRADING: Max Losses Reached"); return; }
    if(g_tradesToday >= InpMaxTradesDay) return;

    static datetime lastD1 = 0;
    if(iTime(_Symbol, PERIOD_D1, 0) != lastD1) {
        lastD1 = iTime(_Symbol, PERIOD_D1, 0);
        double ph = GetPivot(PERIOD_D1, InpLon_SwingLen, InpLon_SwingLen+1, true);
        double pl = GetPivot(PERIOD_D1, InpLon_SwingLen, InpLon_SwingLen+1, false);
        if(ph != EMPTY_VALUE) { AddToArray(g_daily_swing_highs, ph); if(ArraySize(g_daily_swing_highs)>10) ShiftArray(g_daily_swing_highs); }
        if(pl != EMPTY_VALUE) { AddToArray(g_daily_swing_lows, pl); if(ArraySize(g_daily_swing_lows)>10) ShiftArray(g_daily_swing_lows); }
        g_dailyBias = GetDailyBias();
    }

    int currentHour = dt.hour;

    // --- ASIAN SESSION LOGIC ---
    if(InpAsia_Enabled && currentHour >= InpAsia_StartHour && currentHour < InpAsia_ForceCloseH) {
        if(currentHour < InpAsia_EndRangeHour) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_asiaHigh) g_asiaHigh = h;
            if(l < g_asiaLow) g_asiaLow = l;
            g_asiaRangeDefined = true;
        } else if(g_asiaRangeDefined && !HasOpenPosition()) {
            double adxVal[1];
            bool adxOk = (CopyBuffer(h_adx, 0, 1, 1, adxVal) > 0 && adxVal[0] < InpAsia_ADX_Limit);

            if(adxOk) {
                double bbUpper[1], bbLower[1], bbMiddle[1], rsiVal[1];
                if(CopyBuffer(h_bbands, 0, 1, 1, bbMiddle) <= 0) return;
                if(CopyBuffer(h_bbands, 1, 1, 1, bbUpper) <= 0) return;
                if(CopyBuffer(h_bbands, 2, 1, 1, bbLower) <= 0) return;
                if(CopyBuffer(h_rsi, 0, 1, 1, rsiVal) <= 0) return;

                double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
                double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);

                if(bid < g_asiaLow && bid < bbLower[0] && rsiVal[0] < InpAsia_RSI_OverSell)
                    ExecuteTrade(1, ask, ask - InpAsia_SL_Points * SymbolInfoDouble(_Symbol, SYMBOL_POINT), bbMiddle[0], "ASIA");
                else if(ask > g_asiaHigh && ask > bbUpper[0] && rsiVal[0] > InpAsia_RSI_OverBuy)
                    ExecuteTrade(-1, bid, bid + InpAsia_SL_Points * SymbolInfoDouble(_Symbol, SYMBOL_POINT), bbMiddle[0], "ASIA");
            }
        }
    }

    if(currentHour == InpAsia_ForceCloseH && dt.min >= 30) CloseAllByComment("ASIA");

    // --- LONDON SESSION LOGIC (v4.1: niente più gate su currentHour==StartHour) ---
    datetime lonStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpLon_StartHour, InpLon_StartMin));
    datetime lonEnd   = lonStart + (InpLon_DurationMin * 60);

    if(!g_lonExecutedToday && TimeCurrent() >= lonStart) {
        if(TimeCurrent() < lonEnd) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_lonHigh) g_lonHigh = h;
            if(l < g_lonLow) g_lonLow = l;
            g_lonRangeDefined = true;
        } else if(g_lonRangeDefined && !HasOpenPosition()) {
            double m5_close = iClose(_Symbol, InpEntryTF, 1);
            bool bull = (m5_close > g_lonHigh);
            bool bear = (m5_close < g_lonLow);

            bool biasBlock = false;
            if(!InpLon_IgnoreBias && InpLon_UseDailyFilter) {
                if(bull && g_dailyBias == -1) biasBlock = true;
                if(bear && g_dailyBias == 1) biasBlock = true;
                if(g_dailyBias == 0) biasBlock = true;
            }

            if(biasBlock) {
                if(InpShowComments) Print("LONDON BLOCKED: Bias=", g_dailyBias, " | IgnoreBias=", InpLon_IgnoreBias);
            } else {
                if(bull) ExecuteTrade(1, m5_close, g_lonLow, 0, "LONDON");
                else if(bear) ExecuteTrade(-1, m5_close, g_lonHigh, 0, "LONDON");
            }
            g_lonExecutedToday = true; // eseguito (o valutato) esattamente una volta al giorno
        } else if(!g_lonRangeDefined) {
            // range non definito (es. EA avviato dopo lonEnd): non ritentare più oggi
            g_lonExecutedToday = true;
        }
    }

    // --- NEW YORK SESSION LOGIC (stesso pattern, per coerenza e robustezza) ---
    datetime nyStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpNY_StartHour, InpNY_StartMin));
    datetime nyEnd   = nyStart + (InpNY_DurationMin * 60);

    if(!g_nyExecutedToday && TimeCurrent() >= nyStart) {
        if(TimeCurrent() < nyEnd) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_nyHigh) g_nyHigh = h;
            if(l < g_nyLow) g_nyLow = l;
            g_nyRangeDefined = true;
        } else if(g_nyRangeDefined && !HasOpenPosition()) {
            double m5_close = iClose(_Symbol, InpEntryTF, 1);
            bool bull = (m5_close > g_nyHigh);
            bool bear = (m5_close < g_nyLow);

            if(InpNY_UseDailyFilter) {
                if(bull && g_dailyBias == -1) bull = false;
                if(bear && g_dailyBias == 1) bear = false;
            }

            if(bull) ExecuteTrade(1, m5_close, g_nyLow, 0, "NY");
            else if(bear) ExecuteTrade(-1, m5_close, g_nyHigh, 0, "NY");
            g_nyExecutedToday = true;
        } else if(!g_nyRangeDefined) {
            g_nyExecutedToday = true;
        }
    }

    if(InpShowComments) {
        Comment(StringFormat("Tri-Session v4.1 | Bias: %s | Asia H/L: %.2f/%.2f | Lon H/L: %.2f/%.2f | Trades: %d",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")), g_asiaHigh, g_asiaLow, g_lonHigh, g_lonLow, g_tradesToday));
    }
}

// ══════════════════════════════════════════
//  EXECUTION & MANAGEMENT
// ══════════════════════════════════════════
void ExecuteTrade(int dir, double entry, double sl, double tp_dynamic, string session) {
    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    double sl_final = NormalizeDouble(sl, digits);
    double tp_final = 0;

    if(session == "ASIA") tp_final = NormalizeDouble(tp_dynamic, digits);
    else {
        double dist = MathAbs(entry - sl);
        if(dist == 0) return;
        tp_final = NormalizeDouble((dir == 1) ? entry + (dist * InpRR_Target) : entry - (dist * InpRR_Target), digits);
    }

    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double profitPerLot = 0;

    bool calcRes = false;
    if(dir == 1) calcRes = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, sl_final, profitPerLot);
    else calcRes = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, sl_final, profitPerLot);

    if(!calcRes || profitPerLot >= 0) return;

    double lots = MathFloor((InpFixedRiskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
    if(lots < minLot) lots = minLot;

    string cmt = StringFormat("%s|R:%.2f", session, InpFixedRiskAmount);
    bool res = false;

    // v4.1: snapshot dei ticket PRIMA dell'invio, per individuare il ticket nuovo senza
    // dipendere dal parsing del commento (più robusto se il broker tronca/modifica i commenti)
    ulong preTickets[];
    int preCount = PositionsTotal();
    ArrayResize(preTickets, preCount);
    for(int i=0; i<preCount; i++) preTickets[i] = PositionGetTicket(i);

    if(dir == 1) res = trade.Buy(lots, _Symbol, entry, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, _Symbol, entry, sl_final, tp_final, cmt);

    if(res) {
        g_tradesToday++;

        long posId = 0;
        for(int i=PositionsTotal()-1; i>=0; i--) {
            ulong t = PositionGetTicket(i);
            if(t == 0) continue;
            if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
            if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

            bool wasPreExisting = false;
            for(int j=0; j<preCount; j++) if(preTickets[j] == t) { wasPreExisting = true; break; }
            if(!wasPreExisting) { posId = (long)PositionGetInteger(POSITION_IDENTIFIER); break; }
        }

        Print("DEBUG ExecuteTrade: nuova posizione individuata posId=", posId, " session=", session, " preCount=", preCount, " postCount=", PositionsTotal());
        if(posId > 0) RegisterPosition(posId, session);
        else Print("ATTENZIONE: impossibile identificare la nuova posizione per la sessione ", session);

        if(session == "ASIA") g_totalAsia++;
        else if(session == "LONDON") g_totalLon++;
        else if(session == "NY") g_totalNY++;

        Print("TRADE OPENED: ", session, " | PosID: ", posId);
    }
}

void ManagePositions() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket <= 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        string comment = PositionGetString(POSITION_COMMENT);
        double openP = PositionGetDouble(POSITION_PRICE_OPEN);
        double sl = PositionGetDouble(POSITION_SL);
        double tp = PositionGetDouble(POSITION_TP);
        double volume = PositionGetDouble(POSITION_VOLUME);
        int type = (int)PositionGetInteger(POSITION_TYPE);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);

        if(StringFind(comment, "ASIA") < 0 && tp > 0) {
            double originalRiskDist = MathAbs(tp - openP) / InpRR_Target;
            if(originalRiskDist > 0) {
                double currentProfitDist = (type==POSITION_TYPE_BUY) ? (bid - openP) : (openP - ask);
                double currentRR = currentProfitDist / originalRiskDist;

                bool partialAlreadyDone = (StringFind(comment, "PCD") >= 0);
                bool isBE = (type == POSITION_TYPE_BUY) ? (sl >= openP - SymbolInfoDouble(_Symbol, SYMBOL_POINT)) : (sl <= openP + SymbolInfoDouble(_Symbol, SYMBOL_POINT));

                if(currentRR >= InpPartialCloseRR && !partialAlreadyDone && !isBE) {
                    double closeVol = volume * (InpPartialClosePct / 100.0);
                    double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
                    closeVol = MathFloor(closeVol / step) * step;

                    double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
                    if(closeVol >= minVol && (volume - closeVol) >= minVol) {
                        if(trade.PositionClosePartial(ticket, closeVol, 0)) {
                            double newSL = openP;
                            if(type==POSITION_TYPE_BUY && sl < openP) trade.PositionModify(ticket, newSL, tp);
                            if(type==POSITION_TYPE_SELL && sl > openP) trade.PositionModify(ticket, newSL, tp);
                        }
                    }
                }
            }
        }
    }
}

void CloseAllByComment(string keyword) {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
           PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber &&
           StringFind(PositionGetString(POSITION_COMMENT), keyword) >= 0) {
            trade.PositionClose(ticket);
        }
    }
}

// v4.1: distingue chiusura PARZIALE da chiusura TOTALE prima di scartare la mappatura.
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;

    long posId = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
    int dealEntry = (int)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);

    if(dealEntry == DEAL_ENTRY_OUT) {

        bool stillOpenNow = IsPositionStillOpen(posId);
        Print("DEBUG OnTradeTransaction: deal=", trans.deal, " posId=", posId,
              " stillOpen=", stillOpenNow, " mapSession=", GetSessionByPosId(posId),
              " mapSize=", ArraySize(g_posSessionMap));

        // v4.1 FIX CRITICO: se la posizione esiste ANCORA (partial close), non è una chiusura
        // definitiva. Non tocchiamo la mappa sessione e non contiamo nulla adesso: aspettiamo
        // il vero DEAL_ENTRY_OUT finale.
        if(stillOpenNow) return;

        string session = GetSessionByPosId(posId);
        if(session == "UNKNOWN") {
            Print("DEBUG OnTradeTransaction: SESSIONE NON TROVATA per posId=", posId, " -> profitto non contato");
            return;
        }

        // Sommiamo TUTTI i deal OUT per questa posizione (gestisce i parziali automaticamente,
        // ora che sappiamo che la posizione è DAVVERO chiusa)
        // v4.2 FIX CRITICO REALE: HistoryDealsTotal()/HistoryDealGetTicket() operano solo
        // sull'ultimo intervallo selezionato con HistorySelect(). Quell'intervallo era stato
        // fissato UNA VOLTA in RestoreDailyStats() all'avvio del test (finestra iniziale
        // strettissima) e non veniva mai aggiornato: qui sotto il ciclo non trovava quasi mai
        // i deal della posizione appena chiusa, quindi totalPnL restava sempre 0.00 per
        // QUALSIASI sessione. HistorySelectByPosition() seleziona lo storico dei deal per
        // questo specifico posId, indipendentemente da range di date.
        if(!HistorySelectByPosition(posId)) {
            Print("DEBUG OnTradeTransaction: HistorySelectByPosition fallita per posId=", posId);
            return;
        }

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
        Print("DEBUG OnTradeTransaction: totalPnL calcolato per posId=", posId, " session=", session, " = ", DoubleToString(totalPnL,2));

        if(!IsIdAlreadyCounted(posId)) {
            if(session == "ASIA") { g_profitAsia += totalPnL; if(totalPnL > 0) g_winsAsia++; }
            else if(session == "LONDON") { g_profitLon += totalPnL; if(totalPnL > 0) g_winsLon++; }
            else if(session == "NY") { g_profitNY += totalPnL; if(totalPnL > 0) g_winsNY++; }

            AddToLongArray(g_countedLossIds, posId); // flag "già processata", vinta o persa
            if(totalPnL < 0) {
                g_lossesToday++;
                if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;
            }

            UnregisterPosition(posId);
        }
    }
}
//+------------------------------------------------------------------+