//+------------------------------------------------------------------+
//|                                      ORB_NY_Open_v2_RiskFixed.mq5|
//|          Logic: Fixed Risk (OrderCalcProfit) + Robust Tracking   |
//|      Conforme a: Prompt Analista SMC Personalizzato (v2.0)       |
//+------------------------------------------------------------------+
#property copyright "ORB Strategy v2.3 - Risk Unlocked"
#property version   "2.30"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS (INVARIATI - Sez. Strategia)
// ═══════════════════════════════════════════
input group "🕒 Sessioni & Range (NY Time)"
input int      InpORB_StartHour   = 15;    
input int      InpORB_StartMin    = 30;    
input int      InpORB_DurationMin = 15;    

input group "📈 Struttura & Bias (DAILY)"
input bool     InpUseDailyFilter  = true;  
input int      InpSwingLen_Daily  = 5;     

input group "⚡ Entry & Execution (M5)"
input ENUM_TIMEFRAMES InpEntryTF  = PERIOD_M5;
input bool     InpRequireBodyClose= true;  
input double   InpSLBuffer_Points = 30;    

input group "⚖️ Risk & Money Management"
input double   InpFixedRiskAmount = 50.0;   // RISCHIO TARGET (es. 2€) - Non più un limite rigido
input double   InpRR_Target       = 3.0;   // Target Finale R:R
input double   InpPartialCloseRR  = 1.5;   // Chiudi parziale a questo R:R
input double   InpPartialClosePct = 50.0;  // % da chiudere al TP1
input int      InpMaxLossesDay    = 1;     // MAX TRADE chiusi in perdita/giorno -> STOP
input int      InpMaxTradesDay    = 5;     // Max NUOVE POSIZIONI aperte/giorno

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

int    g_dailyBias = 0; 
double g_daily_swing_highs[];
double g_daily_swing_lows[];

datetime g_currentDay = 0;
int      g_tradesToday = 0;
int      g_lossesToday = 0;
bool     g_dayLocked = false;

long g_countedLossIds[];

// ═══════════════════════════════════════════
// 🚀 INIT
// ═══════════════════════════════════════════
int OnInit() {
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);
    
    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    
    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    
    RestoreDailyStats();
    
    Print("✅ ORB v2.3 Ready. Target Risk: ", DoubleToString(InpFixedRiskAmount, 2), " (Soft Limit) | Max Loss/Day: ", IntegerToString(InpMaxLossesDay));
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) { Comment(""); }

// ═══════════════════════════════════════════
// 🔧 UTILS & RISK ENGINE
// ═══════════════════════════════════════════
void AddToArray(double &arr[], double val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }
void ShiftArray(double &arr[]) { if(ArraySize(arr)<=1){ArrayResize(arr,0);return;} for(int i=0;i<ArraySize(arr)-1;i++) arr[i]=arr[i+1]; ArrayResize(arr,ArraySize(arr)-1); }

void AddToLongArray(long &arr[], long val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }

bool IsIdAlreadyCounted(long posId) {
    for(int i=0; i<ArraySize(g_countedLossIds); i++) {
        if(g_countedLossIds[i] == posId) return true;
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
    g_tradesToday = 0;
    g_lossesToday = 0;
    g_dayLocked = false;
    ArrayResize(g_countedLossIds, 0);
    
    if(!HistorySelect(todayStart, TimeCurrent())) return;
    
    int totalDeals = HistoryDealsTotal();
    
    for(int i=0; i<totalDeals; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
        if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
        
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) == DEAL_ENTRY_IN) {
            g_tradesToday++;
        }
    }
    
    long posIds[];
    double posPnL[];
    
    for(int i=0; i<totalDeals; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
        if(HistoryDealGetString(ticket, DEAL_SYMBOL) != _Symbol) continue;
        if(HistoryDealGetInteger(ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;
        
        long posId = HistoryDealGetInteger(ticket, DEAL_POSITION_ID);
        double profit = HistoryDealGetDouble(ticket, DEAL_PROFIT) + 
                        HistoryDealGetDouble(ticket, DEAL_SWAP) + 
                        HistoryDealGetDouble(ticket, DEAL_COMMISSION);
        
        int idx = -1;
        for(int k=0; k<ArraySize(posIds); k++) { if(posIds[k] == posId) { idx = k; break; } }
        
        if(idx == -1) {
            int s = ArraySize(posIds);
            ArrayResize(posIds, s+1); ArrayResize(posPnL, s+1);
            posIds[s] = posId; posPnL[s] = profit;
        } else {
            posPnL[idx] += profit;
        }
    }
    
    for(int k=0; k<ArraySize(posPnL); k++) {
        bool stillOpen = false;
        for(int p=PositionsTotal()-1; p>=0; p--) {
            ulong pticket = PositionGetTicket(p);
            if(pticket > 0 && PositionGetInteger(POSITION_IDENTIFIER) == posIds[k]) {
                stillOpen = true; break;
            }
        }
        
        if(!stillOpen && posPnL[k] < 0) {
            g_lossesToday++;
            AddToLongArray(g_countedLossIds, posIds[k]);
        }
    }
    
    if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;
    if(g_tradesToday > InpMaxTradesDay) g_tradesToday = InpMaxTradesDay;
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
    ManagePositions(); 
    
    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;
    
    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));
    
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
        ArrayResize(g_countedLossIds, 0);
    }
    
    if(g_dayLocked) {
        if(InpShowComments) Comment("🛑 STOP TRADING: Max Losses Reached (", IntegerToString(g_lossesToday), "/", IntegerToString(InpMaxLossesDay), ")");
        return;
    }
    if(g_lossesToday >= InpMaxLossesDay) { g_dayLocked = true; return; }
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
    
    datetime orbStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpORB_StartHour, InpORB_StartMin));
    datetime orbEnd = orbStart + (InpORB_DurationMin * 60);
    
    if(TimeCurrent() >= orbStart && TimeCurrent() < orbEnd) {
        double h = iHigh(_Symbol, PERIOD_M1, 1);
        double l = iLow(_Symbol, PERIOD_M1, 1);
        if(h > g_orbHigh) g_orbHigh = h;
        if(l < g_orbLow) g_orbLow = l;
        g_orbDefined = true;
    }
    
    if(g_orbDefined && TimeCurrent() >= orbEnd && g_tradesToday < InpMaxTradesDay) {
        double m5_close = iClose(_Symbol, InpEntryTF, 1);
        double m5_open = iOpen(_Symbol, InpEntryTF, 1);
        
        bool bullBreakout = (m5_close > g_orbHigh) && (!InpRequireBodyClose || m5_close > m5_open);
        bool bearBreakout = (m5_close < g_orbLow) && (!InpRequireBodyClose || m5_close < m5_open);
        
        if(InpUseDailyFilter) {
            if(bullBreakout && g_dailyBias == -1) bullBreakout = false;
            if(bearBreakout && g_dailyBias == 1) bearBreakout = false;
        }
        
        if(bullBreakout) ExecuteTrade(1, m5_close, g_orbLow);
        else if(bearBreakout) ExecuteTrade(-1, m5_close, g_orbHigh);
    }
    
    if(InpShowComments) {
        string biasStr = (g_dailyBias==1)?"DAILY BULL":(g_dailyBias==-1)?"DAILY BEAR":"NEUTRAL";
        string orbStr = g_orbDefined ? StringFormat("H:%.2f L:%.2f", g_orbHigh, g_orbLow) : "Defining...";
        Comment(StringFormat("ORB v2.3 | %s | Range: %s | Trades: %d | Losses: %d/%d", 
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
    
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
    entry = NormalizeDouble(entry, digits);
    sl = NormalizeDouble(sl, digits);
    tp_final = NormalizeDouble(tp_final, digits);
    
    double distPrice = MathAbs(entry - sl);
    if(distPrice == 0) { Print("❌ TRADE RIFIUTATO: Distanza SL nulla."); return; }
    
    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double maxLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    
    if(minLot == 0) minLot = 0.01;
    if(stepLot == 0) stepLot = 0.01;
    
    double profitPerLot = 0;
    
    if(dir == 1) {
        if(!OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, sl, profitPerLot)) {
            Print("❌ Errore OrderCalcProfit."); return;
        }
    } else {
        if(!OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, sl, profitPerLot)) {
            Print("❌ Errore OrderCalcProfit."); return;
        }
    }
    
    if(profitPerLot >= 0) { Print("❌ Profitto positivo su SL."); return; }
    
    double rawLots = InpFixedRiskAmount / MathAbs(profitPerLot);
    double lots = MathFloor(rawLots / stepLot) * stepLot;
    
    // --- LOGICA LOTTO MINIMO (MODIFICATA: NESSUN BLOCCO PER RISCHIO SUPERIORE) ---
    if(lots < minLot) {
        lots = minLot; // Forza al minimo assoluto
        
        // Calcola rischio reale per il log (informativo, non bloccante)
        double forcedRisk = 0;
        bool calcResult = false;
        if(dir == 1) 
            calcResult = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, lots, entry, sl, forcedRisk);
        else 
            calcResult = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, lots, entry, sl, forcedRisk);
        
        if(calcResult) {
            Print("⚠️ INFO: Lotto forzato a Min Lot (", DoubleToString(minLot, 2), "). Rischio Reale: ", DoubleToString(MathAbs(forcedRisk), 2), "€ (Target: ", InpFixedRiskAmount, "€)");
        } else {
            Print("⚠️ ATTENZIONE: Impossibile calcolare il rischio con lotto minimo forzato.");
        }
    }
    
    if(lots > maxLot) lots = maxLot;
    
    // Ricalcolo rischio finale per il log
    double actualRisk = 0;
    bool riskCalcResult = false;
    if(dir == 1) 
        riskCalcResult = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, lots, entry, sl, actualRisk);
    else 
        riskCalcResult = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, lots, entry, sl, actualRisk);
    
    if(riskCalcResult) {
        actualRisk = MathAbs(actualRisk);
    } else {
        Print("⚠️ ATTENZIONE: Impossibile calcolare il rischio finale per il logging.");
        actualRisk = 0;
    }
    
    // FIX #15, #16: Validazione SL/TP (Stop Level + Freeze Level)
    double stopsLevel = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL) * point;
    double freezeLevel = (double)SymbolInfoInteger(_Symbol, SYMBOL_TRADE_FREEZE_LEVEL) * point;
    double minDist = MathMax(stopsLevel, freezeLevel);
    
    if(minDist > 0) {
        if(MathAbs(entry - sl) < minDist) {
            Print("❌ TRADE RIFIUTATO: Distanza SL troppo breve."); return;
        }
        if(MathAbs(tp_final - entry) < minDist) {
            Print("❌ TRADE RIFIUTATO: Distanza TP troppo breve."); return;
        }
    }
    
    // --- ESECUZIONE ---
    bool res = false;
    string cmt = StringFormat("ORBv2.3|NPC|R:%.2f", InpFixedRiskAmount);
    
    if(dir == 1) res = trade.Buy(lots, _Symbol, entry, sl, tp_final, cmt);
    else res = trade.Sell(lots, _Symbol, entry, sl, tp_final, cmt);
    
    if(res) {
        g_tradesToday++;
        Print("✅ ORB TRADE: ", (dir==1?"BUY":"SELL"), 
              " | Lots: ", DoubleToString(lots, 4), 
              " | Actual Risk: ", DoubleToString(actualRisk, 2), "€",
              " | Entry: ", DoubleToString(entry, digits), 
              " | SL: ", DoubleToString(sl, digits), 
              " | TP: ", DoubleToString(tp_final, digits));
    } else {
        Print("❌ ERRORE ESECUZIONE: ", trade.ResultRetcodeDescription());
    }
}

void ManagePositions() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket <= 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;
        
        double openP = PositionGetDouble(POSITION_PRICE_OPEN);
        double sl = PositionGetDouble(POSITION_SL);
        double tp = PositionGetDouble(POSITION_TP);
        double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
        double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        int type = (int)PositionGetInteger(POSITION_TYPE);
        double volume = PositionGetDouble(POSITION_VOLUME);
        string comment = PositionGetString(POSITION_COMMENT);
        
        double originalRiskDist = 0;
        if(InpRR_Target > 0 && tp > 0) {
            originalRiskDist = MathAbs(tp - openP) / InpRR_Target;
        }
        
        if(originalRiskDist <= 0) continue;
        
        double currentProfitDist = (type==POSITION_TYPE_BUY) ? (bid - openP) : (openP - ask);
        double currentRR = currentProfitDist / originalRiskDist;
        
        bool partialAlreadyDone = (StringFind(comment, "PCD") >= 0);
        bool isBE = (type == POSITION_TYPE_BUY) ? (sl >= openP - SymbolInfoDouble(_Symbol, SYMBOL_POINT)) : (sl <= openP + SymbolInfoDouble(_Symbol, SYMBOL_POINT));
        
        if(currentRR >= InpPartialCloseRR && !partialAlreadyDone && !isBE) {
            
            double closeVol = volume * (InpPartialClosePct / 100.0);
            double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            closeVol = MathFloor(closeVol / step) * step;
            closeVol = NormalizeDouble(closeVol, 2);
            
            double minVol = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            
            if(closeVol >= minVol && (volume - closeVol) >= minVol) {
                bool closed = false;
                if(type==POSITION_TYPE_BUY) closed = trade.PositionClosePartial(ticket, closeVol, 0);
                else closed = trade.PositionClosePartial(ticket, closeVol, 0);
                
                if(closed) {
                    double newSL = openP;
                    bool beModified = false;
                    if(type==POSITION_TYPE_BUY && sl < openP) beModified = trade.PositionModify(ticket, newSL, tp);
                    if(type==POSITION_TYPE_SELL && sl > openP) beModified = trade.PositionModify(ticket, newSL, tp);
                    
                    Print("🔶 PARTIAL CLOSE: Ticket ", IntegerToString(ticket), " | Vol: ", DoubleToString(closeVol, 2), " | BE: ", (beModified?"OK":"FAIL"));
                }
            }
        }
    }
}

void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    
    long dealMagic = HistoryDealGetInteger(trans.deal, DEAL_MAGIC);
    if(dealMagic != (long)InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_ENTRY) != DEAL_ENTRY_OUT) return;
    
    long posId = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
    if(IsIdAlreadyCounted(posId)) return;
    
    bool stillOpen = false;
    for(int p=PositionsTotal()-1; p>=0; p--) {
        ulong pticket = PositionGetTicket(p);
        if(pticket > 0) {
            if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
               PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) {
                if(PositionGetInteger(POSITION_IDENTIFIER) == posId) {
                    stillOpen = true; break;
                }
            }
        }
    }
    
    if(stillOpen) return;
    
    datetime todayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    if(!HistorySelect(todayStart, TimeCurrent())) return;
    
    double totalPnL = 0;
    int dealsCount = HistoryDealsTotal();
    for(int i=0; i<dealsCount; i++) {
        ulong dticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(dticket, DEAL_POSITION_ID) == posId &&
           HistoryDealGetInteger(dticket, DEAL_MAGIC) == (long)InpMagicNumber) {
            totalPnL += HistoryDealGetDouble(dticket, DEAL_PROFIT) + 
                        HistoryDealGetDouble(dticket, DEAL_SWAP) + 
                        HistoryDealGetDouble(dticket, DEAL_COMMISSION);
        }
    }
    
    if(totalPnL < 0) {
        g_lossesToday++;
        AddToLongArray(g_countedLossIds, posId);
        
        Print("📉 LOSS: PosID ", IntegerToString(posId), " | PnL: ", DoubleToString(totalPnL, 2), " | Tot Losses: ", IntegerToString(g_lossesToday));
        
        if(g_lossesToday >= InpMaxLossesDay) {
            g_dayLocked = true;
            Print("🛑 CIRCUIT BREAKER: Stop trading oggi.");
        }
    }
}
//+------------------------------------------------------------------+