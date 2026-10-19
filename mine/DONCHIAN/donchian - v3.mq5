//+------------------------------------------------------------------+
//|                              SMC_Sweep_Bot_v5_HYBRID.mq5         |
//|                Donchian H1 Signal + M15 Aggressive Execution     |
//|                v5.0 - Higher Frequency, Higher Risk              |
//+------------------------------------------------------------------+
#property copyright "Qwen3.7 Assistant"
#property version   "5.00"
#property strict

#include <Trade\Trade.mqh>

enum TRADE_STATE { STATE_IDLE, STATE_WAITING_SETUP, STATE_IN_TRADE };

input group "=== Core Settings ==="
input int      Inp_Donchian_N = 20;
input int      Inp_Donchian_M = 10;
input int      Inp_EMA_Period = 100;
input int      Inp_ADX_Min = 22;              // Leggermente ridotto per più segnali
input double   Inp_Risk_Percent = 0.25;       // AUMENTATO
input double   Inp_SL_ATR_Mult = 1.2;         // RIDOTTO per M15
input double   Inp_BE_Trigger_R = 0.7;        // ANTICIPATO
input double   Inp_Trailing_ATR_Mult = 1.2;   // RIDOTTO
input double   Inp_Partial_TP_R = 0.8;        // ANTICIPATO
input double   Inp_Partial_TP_Percent = 50.0;

input group "=== Execution ==="
input bool     Inp_Use_Refined_Entry = true;
input ENUM_TIMEFRAMES Inp_Entry_TF = PERIOD_M15;
input int      Inp_Entry_Timeout_Minutes = 60; // 1 ORA invece di 3
input int      Inp_Pullback_EMA = 15;         // Più reattivo

input group "=== Filters ==="
input double   Inp_Max_Spread_ATR_Ratio = 0.5; // Aumentato per più trade
input bool     Inp_Filter_Session = false;     // DISABILITATO per 24h trading

input group "=== General ==="
input int      Inp_MagicNumber = 888;
input double   Inp_Slippage_Points = 15;      // Aumentato per execution aggressiva

CTrade trade;
int handle_ema_h4, handle_adx_h1, handle_atr_h1, handle_atr_m15, handle_ema_m15;
TRADE_STATE current_state = STATE_IDLE;
datetime signal_time = 0;
int signal_direction = 0;
double signal_breakout_level = 0, entry_price = 0, initial_risk_points = 0;
bool indicators_ready = false, partial_tp_taken = false;

int OnInit() {
   trade.SetExpertMagicNumber(Inp_MagicNumber);
   trade.SetDeviationInPoints((int)Inp_Slippage_Points);
   trade.SetTypeFilling(ORDER_FILLING_IOC); // Più aggressivo
   
   string sym = _Symbol;
   handle_ema_h4 = iMA(sym, PERIOD_H4, Inp_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   handle_adx_h1 = iADX(sym, PERIOD_H1, 14);
   handle_atr_h1 = iATR(sym, PERIOD_H1, 14);
   handle_atr_m15 = iATR(sym, Inp_Entry_TF, 14);
   handle_ema_m15 = iMA(sym, Inp_Entry_TF, Inp_Pullback_EMA, 0, MODE_EMA, PRICE_CLOSE);
   
   if(handle_ema_h4==INVALID_HANDLE || handle_adx_h1==INVALID_HANDLE || 
      handle_atr_h1==INVALID_HANDLE || handle_atr_m15==INVALID_HANDLE || handle_ema_m15==INVALID_HANDLE)
      return INIT_FAILED;
      
   indicators_ready = true;
   Print("EA v5.0 HYBRID AGGRESSIVE | Risk: ", Inp_Risk_Percent, "% | Timeout: ", Inp_Entry_Timeout_Minutes, "min");
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason) {
   IndicatorRelease(handle_ema_h4); IndicatorRelease(handle_adx_h1);
   IndicatorRelease(handle_atr_h1); IndicatorRelease(handle_atr_m15); IndicatorRelease(handle_ema_m15);
}

void OnTick() {
   if(!indicators_ready) return;
   
   bool has_pos = CheckOpenPosition();
   if(has_pos) { current_state = STATE_IN_TRADE; ManageOpenPositions(); return; }
   else if(current_state == STATE_IN_TRADE) { current_state = STATE_IDLE; signal_direction = 0; }
   
   static datetime last_bar = 0;
   datetime curr_bar = iTime(_Symbol, PERIOD_H1, 0);
   if(last_bar != curr_bar) { last_bar = curr_bar; if(current_state == STATE_IDLE) CheckForNewSignal(); }
   
   if(current_state == STATE_WAITING_SETUP && Inp_Use_Refined_Entry) CheckM15Setup();
}

bool CheckOpenPosition() {
   for(int i=0; i<PositionsTotal(); i++)
      if(PositionGetSymbol(i)==_Symbol && PositionGetInteger(POSITION_MAGIC)==Inp_MagicNumber) return true;
   return false;
}

bool IsSessionActive() {
   if(!Inp_Filter_Session) return true; // 24h trading
   MqlDateTime dt; TimeToStruct(TimeCurrent(), dt);
   int h = dt.hour;
   return (h >= 7 && h <= 22); // Sessione estesa se abilitata
}

void CheckForNewSignal() {
   if(!IsSessionActive()) return;
   
   double ema_h4[], adx_h1[], atr_h1[];
   ArraySetAsSeries(ema_h4, true); ArraySetAsSeries(adx_h1, true); ArraySetAsSeries(atr_h1, true);
   
   if(CopyBuffer(handle_ema_h4, 0, 0, 3, ema_h4) < 3) return;
   if(CopyBuffer(handle_adx_h1, 0, 0, 2, adx_h1) < 2) return;
   if(CopyBuffer(handle_atr_h1, 0, 0, 2, atr_h1) < 2) return;
   
   double ema = ema_h4[1], adx = adx_h1[1];
   bool slope_up = ema_h4[1] > ema_h4[2], slope_down = ema_h4[1] < ema_h4[2];
   
   double dh = GetDonchianHigh(PERIOD_H1, Inp_Donchian_N, 2);
   double dl = GetDonchianLow(PERIOD_H1, Inp_Donchian_N, 2);
   double close_prev = iClose(_Symbol, PERIOD_H1, 1);
   
   // Segnale semplificato: solo Donchian + EMA slope + ADX minimo
   bool long_sig = (close_prev > dh) && slope_up && (adx > Inp_ADX_Min);
   bool short_sig = (close_prev < dl) && slope_down && (adx > Inp_ADX_Min);
   
   if(long_sig) {
      current_state = STATE_WAITING_SETUP; signal_direction = 1;
      signal_time = TimeCurrent(); signal_breakout_level = dh;
      if(!Inp_Use_Refined_Entry) OpenPosition(1, atr_h1[1], false);
   } else if(short_sig) {
      current_state = STATE_WAITING_SETUP; signal_direction = -1;
      signal_time = TimeCurrent(); signal_breakout_level = dl;
      if(!Inp_Use_Refined_Entry) OpenPosition(-1, atr_h1[1], false);
   }
}

void CheckM15Setup() {
   // Timeout in MINUTI invece di ore
   int minutes_elapsed = (int)((TimeCurrent() - signal_time) / 60);
   if(minutes_elapsed >= Inp_Entry_Timeout_Minutes) {
      double atr_h1[]; ArraySetAsSeries(atr_h1, true);
      if(CopyBuffer(handle_atr_h1, 0, 0, 2, atr_h1) >= 2) 
         OpenPosition(signal_direction, atr_h1[1], false);
      current_state = STATE_IDLE; signal_direction = 0; return;
   }
   
   double ema_m15[], atr_m15[];
   ArraySetAsSeries(ema_m15, true); ArraySetAsSeries(atr_m15, true);
   if(CopyBuffer(handle_ema_m15, 0, 0, 2, ema_m15) < 2) return;
   if(CopyBuffer(handle_atr_m15, 0, 0, 2, atr_m15) < 2) return;
   
   double price = (signal_direction==1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double close_m15 = iClose(_Symbol, Inp_Entry_TF, 1);
   
   // Setup più permissivo: basta rottura barra precedente M15
   bool setup = false;
   if(signal_direction == 1) {
      setup = (close_m15 > iHigh(_Symbol, Inp_Entry_TF, 2));
   } else {
      setup = (close_m15 < iLow(_Symbol, Inp_Entry_TF, 2));
   }
   
   if(setup) OpenPosition(signal_direction, atr_m15[1], true);
}

void OpenPosition(int dir, double atr_val, bool refined) {
   if(!IsSessionActive()) { current_state = STATE_IDLE; signal_direction = 0; return; }
   
   double sl_dist = atr_val * Inp_SL_ATR_Mult;
   double risk_pct = Inp_Risk_Percent;
   
   double entry=0, sl=0; bool res=false;
   long spread_pts = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double spread_price = spread_pts * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   if(spread_price > atr_val * Inp_Max_Spread_ATR_Ratio) return;
   
   if(dir == 1) {
      entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = entry - sl_dist;
      double lot = CalcLot(sl_dist, risk_pct);
      if(lot > 0) res = trade.Buy(lot, _Symbol, entry, sl, 0, "Hybrid Long");
   } else {
      entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = entry + sl_dist;
      double lot = CalcLot(sl_dist, risk_pct);
      if(lot > 0) res = trade.Sell(lot, _Symbol, entry, sl, 0, "Hybrid Short");
   }
   
   if(res) {
      entry_price = entry;
      initial_risk_points = sl_dist / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      current_state = STATE_IN_TRADE;
      partial_tp_taken = false;
      Print("POSIZIONE APERTA: ", (dir==1?"LONG":"SHORT"), " @ ", entry, " Risk: ", risk_pct, "%");
   }
}

double CalcLot(double sl_dist, double risk_pct) {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * (risk_pct / 100.0);
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   if(tick_size==0 || tick_value==0) return 0;
   
   double lot = risk_money / ((sl_dist/tick_size) * tick_value);
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lot = MathFloor(lot/step)*step;
   return MathMax(min_lot, MathMin(max_lot, lot));
}

void ManageOpenPositions() {
   for(int i=0; i<PositionsTotal(); i++) {
      if(PositionGetSymbol(i)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=Inp_MagicNumber) continue;
      
      ulong ticket = PositionGetTicket(i);
      double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
      double pos_sl = PositionGetDouble(POSITION_SL);
      double pos_vol = PositionGetDouble(POSITION_VOLUME);
      long pos_type = PositionGetInteger(POSITION_TYPE);
      
      double price = (pos_type==POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      double profit_pts = MathAbs(price - pos_open) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      double r_curr = profit_pts / initial_risk_points;
      
      // TP Parziale anticipato
      if(!partial_tp_taken && r_curr >= Inp_Partial_TP_R) {
         double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
         double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
         double partial_vol = MathFloor((pos_vol * Inp_Partial_TP_Percent/100.0)/step)*step;
         if(partial_vol >= min_lot && (pos_vol-partial_vol) >= min_lot) {
            if(trade.PositionClosePartial(ticket, partial_vol)) partial_tp_taken = true;
         } else partial_tp_taken = true;
      }
      
      // Break-Even anticipato
      if(r_curr >= Inp_BE_Trigger_R) {
         if(pos_type==POSITION_TYPE_BUY && pos_sl < pos_open) trade.PositionModify(ticket, pos_open, 0);
         else if(pos_type==POSITION_TYPE_SELL && pos_sl > pos_open) trade.PositionModify(ticket, pos_open, 0);
      }
      
      // Trailing stretto
      double atr_h1[]; ArraySetAsSeries(atr_h1, true);
      if(CopyBuffer(handle_atr_h1, 0, 0, 2, atr_h1) < 2) continue;
      double atr_val = atr_h1[1];
      double donch_high = GetDonchianHigh(PERIOD_H1, Inp_Donchian_M, 2);
      double donch_low = GetDonchianLow(PERIOD_H1, Inp_Donchian_M, 2);
      
      pos_sl = PositionGetDouble(POSITION_SL);
      double new_sl = 0;
      
      if(pos_type == POSITION_TYPE_BUY) {
         new_sl = MathMax(price - atr_val*Inp_Trailing_ATR_Mult, donch_low);
         if(new_sl > pos_sl && new_sl < price) trade.PositionModify(ticket, new_sl, 0);
      } else {
         new_sl = MathMin(price + atr_val*Inp_Trailing_ATR_Mult, donch_high);
         if((new_sl < pos_sl || pos_sl==0) && new_sl > price) trade.PositionModify(ticket, new_sl, 0);
      }
   }
}

double GetDonchianHigh(ENUM_TIMEFRAMES tf, int period, int shift) {
   double max=0;
   for(int i=shift; i<shift+period; i++) { double h=iHigh(_Symbol,tf,i); if(h>max||i==shift) max=h; }
   return max;
}

double GetDonchianLow(ENUM_TIMEFRAMES tf, int period, int shift) {
   double min=999999;
   for(int i=shift; i<shift+period; i++) { double l=iLow(_Symbol,tf,i); if(l<min||i==shift) min=l; }
   return min;
}