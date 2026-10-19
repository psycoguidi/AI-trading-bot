//+------------------------------------------------------------------+
//|                              SMC_Sweep_Bot_v4_DONCHIAN.mq5       |
//|                Strategia Donchian Breakout - Analisi H1/H4       |
//|                v4.05 - Adaptive Risk + Pausa 12h                 |
//+------------------------------------------------------------------+
#property copyright "Qwen3.7 Assistant"
#property link      ""
#property version   "4.05"
#property strict

#include <Trade\Trade.mqh>

//--- Enum per la macchina a stati
enum TRADE_STATE {
   STATE_IDLE,
   STATE_WAITING_SETUP,
   STATE_IN_TRADE
};

//--- Input Parameters
input group "=== Donchian Channel (H1) ==="
input int      Inp_Donchian_N = 20;
input int      Inp_Donchian_M = 10;

input group "=== Trend Filter (H4) ==="
input int      Inp_EMA_Period = 100;
input int      Inp_EMA_Slope_Bars = 5;

input group "=== Volatility & Quality Filters (H1) ==="
input int      Inp_ADX_Period = 14;
input double   Inp_ADX_Min = 26.0;
input int      Inp_ATR_Regime_Period = 50;
input double   Inp_ATR_Regime_Mult = 1.0;

input group "=== Sessione Operativa (ora server) ==="
input int      Inp_Session_Start_Hour = 8;
input int      Inp_Session_End_Hour = 21;

input group "=== Entry Refinement (M15) ==="
input bool     Inp_Use_Refined_Entry = true;
input ENUM_TIMEFRAMES Inp_Entry_TF = PERIOD_M15;
input int      Inp_Entry_Timeout_Hours = 3;
input int      Inp_Pullback_EMA = 20;
input double   Inp_Entry_Risk_Reduction = 0.5;

input group "=== Risk Management ==="
input double   Inp_Risk_Percent = 0.15;
input double   Inp_SL_ATR_Mult = 1.3;
input int      Inp_ATR_Period = 14;
input double   Inp_BE_Trigger_R = 0.8;
input double   Inp_Trailing_ATR_Mult = 1.5;
input double   Inp_Max_Spread_ATR_Ratio = 0.3;

input group "=== Money Management Adattivo ==="
input double   Inp_DD_Threshold = 0.03;          // Soglia DD per attivare modalità difensiva (3%)
input double   Inp_Risk_Reduction_DD = 0.6;      // Fattore riduzione rischio in DD (60% del normale)
input int      Inp_Consec_Losses_Trigger = 4;
input double   Inp_Risk_Reduction_Factor = 0.5;
input int      Inp_Pause_After_Big_Loss_Hours = 12; // Pausa dopo perdita >1.5R (ore)

input group "=== Take Profit Parziale ==="
input double   Inp_Partial_TP_R = 1.0;
input double   Inp_Partial_TP_Percent = 50.0;

input group "=== General Settings ==="
input int      Inp_MagicNumber = 777;
input double   Inp_Slippage_Points = 10;

//--- Global Variables
CTrade         trade;
int            handle_ema_h4;
int            handle_adx_h1;
int            handle_atr_h1;
int            handle_atr_m15;
int            handle_ema_m15;

TRADE_STATE    current_state = STATE_IDLE;
datetime       signal_time = 0;
int            signal_direction = 0;
double         signal_breakout_level = 0;
double         entry_price = 0;
double         initial_risk_points = 0;
bool           indicators_ready = false;

int            consecutive_losses = 0;
bool           partial_tp_taken = false;
double         initial_equity_at_start = 0;
datetime       last_big_loss_time = 0;

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
  {
   trade.SetExpertMagicNumber(Inp_MagicNumber);
   trade.SetDeviationInPoints((int)Inp_Slippage_Points);
   trade.SetTypeFilling(ORDER_FILLING_FOK);

   string sym = _Symbol;

   handle_ema_h4 = iMA(sym, PERIOD_H4, Inp_EMA_Period, 0, MODE_EMA, PRICE_CLOSE);
   handle_adx_h1 = iADX(sym, PERIOD_H1, Inp_ADX_Period);
   handle_atr_h1 = iATR(sym, PERIOD_H1, Inp_ATR_Period);

   if(Inp_Use_Refined_Entry)
     {
      handle_atr_m15 = iATR(sym, Inp_Entry_TF, Inp_ATR_Period);
      handle_ema_m15 = iMA(sym, Inp_Entry_TF, Inp_Pullback_EMA, 0, MODE_EMA, PRICE_CLOSE);
     }

   if(handle_ema_h4 == INVALID_HANDLE || handle_adx_h1 == INVALID_HANDLE || handle_atr_h1 == INVALID_HANDLE)
     {
      Print("ERRORE CRITICO: Impossibile creare handle indicatori H1/H4");
      return(INIT_FAILED);
     }

   if(Inp_Use_Refined_Entry && (handle_atr_m15 == INVALID_HANDLE || handle_ema_m15 == INVALID_HANDLE))
     {
      Print("ERRORE CRITICO: Impossibile creare handle indicatori M15");
      return(INIT_FAILED);
     }

   // Registra equity iniziale per money management adattivo
   initial_equity_at_start = AccountInfoDouble(ACCOUNT_EQUITY);
   Print("EA v4.05 Inizializzato. Equity iniziale: ", initial_equity_at_start, " | Rischio base: ", Inp_Risk_Percent, "%");
   
   indicators_ready = true;
   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(handle_ema_h4);
   IndicatorRelease(handle_adx_h1);
   IndicatorRelease(handle_atr_h1);
   if(Inp_Use_Refined_Entry)
     {
      IndicatorRelease(handle_atr_m15);
      IndicatorRelease(handle_ema_m15);
     }
  }

//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
   if(!indicators_ready) return;

   bool has_position = CheckOpenPosition();

   if(has_position)
     {
      current_state = STATE_IN_TRADE;
      ManageOpenPositions();
      return;
     }
   else if(current_state == STATE_IN_TRADE)
     {
      UpdateConsecutiveLossCounter();
      current_state = STATE_IDLE;
      signal_direction = 0;
     }

   static datetime last_bar_h1 = 0;
   datetime current_bar_h1 = iTime(_Symbol, PERIOD_H1, 0);

   if(last_bar_h1 != current_bar_h1)
     {
      last_bar_h1 = current_bar_h1;
      if(current_state == STATE_IDLE) CheckForNewSignal();
     }

   if(current_state == STATE_WAITING_SETUP && Inp_Use_Refined_Entry)
     {
      CheckM15Setup();
     }
  }

//+------------------------------------------------------------------+
//| Controlla se c'è una posizione aperta                            |
//+------------------------------------------------------------------+
bool CheckOpenPosition()
  {
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == Inp_MagicNumber)
        return true;
     }
   return false;
  }

//+------------------------------------------------------------------+
//| Aggiorna contatore perdite consecutive e traccia grandi perdite  |
//+------------------------------------------------------------------+
void UpdateConsecutiveLossCounter()
  {
   HistorySelect(0, TimeCurrent());
   int total = HistoryDealsTotal();
   for(int i = total - 1; i >= 0; i--)
     {
      ulong deal_ticket = HistoryDealGetTicket(i);
      if(deal_ticket == 0) continue;
      if(HistoryDealGetInteger(deal_ticket, DEAL_MAGIC) != Inp_MagicNumber) continue;
      if(HistoryDealGetString(deal_ticket, DEAL_SYMBOL) != _Symbol) continue;
      if(HistoryDealGetInteger(deal_ticket, DEAL_ENTRY) != DEAL_ENTRY_OUT) continue;

      double deal_profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT)
                          + HistoryDealGetDouble(deal_ticket, DEAL_SWAP)
                          + HistoryDealGetDouble(deal_ticket, DEAL_COMMISSION);

      if(deal_profit < 0)
        {
         // Calcola perdita in termini di R
         double risk_money = initial_equity_at_start * (Inp_Risk_Percent / 100.0);
         double loss_in_r = MathAbs(deal_profit) / risk_money;
         
         if(loss_in_r > 1.5)
           {
            last_big_loss_time = TimeCurrent();
            Print("DEBUG: Perdita grande (", loss_in_r, "R). Pausa ", Inp_Pause_After_Big_Loss_Hours, "h attivata.");
           }
         
         consecutive_losses++;
         Print("DEBUG: Perdita. Consecutive: ", consecutive_losses);
        }
      else
        {
         if(consecutive_losses > 0) Print("DEBUG: Vincita. Reset contatore.");
         consecutive_losses = 0;
        }
      break;
     }
  }

//+------------------------------------------------------------------+
//| Verifica sessione operativa                                      |
//+------------------------------------------------------------------+
bool IsSessionActive()
  {
   MqlDateTime dt;
   TimeToStruct(TimeCurrent(), dt);
   int hour = dt.hour;

   if(Inp_Session_Start_Hour <= Inp_Session_End_Hour)
      return (hour >= Inp_Session_Start_Hour && hour < Inp_Session_End_Hour);
   else
      return (hour >= Inp_Session_Start_Hour || hour < Inp_Session_End_Hour);
  }

//+------------------------------------------------------------------+
//| Calcola media ATR                                                |
//+------------------------------------------------------------------+
double GetAverageATR(int handle, int period, int shift)
  {
   double atr_buf[];
   ArraySetAsSeries(atr_buf, true);
   if(CopyBuffer(handle, 0, shift, period, atr_buf) < period) return -1;

   double sum = 0;
   for(int i = 0; i < period; i++) sum += atr_buf[i];
   return sum / period;
  }

//+------------------------------------------------------------------+
//| Cerca nuovi segnali su H1/H4                                     |
//+------------------------------------------------------------------+
void CheckForNewSignal()
  {
   if(!IsSessionActive()) return;

   // Pausa post-perdita grande (12 ore invece di 24)
   int pause_seconds = Inp_Pause_After_Big_Loss_Hours * 3600;
   if(TimeCurrent() - last_big_loss_time < pause_seconds)
     {
      // Print("DEBUG: Pausa post-perdita grande attiva.");
      return;
     }

   int ema_h4_bars_needed = Inp_EMA_Slope_Bars + 2;
   double ema_h4[], adx_h1[], atr_h1[];
   ArraySetAsSeries(ema_h4, true);
   ArraySetAsSeries(adx_h1, true);
   ArraySetAsSeries(atr_h1, true);

   if(CopyBuffer(handle_ema_h4, 0, 0, ema_h4_bars_needed, ema_h4) < ema_h4_bars_needed) return;
   if(CopyBuffer(handle_adx_h1, 0, 0, 2, adx_h1) < 2) return;
   if(CopyBuffer(handle_atr_h1, 0, 0, 2, atr_h1) < 2) return;

   double current_ema = ema_h4[1];
   double current_adx = adx_h1[1];

   // Filtro regime volatilità
   double avg_atr_h1 = GetAverageATR(handle_atr_h1, Inp_ATR_Regime_Period, 1);
   if(avg_atr_h1 <= 0) return;
   bool volatility_ok = (atr_h1[1] >= avg_atr_h1 * Inp_ATR_Regime_Mult);
   if(!volatility_ok) return;

   // Filtro pendenza EMA H4
   double ema_past = ema_h4[1 + Inp_EMA_Slope_Bars];
   bool ema_slope_up = (current_ema > ema_past);
   bool ema_slope_down = (current_ema < ema_past);

   double donchian_high = GetDonchianHigh(PERIOD_H1, Inp_Donchian_N, 2);
   double donchian_low  = GetDonchianLow(PERIOD_H1, Inp_Donchian_N, 2);
   double close_prev    = iClose(_Symbol, PERIOD_H1, 1);

   bool long_signal = (close_prev > donchian_high) && (current_ema < close_prev) && ema_slope_up
                       && (current_adx > Inp_ADX_Min);
                       
   bool short_signal = (close_prev < donchian_low) && (current_ema > close_prev) && ema_slope_down
                        && (current_adx > Inp_ADX_Min);

   if(long_signal)
     {
      current_state = STATE_WAITING_SETUP;
      signal_direction = 1;
      signal_time = TimeCurrent();
      signal_breakout_level = donchian_high;
      Print("Segnale LONG H1 rilevato. Attesa M15...");
      if(!Inp_Use_Refined_Entry) OpenPosition(1, atr_h1[1], false);
     }
   else if(short_signal)
     {
      current_state = STATE_WAITING_SETUP;
      signal_direction = -1;
      signal_time = TimeCurrent();
      signal_breakout_level = donchian_low;
      Print("Segnale SHORT H1 rilevato. Attesa M15...");
      if(!Inp_Use_Refined_Entry) OpenPosition(-1, atr_h1[1], false);
     }
  }

//+------------------------------------------------------------------+
//| Cerca setup di entry su M15                                      |
//+------------------------------------------------------------------+
void CheckM15Setup()
  {
   int hours_elapsed = (int)((TimeCurrent() - signal_time) / 3600);
   if(hours_elapsed >= Inp_Entry_Timeout_Hours)
     {
      Print("Timeout. Fallback H1.");
      double atr_h1[];
      ArraySetAsSeries(atr_h1, true);
      if(CopyBuffer(handle_atr_h1, 0, 0, 2, atr_h1) >= 2)
        OpenPosition(signal_direction, atr_h1[1], false);
      current_state = STATE_IDLE;
      signal_direction = 0;
      return;
     }

   double ema_m15[], atr_m15[];
   ArraySetAsSeries(ema_m15, true);
   ArraySetAsSeries(atr_m15, true);

   if(CopyBuffer(handle_ema_m15, 0, 0, 2, ema_m15) < 2) return;
   if(CopyBuffer(handle_atr_m15, 0, 0, 2, atr_m15) < 2) return;

   double current_price = (signal_direction == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
   double current_ema_m15 = ema_m15[1];
   double close_m15 = iClose(_Symbol, Inp_Entry_TF, 1);

   bool setup_found = false;

   if(signal_direction == 1)
     {
      bool near_ema = MathAbs(current_price - current_ema_m15) < (atr_m15[1] * 0.5);
      bool near_breakout = MathAbs(current_price - signal_breakout_level) < (atr_m15[1] * 0.3);
      double prev_high_m15 = iHigh(_Symbol, Inp_Entry_TF, 2);
      bool confirmation = (close_m15 > prev_high_m15);
      if((near_ema || near_breakout) && confirmation) setup_found = true;
     }
   else if(signal_direction == -1)
     {
      bool near_ema = MathAbs(current_price - current_ema_m15) < (atr_m15[1] * 0.5);
      bool near_breakout = MathAbs(current_price - signal_breakout_level) < (atr_m15[1] * 0.3);
      double prev_low_m15 = iLow(_Symbol, Inp_Entry_TF, 2);
      bool confirmation = (close_m15 < prev_low_m15);
      if((near_ema || near_breakout) && confirmation) setup_found = true;
     }

   if(setup_found)
     {
      Print("Setup M15 trovato.");
      OpenPosition(signal_direction, atr_m15[1], true);
     }
  }

//+------------------------------------------------------------------+
//| Apre la posizione con Money Management Adattivo                  |
//+------------------------------------------------------------------+
void OpenPosition(int direction, double atr_val, bool is_refined_entry)
  {
   if(!IsSessionActive())
     {
      Print("DEBUG: Fuori sessione.");
      current_state = STATE_IDLE;
      signal_direction = 0;
      return;
     }

   double sl_dist = atr_val * Inp_SL_ATR_Mult;
   
   // === MONEY MANAGEMENT ADATTIVO ===
   double base_risk = Inp_Risk_Percent;
   double current_equity = AccountInfoDouble(ACCOUNT_EQUITY);
   
   // Modalità difensiva: solo se equity scende sotto soglia DD (default 3%)
   double dd_threshold_equity = initial_equity_at_start * (1.0 - Inp_DD_Threshold);
   if(current_equity < dd_threshold_equity)
     {
      base_risk *= Inp_Risk_Reduction_DD; // Riduzione al 60% del normale
      Print("DEBUG: Equity in DD >", Inp_DD_Threshold*100, "%. Rischio ridotto a ", base_risk, "%");
     }
   // Reset immediato quando recuperi sopra l'equity iniziale
   else if(current_equity >= initial_equity_at_start && base_risk < Inp_Risk_Percent)
     {
      Print("DEBUG: Equity recuperata. Rischio ripristinato a ", Inp_Risk_Percent, "%");
     }

   double risk_percent = base_risk;
   if(is_refined_entry) risk_percent *= Inp_Entry_Risk_Reduction;
   if(consecutive_losses >= Inp_Consec_Losses_Trigger)
     {
      risk_percent *= Inp_Risk_Reduction_Factor;
      Print("DEBUG: Rischio ridotto per ", consecutive_losses, " perdite consecutive.");
     }

   double entry = 0;
   double sl = 0;
   bool res = false;

   // Filtro spread
   long current_spread_points_long = SymbolInfoInteger(_Symbol, SYMBOL_SPREAD);
   double current_spread_points = (double)current_spread_points_long;
   double current_spread_price = current_spread_points * SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   double max_allowed_spread = atr_val * Inp_Max_Spread_ATR_Ratio;

   if(current_spread_price > max_allowed_spread)
     {
      Print("DEBUG: Spread troppo alto.");
      return;
     }

   if(direction == 1)
     {
      entry = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
      sl = entry - sl_dist;
      double lot = CalculateLotSize(sl_dist, risk_percent);
      double risk_money = current_equity * (risk_percent / 100.0);
      Print("DEBUG RISK LONG: Lot=", lot, " Risk%=", risk_percent, " Equity=", current_equity);
      if(lot > 0) res = trade.Buy(lot, _Symbol, entry, sl, 0, "Turtle Long");
     }
   else if(direction == -1)
     {
      entry = SymbolInfoDouble(_Symbol, SYMBOL_BID);
      sl = entry + sl_dist;
      double lot = CalculateLotSize(sl_dist, risk_percent);
      double risk_money = current_equity * (risk_percent / 100.0);
      Print("DEBUG RISK SHORT: Lot=", lot, " Risk%=", risk_percent, " Equity=", current_equity);
      if(lot > 0) res = trade.Sell(lot, _Symbol, entry, sl, 0, "Turtle Short");
     }
   else
     {
      Print("ERRORE: Direzione non valida.");
      return;
     }

   if(res)
     {
      entry_price = entry;
      initial_risk_points = sl_dist / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
      current_state = STATE_IN_TRADE;
      partial_tp_taken = false;
      Print("Posizione aperta: ", (direction==1?"LONG":"SHORT"), " @ ", entry);
     }
   else
     {
      Print("ERRORE APERTURA: ", trade.ResultRetcodeDescription());
     }
  }

//+------------------------------------------------------------------+
//| Calcola dimensione lotto                                         |
//+------------------------------------------------------------------+
double CalculateLotSize(double sl_distance_price, double risk_percent)
  {
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);
   double risk_money = equity * (risk_percent / 100.0);
   
   double tick_size = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_SIZE);
   double tick_value = SymbolInfoDouble(_Symbol, SYMBOL_TRADE_TICK_VALUE);
   
   if(tick_size == 0 || tick_value == 0) return 0;
   
   double sl_in_ticks = sl_distance_price / tick_size;
   double lot = risk_money / (sl_in_ticks * tick_value);
   
   double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double max_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX);
   double step_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   
   lot = MathFloor(lot / step_lot) * step_lot;
   lot = MathMax(min_lot, MathMin(max_lot, lot));
   
   return lot;
  }

//+------------------------------------------------------------------+
//| Gestisce posizioni aperte                                        |
//+------------------------------------------------------------------+
void ManageOpenPositions()
  {
   for(int i = 0; i < PositionsTotal(); i++)
     {
      if(PositionGetSymbol(i) == _Symbol && PositionGetInteger(POSITION_MAGIC) == Inp_MagicNumber)
        {
         ulong ticket = PositionGetTicket(i);
         double pos_open = PositionGetDouble(POSITION_PRICE_OPEN);
         double pos_sl = PositionGetDouble(POSITION_SL);
         double pos_volume = PositionGetDouble(POSITION_VOLUME);
         long pos_type_long = PositionGetInteger(POSITION_TYPE);

         double current_price = (pos_type_long == POSITION_TYPE_BUY) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
         double profit_points = MathAbs(current_price - pos_open) / SymbolInfoDouble(_Symbol, SYMBOL_POINT);
         double r_current = profit_points / initial_risk_points;

         // TP Parziale
         if(!partial_tp_taken && r_current >= Inp_Partial_TP_R)
           {
            double volume_step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            double min_lot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            double partial_volume = pos_volume * (Inp_Partial_TP_Percent / 100.0);
            partial_volume = MathFloor(partial_volume / volume_step) * volume_step;
            double remaining_volume = pos_volume - partial_volume;
            
            if(partial_volume >= min_lot && remaining_volume >= min_lot)
              {
               if(trade.PositionClosePartial(ticket, partial_volume))
                 {
                  partial_tp_taken = true;
                  Print("DEBUG: TP Parziale eseguito.");
                 }
              }
            else
              {
               partial_tp_taken = true;
              }
           }

         // Break-Even
         if(r_current >= Inp_BE_Trigger_R)
           {
            if(pos_type_long == POSITION_TYPE_BUY && pos_sl < pos_open)
              trade.PositionModify(ticket, pos_open, 0);
            else if(pos_type_long == POSITION_TYPE_SELL && pos_sl > pos_open)
              trade.PositionModify(ticket, pos_open, 0);
           }

         // Trailing Ibrido
         double atr_h1[];
         ArraySetAsSeries(atr_h1, true);
         if(CopyBuffer(handle_atr_h1, 0, 0, 2, atr_h1) < 2) continue;

         double atr_val = atr_h1[1];
         double donchian_exit_high = GetDonchianHigh(PERIOD_H1, Inp_Donchian_M, 2);
         double donchian_exit_low  = GetDonchianLow(PERIOD_H1, Inp_Donchian_M, 2);
         double new_sl = 0;

         pos_sl = PositionGetDouble(POSITION_SL);

         if(pos_type_long == POSITION_TYPE_BUY)
           {
            double atr_sl = current_price - (atr_val * Inp_Trailing_ATR_Mult);
            double donch_sl = donchian_exit_low;
            new_sl = MathMax(atr_sl, donch_sl);
            if(new_sl > pos_sl && new_sl < current_price)
              trade.PositionModify(ticket, new_sl, 0);
           }
         else if(pos_type_long == POSITION_TYPE_SELL)
           {
            double atr_sl = current_price + (atr_val * Inp_Trailing_ATR_Mult);
            double donch_sl = donchian_exit_high;
            new_sl = MathMin(atr_sl, donch_sl);
            if((new_sl < pos_sl || pos_sl == 0) && new_sl > current_price)
              trade.PositionModify(ticket, new_sl, 0);
           }
        }
     }
  }

//+------------------------------------------------------------------+
//| Calcola Donchian High/Low                                        |
//+------------------------------------------------------------------+
double GetDonchianHigh(ENUM_TIMEFRAMES tf, int period, int shift)
  {
   double max_val = 0;
   for(int i = shift; i < shift + period; i++)
     {
      double h = iHigh(_Symbol, tf, i);
      if(h > max_val || i == shift) max_val = h;
     }
   return max_val;
  }

double GetDonchianLow(ENUM_TIMEFRAMES tf, int period, int shift)
  {
   double min_val = 999999;
   for(int i = shift; i < shift + period; i++)
     {
      double l = iLow(_Symbol, tf, i);
      if(l < min_val || i == shift) min_val = l;
     }
   return min_val;
  }