//+------------------------------------------------------------------+
//| ORB_Original_NY_0930_V1.mq5                                     |
//| Simple New York opening-range breakout for XAUUSD / M5.          |
//| Range: first 30 minutes from the configured broker-server time.  |
//| Entry: completed M5 candle closes beyond the range.              |
//| SL: opposite side of range. TP: 2R. Partial: 50% at 1R.          |
//+------------------------------------------------------------------+
#property strict
#property version   "1.00"
#property description "Simple NY ORB: 09:30-10:00 local equivalent, M5 close breakout, 1:2 RR, partial at 1R."

#include <Trade/Trade.mqh>

input group "Opening range (broker server time)"
input int      InpStartHour          = 15;       // NY 09:30 is commonly 15:30 server time (check broker/DST)
input int      InpStartMinute        = 30;
input int      InpRangeMinutes       = 30;       // First 2 M15 candles or 1 M30 candle
input ENUM_TIMEFRAMES InpSignalTF    = PERIOD_M5;

input group "Entry and exits"
input double   InpRiskReward         = 2.0;
input bool     InpUsePartialClose    = true;
input double   InpPartialAtR         = 1.0;
input double   InpPartialClosePct    = 50.0;
input bool     InpRearmAfterReturn   = true;     // Re-enter after price closes back inside the range
input int      InpMaxOpenPositions  = 1;        // 0 = no EA position limit (hedging accounts only)
input int      InpDeviationPoints   = 30;

input group "Position sizing"
input double   InpRiskPercent        = 0.50;     // Percent of current balance risked per entry
input double   InpFixedLot           = 0.0;      // 0 = calculate from risk percent
input double   InpMaxLot             = 10.0;     // Safety cap for calculated/fixed volume

input group "General"
input ulong    InpMagicNumber        = 2026093001;
input bool     InpShowRange          = true;

CTrade trade;

int      g_dayKey = -1;
datetime g_sessionStart = 0;
datetime g_rangeEnd = 0;
datetime g_lastSignalBar = 0;
double   g_rangeHigh = 0.0;
double   g_rangeLow = 0.0;
bool     g_rangeReady = false;
bool     g_longArmed = true;
bool     g_shortArmed = true;
string   g_rangeObject = "ORB_Original_Range";

int DateKey(datetime t)
{
   MqlDateTime d;
   TimeToStruct(t, d);
   return d.year * 10000 + d.mon * 100 + d.day;
}

datetime DayStart(datetime t)
{
   MqlDateTime d;
   TimeToStruct(t, d);
   d.hour = 0;
   d.min = 0;
   d.sec = 0;
   return StructToTime(d);
}

double NormalizePrice(double price)
{
   return NormalizeDouble(price, (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS));
}

int VolumeDigits()
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   for(int digits = 0; digits <= 8; digits++)
      if(MathAbs(step - NormalizeDouble(step, digits)) < 1e-10)
         return digits;
   return 8;
}

double NormalizeVolumeDown(double volume)
{
   double step = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
   double minv = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
   double maxv = MathMin(SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MAX), InpMaxLot);
   if(step <= 0.0 || maxv < minv)
      return 0.0;
   volume = MathMin(volume, maxv);
   volume = MathFloor((volume + 1e-10) / step) * step;
   volume = NormalizeDouble(volume, VolumeDigits());
   if(volume < minv)
      return 0.0;
   return volume;
}

double CalculateVolume(ENUM_ORDER_TYPE orderType, double entry, double stop)
{
   if(InpFixedLot > 0.0)
      return NormalizeVolumeDown(InpFixedLot);

   double riskMoney = AccountInfoDouble(ACCOUNT_BALANCE) * InpRiskPercent / 100.0;
   if(riskMoney <= 0.0)
      return 0.0;

   double oneLotResult = 0.0;
   if(!OrderCalcProfit(orderType, _Symbol, 1.0, entry, stop, oneLotResult))
      return 0.0;
   double oneLotRisk = MathAbs(oneLotResult);
   if(oneLotRisk <= 0.0)
      return 0.0;
   return NormalizeVolumeDown(riskMoney / oneLotRisk);
}

int CountOwnPositions()
{
   int count = 0;
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) == _Symbol &&
         (ulong)PositionGetInteger(POSITION_MAGIC) == InpMagicNumber)
         count++;
   }
   return count;
}

bool IsNewSignalBar()
{
   datetime barTime = iTime(_Symbol, InpSignalTF, 0);
   if(barTime <= 0 || barTime == g_lastSignalBar)
      return false;
   g_lastSignalBar = barTime;
   return true;
}

void DrawRange()
{
   if(!InpShowRange || !g_rangeReady)
      return;
   ObjectDelete(0, g_rangeObject);
   if(ObjectCreate(0, g_rangeObject, OBJ_RECTANGLE, 0, g_sessionStart, g_rangeHigh, g_rangeEnd, g_rangeLow))
   {
      ObjectSetInteger(0, g_rangeObject, OBJPROP_COLOR, clrDodgerBlue);
      ObjectSetInteger(0, g_rangeObject, OBJPROP_STYLE, STYLE_SOLID);
      ObjectSetInteger(0, g_rangeObject, OBJPROP_WIDTH, 1);
      ObjectSetInteger(0, g_rangeObject, OBJPROP_FILL, false);
      ObjectSetInteger(0, g_rangeObject, OBJPROP_BACK, true);
   }
}

bool BuildOpeningRange()
{
   if(TimeTradeServer() < g_rangeEnd)
      return false;

   MqlRates rates[];
   ArraySetAsSeries(rates, false);
   int copied = CopyRates(_Symbol, PERIOD_M1, g_sessionStart, g_rangeEnd - 1, rates);
   if(copied <= 0)
      return false;

   double high = -DBL_MAX;
   double low = DBL_MAX;
   for(int i = 0; i < copied; i++)
   {
      if(rates[i].time < g_sessionStart || rates[i].time >= g_rangeEnd)
         continue;
      high = MathMax(high, rates[i].high);
      low = MathMin(low, rates[i].low);
   }

   if(high <= low || high == -DBL_MAX || low == DBL_MAX)
      return false;

   g_rangeHigh = NormalizePrice(high);
   g_rangeLow = NormalizePrice(low);
   g_rangeReady = true;
   DrawRange();
   PrintFormat("ORB opening range ready: %s | High=%.*f Low=%.*f",
               TimeToString(g_sessionStart, TIME_DATE|TIME_MINUTES),
               _Digits, g_rangeHigh, _Digits, g_rangeLow);
   return true;
}

bool PartialClosePosition(ulong ticket, double volume)
{
   if(!PositionSelectByTicket(ticket))
      return false;

   ENUM_ACCOUNT_MARGIN_MODE mode = (ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE);
   if(mode == ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)
      return trade.PositionClosePartial(ticket, volume);

   // Netting: send an opposite deal linked to this position to reduce its size.
   ENUM_POSITION_TYPE posType = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   request.action = TRADE_ACTION_DEAL;
   request.position = ticket;
   request.symbol = _Symbol;
   request.magic = InpMagicNumber;
   request.volume = volume;
   request.deviation = InpDeviationPoints;
   request.type = (posType == POSITION_TYPE_BUY ? ORDER_TYPE_SELL : ORDER_TYPE_BUY);
   request.price = (request.type == ORDER_TYPE_SELL ?
                    SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                    SymbolInfoDouble(_Symbol, SYMBOL_ASK));
   long filling = SymbolInfoInteger(_Symbol, SYMBOL_FILLING_MODE);
   request.type_filling = ((filling & SYMBOL_FILLING_FOK) != 0 ? ORDER_FILLING_FOK : ORDER_FILLING_IOC);
   if(!OrderSend(request, result))
      return false;
   return result.retcode == TRADE_RETCODE_DONE || result.retcode == TRADE_RETCODE_DONE_PARTIAL;
}

void ManagePartialCloses()
{
   if(!InpUsePartialClose || InpPartialClosePct <= 0.0 || InpPartialClosePct >= 100.0)
      return;

   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(ticket == 0 || !PositionSelectByTicket(ticket))
         continue;
      if(PositionGetString(POSITION_SYMBOL) != _Symbol ||
         (ulong)PositionGetInteger(POSITION_MAGIC) != InpMagicNumber)
         continue;

      string gv = StringFormat("ORB_ORIG_PART_%I64u", ticket);
      if(GlobalVariableCheck(gv))
         continue;

      double entry = PositionGetDouble(POSITION_PRICE_OPEN);
      double stop = PositionGetDouble(POSITION_SL);
      double volume = PositionGetDouble(POSITION_VOLUME);
      ENUM_POSITION_TYPE type = (ENUM_POSITION_TYPE)PositionGetInteger(POSITION_TYPE);
      double riskDistance = MathAbs(entry - stop);
      if(riskDistance <= 0.0)
         continue;

      double trigger = (type == POSITION_TYPE_BUY ?
                        entry + riskDistance * InpPartialAtR :
                        entry - riskDistance * InpPartialAtR);
      double market = (type == POSITION_TYPE_BUY ?
                       SymbolInfoDouble(_Symbol, SYMBOL_BID) :
                       SymbolInfoDouble(_Symbol, SYMBOL_ASK));
      bool reached = (type == POSITION_TYPE_BUY ? market >= trigger : market <= trigger);
      if(!reached)
         continue;

      double closeVolume = NormalizeVolumeDown(volume * InpPartialClosePct / 100.0);
      double remaining = NormalizeVolumeDown(volume - closeVolume);
      double minVolume = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
      if(closeVolume <= 0.0 || remaining < minVolume)
      {
         // If this broker cannot split the position at its volume step, leave it intact.
         GlobalVariableSet(gv, 1.0);
         continue;
      }

      if(PartialClosePosition(ticket, closeVolume))
      {
         GlobalVariableSet(gv, 1.0);
         PrintFormat("ORB partial close: ticket %I64u, volume %.2f at %.2fR",
                     ticket, closeVolume, InpPartialAtR);
      }
   }
}

bool OpenBreakout(ENUM_ORDER_TYPE orderType)
{
   double entry = (orderType == ORDER_TYPE_BUY ?
                   SymbolInfoDouble(_Symbol, SYMBOL_ASK) :
                   SymbolInfoDouble(_Symbol, SYMBOL_BID));
   double stop = (orderType == ORDER_TYPE_BUY ? g_rangeLow : g_rangeHigh);
   double riskDistance = MathAbs(entry - stop);
   if(riskDistance <= 0.0)
      return false;

   double target = (orderType == ORDER_TYPE_BUY ?
                    entry + riskDistance * InpRiskReward :
                    entry - riskDistance * InpRiskReward);
   entry = NormalizePrice(entry);
   stop = NormalizePrice(stop);
   target = NormalizePrice(target);

   long stopsLevel = SymbolInfoInteger(_Symbol, SYMBOL_TRADE_STOPS_LEVEL);
   double minDistance = stopsLevel * _Point;
   if(MathAbs(entry - stop) < minDistance || MathAbs(target - entry) < minDistance)
   {
      Print("ORB entry skipped: broker minimum stop distance exceeds the range-based SL/TP.");
      return false;
   }

   double volume = CalculateVolume(orderType, entry, stop);
   if(volume <= 0.0)
   {
      Print("ORB entry skipped: risk sizing produced less than the broker minimum volume.");
      return false;
   }

   bool sent = (orderType == ORDER_TYPE_BUY ?
                trade.Buy(volume, _Symbol, 0.0, stop, target, "Original NY ORB Buy") :
                trade.Sell(volume, _Symbol, 0.0, stop, target, "Original NY ORB Sell"));
   if(!sent)
   {
      PrintFormat("ORB order rejected: %u %s", trade.ResultRetcode(), trade.ResultRetcodeDescription());
      return false;
   }

   PrintFormat("ORB %s opened, volume %.2f, range low %.5f high %.5f, SL %.5f, TP %.5f",
               (orderType == ORDER_TYPE_BUY ? "BUY" : "SELL"), volume,
               g_rangeLow, g_rangeHigh, stop, target);
   return true;
}

void ResetTradingDay(datetime now)
{
   g_dayKey = DateKey(now);
   g_sessionStart = DayStart(now) + InpStartHour * 3600 + InpStartMinute * 60;
   g_rangeEnd = g_sessionStart + InpRangeMinutes * 60;
   g_rangeHigh = 0.0;
   g_rangeLow = 0.0;
   g_rangeReady = false;
   g_longArmed = true;
   g_shortArmed = true;
   ObjectDelete(0, g_rangeObject);
}

int OnInit()
{
   if(InpStartHour < 0 || InpStartHour > 23 || InpStartMinute < 0 || InpStartMinute > 59 ||
      InpRangeMinutes < 1 || InpRiskReward <= 0.0 || InpRiskPercent <= 0.0 ||
      InpPartialAtR <= 0.0 || InpPartialClosePct <= 0.0 || InpPartialClosePct >= 100.0 ||
      InpMaxLot <= 0.0)
      return INIT_PARAMETERS_INCORRECT;

   trade.SetExpertMagicNumber(InpMagicNumber);
   trade.SetDeviationInPoints(InpDeviationPoints);
   trade.SetTypeFillingBySymbol(_Symbol);
   ResetTradingDay(TimeTradeServer());
   g_lastSignalBar = iTime(_Symbol, InpSignalTF, 0);
   return INIT_SUCCEEDED;
}

void OnDeinit(const int reason)
{
   if(InpShowRange)
      ObjectDelete(0, g_rangeObject);
}

void OnTick()
{
   datetime now = TimeTradeServer();
   if(DateKey(now) != g_dayKey)
      ResetTradingDay(now);

   ManagePartialCloses();

   if(!g_rangeReady)
      BuildOpeningRange();
   if(!g_rangeReady || now < g_rangeEnd || !IsNewSignalBar())
      return;

   double close1 = iClose(_Symbol, InpSignalTF, 1);
   if(close1 <= 0.0)
      return;

   if(InpRearmAfterReturn && close1 < g_rangeHigh && close1 > g_rangeLow)
   {
      g_longArmed = true;
      g_shortArmed = true;
   }

   if(InpMaxOpenPositions > 0 && CountOwnPositions() >= InpMaxOpenPositions)
      return;

   if(close1 > g_rangeHigh && g_longArmed)
   {
      if(OpenBreakout(ORDER_TYPE_BUY))
         g_longArmed = false;
   }
   else if(close1 < g_rangeLow && g_shortArmed)
   {
      if(OpenBreakout(ORDER_TYPE_SELL))
         g_shortArmed = false;
   }
}
//+------------------------------------------------------------------+
