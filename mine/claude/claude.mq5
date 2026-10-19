//+------------------------------------------------------------------+
//|                                     MultiAsset_Trend_EA_v3.mq5   |
//|  Fix critico: position sizing via OrderCalcProfit (accurato per  |
//|  ogni simbolo) + drawdown guard che chiude posizioni anche su    |
//|  soglia giornaliera, non solo totale.                            |
//+------------------------------------------------------------------+
#property copyright "Custom EA"
#property version   "3.00"
#property strict

#include <Trade\Trade.mqh>
CTrade trade;

input string InpSymbols        = "XAUUSD";
input string InpSymbolSuffix   = "";

input int    InpEMA_Fast       = 50;
input int    InpEMA_Slow       = 200;

input int    InpRSI_Period     = 14;
input double InpRSI_ZoneLow    = 45.0;
input double InpRSI_ZoneHigh   = 55.0;
input int    InpRSI_LookbackBars = 3;

input int    InpBreakoutBars   = 20;
input bool   InpUseBreakout    = true;

input int    InpADX_Period     = 14;
input double InpADX_MinLevel   = 15.0;
input bool   InpUseADXFilter   = true;

input int    InpATR_Period     = 14;
input double InpSL_ATR_Mult    = 1.5;
input double InpTP_ATR_Mult    = 2.5;
input double InpMaxATR_Spike   = 3.0;

input double InpRiskPercent    = 0.4;
input double InpMaxDailyDD     = 5.0;
input double InpMaxTotalDD     = 10.0;
input int    InpMaxTradesDay   = 15;
input int    InpMaxTradesPerSymbolDay = 6;
input int    InpMagicNumber    = 778003;

input int    InpSessionStartHr = 8;
input int    InpSessionEndHr   = 21;

string   g_symbols[];
double   g_dayStartEquity = 0;
double   g_initialBalance = 0;
datetime g_currentDay     = 0;
int      g_tradesToday    = 0;
int      g_tradesPerSymbolToday[];
bool     g_ddHalt         = false;

int OnInit()
{
   StringSplit(InpSymbols, ',', g_symbols);
   ArrayResize(g_tradesPerSymbolToday, ArraySize(g_symbols));
   for(int i=0;i<ArraySize(g_symbols);i++)
   {
      g_symbols[i] = g_symbols[i] + InpSymbolSuffix;
      if(!SymbolSelect(g_symbols[i], true))
         Print("ERRORE: simbolo non trovato -> ", g_symbols[i]);
      g_tradesPerSymbolToday[i] = 0;
   }
   g_initialBalance = AccountInfoDouble(ACCOUNT_BALANCE);
   g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
   g_currentDay = TimeCurrent();
   trade.SetExpertMagicNumber(InpMagicNumber);
   return(INIT_SUCCEEDED);
}

void OnTick()
{
   CheckNewDay();
   if(!CheckDrawdownGuard()) return;
   if(!IsSessionOpen()) return;
   if(g_tradesToday >= InpMaxTradesDay) return;

   for(int i=0;i<ArraySize(g_symbols);i++)
   {
      if(g_tradesPerSymbolToday[i] >= InpMaxTradesPerSymbolDay) continue;
      ProcessSymbol(g_symbols[i], i);
   }
}

void CheckNewDay()
{
   MqlDateTime t1, t2;
   TimeToStruct(TimeCurrent(), t1);
   TimeToStruct(g_currentDay, t2);
   if(t1.day != t2.day || t1.mon != t2.mon || t1.year != t2.year)
   {
      g_currentDay      = TimeCurrent();
      g_dayStartEquity  = AccountInfoDouble(ACCOUNT_EQUITY);
      g_tradesToday     = 0;
      g_ddHalt          = false;
      for(int i=0;i<ArraySize(g_tradesPerSymbolToday);i++) g_tradesPerSymbolToday[i]=0;
   }
}

// FIX: ora chiude le posizioni aperte anche sulla soglia giornaliera, non solo su quella totale
bool CheckDrawdownGuard()
{
   double equity = AccountInfoDouble(ACCOUNT_EQUITY);

   double dailyDD = (g_dayStartEquity - equity) / g_dayStartEquity * 100.0;
   if(dailyDD >= InpMaxDailyDD)
   {
      if(!g_ddHalt) { Print("STOP GIORNALIERO: DD ", DoubleToString(dailyDD,2), "% >= ", InpMaxDailyDD, "%. Chiudo posizioni aperte."); CloseAllPositions(); }
      g_ddHalt = true;
      return false;
   }

   double totalDD = (g_initialBalance - equity) / g_initialBalance * 100.0;
   if(totalDD >= InpMaxTotalDD)
   {
      Print("STOP EA: drawdown totale ", DoubleToString(totalDD,2), "% >= ", InpMaxTotalDD, "%. Chiudo tutto e fermo l'EA.");
      CloseAllPositions();
      ExpertRemove();
      return false;
   }
   if(g_ddHalt) return false;
   return true;
}

void CloseAllPositions()
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) && PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
         trade.PositionClose(ticket);
   }
}

bool IsSessionOpen()
{
   MqlDateTime t;
   TimeToStruct(TimeCurrent(), t);
   return (t.hour >= InpSessionStartHr && t.hour < InpSessionEndHr);
}

void ProcessSymbol(string symbol, int idx)
{
   if(HasOpenPosition(symbol)) return;

   int hEmaFastH1 = iMA(symbol, PERIOD_H1, InpEMA_Fast, 0, MODE_EMA, PRICE_CLOSE);
   int hEmaSlowH1 = iMA(symbol, PERIOD_H1, InpEMA_Slow, 0, MODE_EMA, PRICE_CLOSE);
   double emaFastH1[], emaSlowH1[];
   ArraySetAsSeries(emaFastH1,true); ArraySetAsSeries(emaSlowH1,true);
   if(hEmaFastH1==INVALID_HANDLE || hEmaSlowH1==INVALID_HANDLE) return;
   if(CopyBuffer(hEmaFastH1,0,0,2,emaFastH1)<2) return;
   if(CopyBuffer(hEmaSlowH1,0,0,2,emaSlowH1)<2) return;
   bool trendUp   = emaFastH1[0] > emaSlowH1[0];
   bool trendDown = emaFastH1[0] < emaSlowH1[0];

   int hRsi = iRSI(symbol, PERIOD_M15, InpRSI_Period, PRICE_CLOSE);
   int hAdx = iADX(symbol, PERIOD_M15, InpADX_Period);
   int hAtr = iATR(symbol, PERIOD_M15, InpATR_Period);
   if(hRsi==INVALID_HANDLE || hAdx==INVALID_HANDLE || hAtr==INVALID_HANDLE) return;

   double rsiVal[], adxVal[], atrVal[];
   ArraySetAsSeries(rsiVal,true); ArraySetAsSeries(adxVal,true); ArraySetAsSeries(atrVal,true);
   int lookN = InpRSI_LookbackBars+1;
   if(CopyBuffer(hRsi,0,0,lookN,rsiVal)<lookN) return;
   if(CopyBuffer(hAdx,0,0,2,adxVal)<2) return;
   if(CopyBuffer(hAtr,0,0,20,atrVal)<20) return;

   double atrNow = atrVal[0];
   double atrAvg = 0; for(int i=0;i<20;i++) atrAvg += atrVal[i]; atrAvg/=20.0;
   if(atrNow > atrAvg * InpMaxATR_Spike) return;

   bool adxOk = (!InpUseADXFilter) || (adxVal[0] >= InpADX_MinLevel);

   bool momentumLong=false, momentumShort=false;
   for(int i=1;i<lookN;i++)
   {
      if(rsiVal[i] < InpRSI_ZoneLow && rsiVal[0] >= InpRSI_ZoneLow) momentumLong=true;
      if(rsiVal[i] > InpRSI_ZoneHigh && rsiVal[0] <= InpRSI_ZoneHigh) momentumShort=true;
   }

   bool breakoutLong=false, breakoutShort=false;
   if(InpUseBreakout)
   {
      double highs[], lows[];
      ArraySetAsSeries(highs,true); ArraySetAsSeries(lows,true);
      if(CopyHigh(symbol, PERIOD_M15, 1, InpBreakoutBars, highs)==InpBreakoutBars &&
         CopyLow(symbol, PERIOD_M15, 1, InpBreakoutBars, lows)==InpBreakoutBars)
      {
         double hh = highs[ArrayMaximum(highs)];
         double ll = lows[ArrayMinimum(lows)];
         double closeNow = iClose(symbol, PERIOD_M15, 0);
         breakoutLong  = closeNow > hh;
         breakoutShort = closeNow < ll;
      }
   }

   double ask = SymbolInfoDouble(symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(symbol, SYMBOL_BID);

   bool goLong  = trendUp   && adxOk && (momentumLong  || breakoutLong);
   bool goShort = trendDown && adxOk && (momentumShort || breakoutShort);

   if(goLong)       OpenTrade(symbol, ORDER_TYPE_BUY,  ask, atrNow, idx);
   else if(goShort) OpenTrade(symbol, ORDER_TYPE_SELL, bid, atrNow, idx);
}

bool HasOpenPosition(string symbol)
{
   for(int i=PositionsTotal()-1;i>=0;i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket) &&
         PositionGetString(POSITION_SYMBOL)==symbol &&
         PositionGetInteger(POSITION_MAGIC)==InpMagicNumber)
         return true;
   }
   return false;
}

void OpenTrade(string symbol, ENUM_ORDER_TYPE type, double price, double atr, int idx)
{
   double slDist = atr * InpSL_ATR_Mult;
   double tpDist = atr * InpTP_ATR_Mult;
   double sl = (type==ORDER_TYPE_BUY) ? price - slDist : price + slDist;
   double tp = (type==ORDER_TYPE_BUY) ? price + tpDist : price - tpDist;

   double lot = CalcLotSize(symbol, type, price, sl);
   if(lot <= 0) { Print("Lot size 0 per ", symbol, " - trade saltato"); return; }

   int digits = (int)SymbolInfoInteger(symbol, SYMBOL_DIGITS);
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);

   bool ok;
   if(type==ORDER_TYPE_BUY) ok = trade.Buy(lot, symbol, price, sl, tp, "TrendMomEA");
   else                     ok = trade.Sell(lot, symbol, price, sl, tp, "TrendMomEA");

   if(ok) { g_tradesToday++; g_tradesPerSymbolToday[idx]++; }
}

// FIX: calcolo rischio reale in denaro tramite OrderCalcProfit, accurato per qualsiasi simbolo
double CalcLotSize(string symbol, ENUM_ORDER_TYPE type, double entryPrice, double slPrice)
{
   double equity    = AccountInfoDouble(ACCOUNT_EQUITY);
   double riskMoney = equity * (InpRiskPercent/100.0);

   double profitFor1Lot;
   if(!OrderCalcProfit(type, symbol, 1.0, entryPrice, slPrice, profitFor1Lot))
   {
      Print("OrderCalcProfit fallito per ", symbol);
      return 0;
   }
   double lossFor1Lot = MathAbs(profitFor1Lot);
   if(lossFor1Lot <= 0) return 0;

   double lot = riskMoney / lossFor1Lot;

   double minLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MIN);
   double maxLot  = SymbolInfoDouble(symbol, SYMBOL_VOLUME_MAX);
   double stepLot = SymbolInfoDouble(symbol, SYMBOL_VOLUME_STEP);

   lot = MathFloor(lot/stepLot)*stepLot;
   lot = MathMax(minLot, MathMin(maxLot, lot));
   return lot;
}