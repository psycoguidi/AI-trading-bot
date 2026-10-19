//+------------------------------------------------------------------+
//|                                     Moving Averages Improved.mq5 |
//|         Basato su MetaQuotes "Moving Averages.mq5" (2000-2026)   |
//|         Modifiche: trend filter, ATR SL/TP, filtro volatilità    |
//+------------------------------------------------------------------+
#property copyright "Adattato da MetaQuotes Ltd."
#property link      "https://www.mql5.com"
#property version   "2.00"

#include <Trade\Trade.mqh>

//--- Money management
input double MaximumRisk        = 0.02;    // Rischio massimo per trade (% margine libero)
input double DecreaseFactor     = 3;       // Fattore riduzione lotto dopo perdite

//--- Segnale principale (MA veloce = crossover come originale)
input int    MovingPeriod       = 12;      // Periodo MA veloce
input int    MovingShift        = 6;       // Shift MA veloce

//--- NUOVO: filtro di trend (MA lenta)
input bool   UseTrendFilter     = true;    // Attiva filtro di trend
input int    TrendPeriod        = 100;     // Periodo MA lenta (trend)
input ENUM_MA_METHOD TrendMethod = MODE_SMA; // Metodo MA lenta

//--- NUOVO: Stop Loss / Take Profit basati su ATR
input bool   UseATR_SLTP        = true;    // Attiva SL/TP dinamici
input int    ATR_Period         = 14;      // Periodo ATR
input double ATR_SL_Mult        = 1.5;     // Moltiplicatore ATR per Stop Loss
input double ATR_TP_Mult        = 2.5;     // Moltiplicatore ATR per Take Profit (>SL per profit factor migliore)

//--- NUOVO: filtro volatilità minima (evita mercato piatto)
input bool   UseVolatilityFilter = true;   // Attiva filtro volatilità minima
input double MinATR_Points      = 50;      // ATR minimo in punti per operare

//--- NUOVO: cooldown tra trade
input int    CooldownBars       = 3;       // Barre minime di attesa tra un trade e l'altro sullo stesso simbolo

//---
int    ExtHandleFast=0;
int    ExtHandleSlow=0;
int    ExtHandleATR=0;
bool   ExtHedging=false;
CTrade ExtTrade;
datetime ExtLastTradeBarTime=0;

#define MA_MAGIC 1234502   // magic number diverso dall'originale per non confondere le statistiche
//+------------------------------------------------------------------+
//| Calcolo lotto ottimale (invariato rispetto all'originale)        |
//+------------------------------------------------------------------+
double TradeSizeOptimized(void)
  {
   double price=0.0;
   double margin=0.0;
   if(!SymbolInfoDouble(_Symbol,SYMBOL_ASK,price))
      return(0.0);
   if(!OrderCalcMargin(ORDER_TYPE_BUY,_Symbol,1.0,price,margin))
      return(0.0);
   if(margin<=0.0)
      return(0.0);

   double lot=NormalizeDouble(AccountInfoDouble(ACCOUNT_MARGIN_FREE)*MaximumRisk/margin,2);

   if(DecreaseFactor>0)
     {
      HistorySelect(0,TimeCurrent());
      int    orders=HistoryDealsTotal();
      int    losses=0;

      for(int i=orders-1;i>=0;i--)
        {
         ulong ticket=HistoryDealGetTicket(i);
         if(ticket==0)
           {
            Print("HistoryDealGetTicket failed, no trade history");
            break;
           }
         if(HistoryDealGetString(ticket,DEAL_SYMBOL)!=_Symbol)
            continue;
         if(HistoryDealGetInteger(ticket,DEAL_MAGIC)!=MA_MAGIC)
            continue;
         double profit=HistoryDealGetDouble(ticket,DEAL_PROFIT);
         if(profit>0.0)
            break;
         if(profit<0.0)
            losses++;
        }
      if(losses>1)
         lot=NormalizeDouble(lot-lot*losses/DecreaseFactor,1);
     }

   double stepvol=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   lot=stepvol*NormalizeDouble(lot/stepvol,0);

   double minvol=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN);
   if(lot<minvol)
      lot=minvol;

   double maxvol=SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX);
   if(lot>maxvol)
      lot=maxvol;

   return(lot);
  }
//+------------------------------------------------------------------+
//| Verifica cooldown tra trade                                      |
//+------------------------------------------------------------------+
bool CooldownElapsed(void)
  {
   if(CooldownBars<=0)
      return(true);

   datetime barTime = iTime(_Symbol,_Period,0);
   int barsElapsed = iBarShift(_Symbol,_Period,ExtLastTradeBarTime);

   if(ExtLastTradeBarTime==0)
      return(true);

   return(barsElapsed>=CooldownBars);
  }
//+------------------------------------------------------------------+
//| Verifica filtro di volatilità minima                              |
//+------------------------------------------------------------------+
bool VolatilityOK(double atrValue)
  {
   if(!UseVolatilityFilter)
      return(true);

   double atrPoints = atrValue/_Point;
   return(atrPoints>=MinATR_Points);
  }
//+------------------------------------------------------------------+
//| Check per apertura posizione                                      |
//+------------------------------------------------------------------+
void CheckForOpen(void)
  {
   MqlRates rt[2];
   if(CopyRates(_Symbol,_Period,0,2,rt)!=2)
     {
      Print("CopyRates of ",_Symbol," failed, no history");
      return;
     }
   if(rt[1].tick_volume>1)
      return;

   //--- MA veloce (segnale di crossover, come originale)
   double maFast[1];
   if(CopyBuffer(ExtHandleFast,0,0,1,maFast)!=1)
     {
      Print("CopyBuffer from fast MA failed");
      return;
     }

   //--- ATR per SL/TP e filtro volatilità
   double atr[1];
   if(UseATR_SLTP || UseVolatilityFilter)
     {
      if(CopyBuffer(ExtHandleATR,0,0,1,atr)!=1)
        {
         Print("CopyBuffer from ATR failed");
         return;
        }
      if(!VolatilityOK(atr[0]))
         return; // mercato troppo piatto, salta il segnale
     }

   //--- filtro cooldown
   if(!CooldownElapsed())
      return;

   //--- segnale di base (crossover come originale)
   ENUM_ORDER_TYPE signal=WRONG_VALUE;

   if(rt[0].open>maFast[0] && rt[0].close<maFast[0])
      signal=ORDER_TYPE_SELL;
   else
     {
      if(rt[0].open<maFast[0] && rt[0].close>maFast[0])
         signal=ORDER_TYPE_BUY;
     }

   if(signal==WRONG_VALUE)
      return;

   //--- NUOVO: filtro di trend con MA lenta
   if(UseTrendFilter)
     {
      double maSlow[1];
      if(CopyBuffer(ExtHandleSlow,0,0,1,maSlow)!=1)
        {
         Print("CopyBuffer from slow MA failed");
         return;
        }
      //--- opera solo nella direzione del trend maggiore
      if(signal==ORDER_TYPE_BUY && rt[0].close<maSlow[0])
         return; // niente buy contro trend ribassista
      if(signal==ORDER_TYPE_SELL && rt[0].close>maSlow[0])
         return; // niente sell contro trend rialzista
     }

   //--- apertura con SL/TP dinamici
   if(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) && Bars(_Symbol,_Period)>100)
     {
      double price = SymbolInfoDouble(_Symbol,signal==ORDER_TYPE_SELL?SYMBOL_BID:SYMBOL_ASK);
      double sl=0.0, tp=0.0;

      if(UseATR_SLTP)
        {
         double slDist = atr[0]*ATR_SL_Mult;
         double tpDist = atr[0]*ATR_TP_Mult;
         if(signal==ORDER_TYPE_BUY)
           {
            sl = price - slDist;
            tp = price + tpDist;
           }
         else
           {
            sl = price + slDist;
            tp = price - tpDist;
           }
        }

      double lot = TradeSizeOptimized();
      if(ExtTrade.PositionOpen(_Symbol,signal,lot,price,sl,tp))
         ExtLastTradeBarTime = iTime(_Symbol,_Period,0);
     }
  }
//+------------------------------------------------------------------+
//| Check per chiusura posizione (segnale opposto, come originale)   |
//| Con SL/TP attivi questo diventa una chiusura di sicurezza extra  |
//+------------------------------------------------------------------+
void CheckForClose(void)
  {
   MqlRates rt[2];
   if(CopyRates(_Symbol,_Period,0,2,rt)!=2)
     {
      Print("CopyRates of ",_Symbol," failed, no history");
      return;
     }
   if(rt[1].tick_volume>1)
      return;

   double maFast[1];
   if(CopyBuffer(ExtHandleFast,0,0,1,maFast)!=1)
     {
      Print("CopyBuffer from fast MA failed");
      return;
     }

   bool signal=false;
   long type=PositionGetInteger(POSITION_TYPE);

   if(type==(long)POSITION_TYPE_BUY && rt[0].open>maFast[0] && rt[0].close<maFast[0])
      signal=true;
   if(type==(long)POSITION_TYPE_SELL && rt[0].open<maFast[0] && rt[0].close>maFast[0])
      signal=true;

   if(signal)
     {
      if(TerminalInfoInteger(TERMINAL_TRADE_ALLOWED) && Bars(_Symbol,_Period)>100)
         ExtTrade.PositionClose(_Symbol,3);
     }
  }
//+------------------------------------------------------------------+
//| Selezione posizione (netting / hedging, invariato)                |
//+------------------------------------------------------------------+
bool SelectPosition()
  {
   bool res=false;
   if(ExtHedging)
     {
      uint total=PositionsTotal();
      for(uint i=0; i<total; i++)
        {
         string position_symbol=PositionGetSymbol(i);
         if(_Symbol==position_symbol && MA_MAGIC==PositionGetInteger(POSITION_MAGIC))
           {
            res=true;
            break;
           }
        }
     }
   else
     {
      if(!PositionSelect(_Symbol))
         return(false);
      else
         return(PositionGetInteger(POSITION_MAGIC)==MA_MAGIC);
     }
   return(res);
  }
//+------------------------------------------------------------------+
//| Inizializzazione                                                   |
//+------------------------------------------------------------------+
int OnInit(void)
  {
   ExtHedging=((ENUM_ACCOUNT_MARGIN_MODE)AccountInfoInteger(ACCOUNT_MARGIN_MODE)==ACCOUNT_MARGIN_MODE_RETAIL_HEDGING);
   ExtTrade.SetExpertMagicNumber(MA_MAGIC);
   ExtTrade.SetMarginMode();
   ExtTrade.SetTypeFillingBySymbol(Symbol());

   ExtHandleFast=iMA(_Symbol,_Period,MovingPeriod,MovingShift,MODE_SMA,PRICE_CLOSE);
   if(ExtHandleFast==INVALID_HANDLE)
     {
      printf("Errore creazione MA veloce");
      return(INIT_FAILED);
     }

   if(UseTrendFilter)
     {
      ExtHandleSlow=iMA(_Symbol,_Period,TrendPeriod,0,TrendMethod,PRICE_CLOSE);
      if(ExtHandleSlow==INVALID_HANDLE)
        {
         printf("Errore creazione MA lenta (trend filter)");
         return(INIT_FAILED);
        }
     }

   if(UseATR_SLTP || UseVolatilityFilter)
     {
      ExtHandleATR=iATR(_Symbol,_Period,ATR_Period);
      if(ExtHandleATR==INVALID_HANDLE)
        {
         printf("Errore creazione ATR");
         return(INIT_FAILED);
        }
     }

   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Tick                                                                |
//+------------------------------------------------------------------+
void OnTick(void)
  {
   if(SelectPosition())
      CheckForClose();
   else
      CheckForOpen();
  }
//+------------------------------------------------------------------+
//| Deinizializzazione                                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   if(ExtHandleFast!=INVALID_HANDLE) IndicatorRelease(ExtHandleFast);
   if(ExtHandleSlow!=INVALID_HANDLE && ExtHandleSlow!=0) IndicatorRelease(ExtHandleSlow);
   if(ExtHandleATR!=INVALID_HANDLE && ExtHandleATR!=0) IndicatorRelease(ExtHandleATR);
  }
//+------------------------------------------------------------------+