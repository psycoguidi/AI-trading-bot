// prova503: NAS100 ricerca 2025, guard FTMO+calendario UTC; solo ORB NY 09:30 ET, target 2R; screening M1, non attivata
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
input string InpBTCsymbol="BTCUSD";
input bool InpNASOnlyNY=true;
// prova500: NAS100 baseline stessa BTC474/495/497, screening OHLC M1 gennaio2025 in attesa tick reali; nessuna selezione OOS e nessuna attivazione
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova497: NAS100, ripetizione nativa della495 dopo download M1 e riavvio isolato; stessi input, nessuna ottimizzazione o attivazione
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova495: Nuovo obiettivo NAS100; baseline stessa strategia della BTC474, stessi segnali/H4/bande/runner; soli giorni aperti; costi Afterprime; non ancora adattata a FTMO
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova474: Verifica nativa Model4 ORB+BAND selezionato, input congelati; periodo riutilizzato, non OOS
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova467: ORB_BAND; ORB prima, rischio bande65% del rischio base1%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova336: Verifica nativa Model4 miglior variante328-335 entroDD15, input invariati; non OOS
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova328: PROTECTED_STACK; controllo risk1%,nessuno stack/uscita stale
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova319: Verifica nativa Model4 del migliore ORB_EXITS eleggibile; stessi input, non OOS
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova315: ORB_EXITS; 10R,trailing4R/gap3R,rischio0.75%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova307: ORB_SESSION_TIME; NY16:30,London10/Asia03fissi server;H4,rischio0.75%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova277: ORB H4 EMA12/36, efficienza>=0.15, rischio1.25%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova266: Controllo su tick reali della264, rischio weekend dimezzato e parametri invariati
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova264: ORB3 asimmetrica: rischio weekend dimezzato, max1 posizione
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova220: ORB3 sessioni asimmetrica, rischio1.25%, long dimezzato
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova209: Rischio long dimezzato, rischio short invariato
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova198: Maggiore frequenza: range minimo 0.05 ATR, rischio 0.75%
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova188: ORB continuazione, uscite 3R, senza BE e senza piramide
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova186: Prova185 con break-even allineato alla griglia tick e verifica distanza minima del contratto crypto
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
// prova185: Adattamento iniziale di prova130 alle crypto, sette giorni su sette
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
//+------------------------------------------------------------------+
//|                              Tri_Session_Master_v5.5_Fade.mq5    |
//|                   Logic: London/NY/Asia ORB - Sniper Filters     |
//|                   + FADE dei falsi breakout + gestione dinamica  |
//|                                                                    |
//|   v5.5 NOVITA':                                                   |
//|    - LONDON_FADE RIMOSSA: se il breakout su Londra ha corpo       |
//|      debole, non si apre nessun trade fade. La continuazione     |
//|      resta invariata. NY e Asia mantengono la fade come prima.   |
//+------------------------------------------------------------------+
#property copyright "Tri-Session Master v5.5 Fade"
#property version   "5.50"
#property strict

#include <Trade\Trade.mqh>

// ═══════════════════════════════════════════
// 📥 INPUTS
// ═══════════════════════════════════════════
input group "London Session (ORB)"
input int      InpLon_StartHour     = 10;
input int      InpLon_StartMin      = 0;
input int      InpLon_DurationMin   = 60;
input bool     InpLon_UseDailyFilter= true;
input bool     InpLon_IgnoreBias    = true;

input group "New York Session (ORB)"
input int      InpNY_StartHour      = 16;
input int      InpNY_StartMin       = 30;
input int      InpNY_DurationMin    = 30;
input bool     InpNY_UseDailyFilter = false;

input group "Asian Session (ORB)"
input bool     InpUseAsia            = false;
input int      InpAsia_StartHour     = 3;
input int      InpAsia_StartMin      = 0;
input int      InpAsia_DurationMin   = 60;
input bool     InpAsia_UseDailyFilter= false;

input group "Sniper Filters (Crucial)"
input ENUM_TIMEFRAMES InpEntryTF    = PERIOD_M5;
input int      InpMaxEntryDelayMin  = 360;
input double   InpBodyFilterPct     = 50.0;

input group "Fade dei Falsi Breakout (NON si applica a LONDON)"
input bool     InpFadeEnabled       = false;
input double   InpFadeRiskPct       = 50.0;
input double   InpFadeSLBufferPoints= 30;

input double InpATRMinRangeMultAsia = 0.05;
input double InpATRMinRangeMultNY = 0.05;

input group "Volatility Filter"
input bool     InpUseATRFilter      = true;
input int      InpATRPeriod         = 14;
input double   InpATRMinRangeMult   = 0.05;

input group "Break-Even & Partial Close"
input bool     InpUseBreakEven      = false;
input double   InpBreakEvenTriggerR = 1.0;
input double   InpBreakEvenLockR    = 0.1;
input bool     InpUsePartialClose   = false;
input double   InpPartialClosePct   = 50.0;
input double   InpPartialCloseAtR   = 1.0;

input group "Risk & Money Management"
input double   InpFixedRiskAmount   = 80.0;
input bool     InpUseEquityRisk     = true;
input double   InpRiskPercent       = 1.0;
input double   InpRR_Target         = 2.0;
input int      InpMaxLossesDay      = 3;
input int      InpMaxTradesDay      = 6;
input ulong    InpMagicNumber       = 20261010503;
input bool     InpShowComments      = false;

input group "Daily Loss Control"
input bool     InpUseEquityDailyLock = false;
input double   InpMaxDailyLossPct    = 3.0;

// ═══════════════════════════════════════════
// 🧠 GLOBALS
// ══════════════════════════════════════════
input double InpPyramidAtR=1.2;
input double InpPyramidRiskPercent=0.0;
input double InpPyramidExtraTargetR=1.0;
input double InpTotalOpenStopRiskPct=3.5;
double g_profitPyramid=0; int g_winsPyramid=0,g_totalPyramid=0;
input double InpAsiaRiskMultiplier=1.0;
input double InpNYRiskMultiplier=1.0;
input double InpLondonRiskMultiplier=1.0;
input group "Crypto: operativita e costo dello spread"
input bool InpTradeWeekend=false;
input double InpMaxSpreadToStopPct=10.0;
input bool InpRequireWeekendSessions=false;
bool g_lonLoaded=false,g_nyLoaded=false,g_asiaLoaded=false;
input group "Crypto: durata, direzione e rischio aggregato"
input double InpCryptoMaxHoldHours=0.0;
input bool InpCryptoAllowLong=true;
input bool InpCryptoAllowShort=true;
input double InpCryptoLongRiskMult=1.0;
input double InpCryptoShortRiskMult=1.0;
input double InpCryptoMaxStopATR=0.0;
input int InpCryptoMaxPositions=1;
input bool InpCryptoInitializeDailyBias=false;
input double InpCryptoWeekendRiskMult=0.75;
CTrade trade;
// Protezione comune FTMO 2-Step. Bozza di ricerca: ordini fuori tester disabilitati.
// Le soglie riguardano l'intero conto, incluse altre posizioni e commissioni.
enum ENUM_PORTFOLIO_DST { PORTFOLIO_DST_NONE=0, PORTFOLIO_DST_EU=1, PORTFOLIO_DST_US=2 };
input group "FTMO 2-Step: controllo comune del conto"
input double InpPortfolioInitialCapital=10000.0;
input double InpPortfolioDailyLoss=500.0;
input double InpPortfolioTotalLoss=1000.0;
input double InpPortfolioSafetyBuffer=100.0;
input double InpPortfolioCostReserve=8.0;
input double InpPortfolioRiskCap=100.0;
input string InpPortfolioOwnedMagics="";
input string InpPortfolioStateID="FTMO_20261011";
input bool InpPortfolioPermitAccountOrders=false;
input bool InpPortfolioClockConfirmed=false;
input bool InpPortfolioHistoryConfirmed=false;
input int InpPortfolioUTCWinter=2;
input int InpPortfolioUTCSummer=3;
input ENUM_PORTFOLIO_DST InpPortfolioDST=PORTFOLIO_DST_US;
input bool InpPortfolioCloseLongBreaks=true;
input int InpPortfolioMarketCloseBufferMin=15;
input int InpPortfolioMidnightBufferMin=15;

int g_portfolioDay=0,g_portfolioLockedDay=0;
double g_portfolioMidnight=0,g_portfolioMinEquity=DBL_MAX;
double g_portfolioWorstDaily=0;
bool g_portfolioTotalStop=false,g_portfolioFailed=false;
long g_portfolioOwned[];
int g_portfolioAuditDays[];
double g_portfolioAuditMidnight[],g_portfolioAuditMinimum[];
CTrade g_portfolioTrade;

datetime PortfolioDate(int year,int month,int day,int hour=0){
 MqlDateTime d={};d.year=year;d.mon=month;d.day=day;d.hour=hour;return StructToTime(d);
}
datetime PortfolioEUChange(int year,int month){
 int next=month+1,y=year;if(next==13){next=1;y++;}
 MqlDateTime d;TimeToStruct(PortfolioDate(y,next,1)-86400,d);
 return PortfolioDate(year,month,d.day-d.day_of_week,1);
}
datetime PortfolioUSChange(int year,int month){
 MqlDateTime d;TimeToStruct(PortfolioDate(year,month,1),d);
 int day=1+(7-d.day_of_week)%7+(month==3?7:0);
 return PortfolioDate(year,month,day,month==3?7:6);
}
int PortfolioServerOffset(datetime server){
 if(InpPortfolioDST==PORTFOLIO_DST_NONE)return InpPortfolioUTCWinter;
 MqlDateTime d;TimeToStruct(server,d);
 datetime start=InpPortfolioDST==PORTFOLIO_DST_US?PortfolioUSChange(d.year,3):PortfolioEUChange(d.year,3);
 datetime end=InpPortfolioDST==PORTFOLIO_DST_US?PortfolioUSChange(d.year,11):PortfolioEUChange(d.year,10);
 start+=InpPortfolioUTCWinter*3600;end+=InpPortfolioUTCSummer*3600;
 return server>=start && server<end?InpPortfolioUTCSummer:InpPortfolioUTCWinter;
}
datetime PortfolioCET(datetime server){
 static datetime minute=0;static int correction=0;
 datetime current=(datetime)(server/60);
 if(current!=minute){
  minute=current;datetime utc=server-PortfolioServerOffset(server)*3600;
  MqlDateTime d;TimeToStruct(utc,d);
  int cet=utc>=PortfolioEUChange(d.year,3) && utc<PortfolioEUChange(d.year,10)?2:1;
  correction=(cet-PortfolioServerOffset(server))*3600;
 }
 return server+correction;
}
int PortfolioDayKey(datetime server){
 static datetime begin=0;static int key=0;
 datetime local=PortfolioCET(server);
 if(local<begin || local>=begin+86400){
  MqlDateTime d;TimeToStruct(local,d);key=d.year*10000+d.mon*100+d.day;
  d.hour=0;d.min=0;d.sec=0;begin=StructToTime(d);
 }
 return key;
}
string PortfolioKey(string suffix){
 // Il login puo' essere uguale su server diversi: includere un hash del server.
 string server=AccountInfoString(ACCOUNT_SERVER);uint hash=2166136261;
 for(int i=0;i<StringLen(server);i++){hash^=(uint)StringGetCharacter(server,i);hash*=16777619;}
 return StringFormat("%s_%I64d_%u_%s",InpPortfolioStateID,AccountInfoInteger(ACCOUNT_LOGIN),hash,suffix);
}
bool PortfolioOwn(long magic){
 for(int i=0;i<ArraySize(g_portfolioOwned);i++)if(g_portfolioOwned[i]==magic)return true;
 return false;
}
void PortfolioCloseOwn(){
 for(int i=OrdersTotal()-1;i>=0;i--){
  ulong ticket=OrderGetTicket(i);
  if(ticket>0 && PortfolioOwn(OrderGetInteger(ORDER_MAGIC))){
   if(!g_portfolioTrade.OrderDelete(ticket) || g_portfolioTrade.ResultRetcode()!=TRADE_RETCODE_DONE)
    PrintFormat("PORTFOLIO_PENDING_DELETE_FAILED %I64u ret=%u",ticket,g_portfolioTrade.ResultRetcode());
  }
 }
 for(int i=PositionsTotal()-1;i>=0;i--){
  ulong ticket=PositionGetTicket(i);
  if(ticket>0 && PortfolioOwn(PositionGetInteger(POSITION_MAGIC))){
   g_portfolioTrade.SetTypeFillingBySymbol(PositionGetString(POSITION_SYMBOL));
   bool sent=g_portfolioTrade.PositionClose(ticket);
   if(!sent || (g_portfolioTrade.ResultRetcode()!=TRADE_RETCODE_DONE && g_portfolioTrade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))
    PrintFormat("PORTFOLIO_CLOSE_FAILED %I64u ret=%u",ticket,g_portfolioTrade.ResultRetcode());
  }
 }
}
double PortfolioFloor(){return MathMax(InpPortfolioInitialCapital-InpPortfolioTotalLoss,g_portfolioMidnight-InpPortfolioDailyLoss);}
bool PortfolioRebuildMidnight(){
 datetime now=TimeCurrent();int key=PortfolioDayKey(now);
 // Due giorni server comprendono la mezzanotte CET anche nelle settimane di cambio DST.
 if(!HistorySelect(now-3*86400,now))return false;
 double delta=0;
 for(int i=0;i<HistoryDealsTotal();i++){
  ulong deal=HistoryDealGetTicket(i);
  if(PortfolioDayKey((datetime)HistoryDealGetInteger(deal,DEAL_TIME))!=key)continue;
  long type=HistoryDealGetInteger(deal,DEAL_TYPE);
  if(type==DEAL_TYPE_CREDIT)continue;
  // Il deposito iniziale virtuale non e' profitto della prima giornata.
  // Queste prove non effettuano versamenti, prelievi o reset nel tester.
  if(MQLInfoInteger(MQL_TESTER) && type==DEAL_TYPE_BALANCE)continue;
  delta+=HistoryDealGetDouble(deal,DEAL_PROFIT)+HistoryDealGetDouble(deal,DEAL_SWAP)+HistoryDealGetDouble(deal,DEAL_COMMISSION)+HistoryDealGetDouble(deal,DEAL_FEE);
 }
 g_portfolioMidnight=AccountInfoDouble(ACCOUNT_BALANCE)-delta;g_portfolioDay=key;
 if(!MQLInfoInteger(MQL_TESTER)){
  if(GlobalVariableCheck(PortfolioKey("DAY")) && (int)GlobalVariableGet(PortfolioKey("DAY"))==key && GlobalVariableCheck(PortfolioKey("MIDNIGHT")))
   g_portfolioMidnight=MathMax(g_portfolioMidnight,GlobalVariableGet(PortfolioKey("MIDNIGHT")));
  GlobalVariableSet(PortfolioKey("DAY"),key);GlobalVariableSet(PortfolioKey("MIDNIGHT"),g_portfolioMidnight);
 }
 return g_portfolioMidnight>0;
}
void PortfolioObserve(){
 double equity=AccountInfoDouble(ACCOUNT_EQUITY);
 g_portfolioMinEquity=MathMin(g_portfolioMinEquity,equity);
 g_portfolioWorstDaily=MathMax(g_portfolioWorstDaily,g_portfolioMidnight-equity);
 int n=ArraySize(g_portfolioAuditDays),index=n-1;
 if(n==0 || g_portfolioAuditDays[index]!=g_portfolioDay){
  index=n;ArrayResize(g_portfolioAuditDays,n+1);ArrayResize(g_portfolioAuditMidnight,n+1);ArrayResize(g_portfolioAuditMinimum,n+1);
  g_portfolioAuditDays[index]=g_portfolioDay;g_portfolioAuditMidnight[index]=g_portfolioMidnight;g_portfolioAuditMinimum[index]=equity;
 }
 g_portfolioAuditMinimum[index]=MathMin(g_portfolioAuditMinimum[index],equity);
 if(equity<PortfolioFloor()){
  if(!g_portfolioFailed)PrintFormat("PORTFOLIO_FTMO_BREACH equity=%.2f floor=%.2f",equity,PortfolioFloor());
  g_portfolioFailed=true;
  if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(PortfolioKey("FAILED"),1);
 }
}
double PortfolioRemainingStopRisk(){
 double used=0;
 for(int i=PositionsTotal()-1;i>=0;i--){
  ulong ticket=PositionGetTicket(i);if(ticket==0)continue;
  string symbol=PositionGetString(POSITION_SYMBOL);double stop=PositionGetDouble(POSITION_SL),pnl=0;
  bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;MqlTick q;
  if(stop<=0 || !SymbolInfoTick(symbol,q) || q.bid<=0 || q.ask<=0 ||
     !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,PositionGetDouble(POSITION_VOLUME),buy?q.bid:q.ask,stop,pnl))return DBL_MAX;
  used+=MathMax(0,-pnl)+InpPortfolioCostReserve;
 }
 // Nessun ingresso aggiuntivo con pendenti esterni: il loro rischio non e' noto.
 if(OrdersTotal()>0)return DBL_MAX;
 return used;
}
bool PortfolioComputeMarketBreak(string symbol){
 if(!InpPortfolioCloseLongBreaks)return false;
 datetime now=TimeCurrent();MqlDateTime d;TimeToStruct(now,d);
 datetime midnight=PortfolioDate(d.year,d.mon,d.day),end=0;bool inside=false;
 for(int delta=-1;delta<=0;delta++){
  int weekday=(d.day_of_week+delta+7)%7;datetime from,to;
  for(uint n=0;SymbolInfoSessionTrade(symbol,(ENUM_DAY_OF_WEEK)weekday,n,from,to);n++){
   datetime begin=(datetime)((long)midnight+delta*86400+(long)from);
   datetime finish=(datetime)((long)midnight+delta*86400+(long)to);
   if(finish<=begin)finish+=86400;
   if(now>=begin && now<finish){inside=true;if(finish>end)end=finish;}
  }
 }
 if(!inside)return true;
 if(end-now>InpPortfolioMarketCloseBufferMin*60)return false;
 datetime next=0;
 for(int delta=0;delta<=7;delta++){
  datetime from,to;int weekday=(d.day_of_week+delta)%7;
  for(uint n=0;SymbolInfoSessionTrade(symbol,(ENUM_DAY_OF_WEEK)weekday,n,from,to);n++){
   datetime begin=(datetime)((long)midnight+delta*86400+(long)from);
   if(begin>=end && (next==0 || begin<next))next=begin;
  }
 }
 return next==0 || next-end>2*3600;
}
bool PortfolioMarketBreak(string symbol){
 // Le sessioni sono espresse al minuto: evita scansioni su ogni tick reale.
 static string symbols[];static long minutes[];static bool blocked[];
 long minute=(long)(TimeCurrent()/60);int index=-1;
 for(int i=0;i<ArraySize(symbols);i++)if(symbols[i]==symbol){index=i;break;}
 if(index<0){index=ArraySize(symbols);ArrayResize(symbols,index+1);ArrayResize(minutes,index+1);ArrayResize(blocked,index+1);symbols[index]=symbol;minutes[index]=-1;}
 if(minutes[index]!=minute){minutes[index]=minute;blocked[index]=PortfolioComputeMarketBreak(symbol);}
 return blocked[index];
}
bool PortfolioUpdate(){
 if(!MQLInfoInteger(MQL_TESTER) && (!InpPortfolioPermitAccountOrders || !InpPortfolioClockConfirmed || !InpPortfolioHistoryConfirmed))return false;
 if(g_portfolioDay!=PortfolioDayKey(TimeCurrent()) && !PortfolioRebuildMidnight())return false;
 PortfolioObserve();
 double equity=AccountInfoDouble(ACCOUNT_EQUITY);
 bool total=equity<=InpPortfolioInitialCapital-InpPortfolioTotalLoss+InpPortfolioSafetyBuffer;
 bool daily=equity<=g_portfolioMidnight-InpPortfolioDailyLoss+InpPortfolioSafetyBuffer;
 if(total)g_portfolioTotalStop=true;
 if(daily)g_portfolioLockedDay=g_portfolioDay;
 if(!MQLInfoInteger(MQL_TESTER)){
  if(g_portfolioTotalStop)GlobalVariableSet(PortfolioKey("TOTAL_STOP"),1);
  if(daily)GlobalVariableSet(PortfolioKey("LOCKDAY"),g_portfolioDay);
 }
 if(g_portfolioFailed || g_portfolioTotalStop || g_portfolioLockedDay==g_portfolioDay){PortfolioCloseOwn();PortfolioObserve();return false;}
 // Chiudere solo i ticket del gruppo prima delle lunghe pause dello strumento.
 for(int i=PositionsTotal()-1;i>=0;i--){
  ulong ticket=PositionGetTicket(i);if(ticket==0 || !PortfolioOwn(PositionGetInteger(POSITION_MAGIC)))continue;
  string symbol=PositionGetString(POSITION_SYMBOL);
  if(PortfolioMarketBreak(symbol)){
   g_portfolioTrade.SetTypeFillingBySymbol(symbol);
   if(!g_portfolioTrade.PositionClose(ticket))PrintFormat("PORTFOLIO_MARKET_CLOSE_FAILED %I64u",ticket);
  }
 }
 MqlDateTime cet;TimeToStruct(PortfolioCET(TimeCurrent()),cet);
 if(cet.hour*60+cet.min>=1440-InpPortfolioMidnightBufferMin){
  double nextFloor=MathMax(InpPortfolioInitialCapital-InpPortfolioTotalLoss,AccountInfoDouble(ACCOUNT_BALANCE)-InpPortfolioDailyLoss);
  double used=PortfolioRemainingStopRisk();
  if(used==DBL_MAX || AccountInfoDouble(ACCOUNT_EQUITY)-used<=nextFloor+InpPortfolioSafetyBuffer)PortfolioCloseOwn();
  return false;
 }
 return true;
}
double PortfolioEntryRisk(double requested,string symbol){
 if(!PortfolioUpdate() || PortfolioMarketBreak(symbol))return 0;
 double used=PortfolioRemainingStopRisk();if(used==DBL_MAX)return 0;
 double capacity=AccountInfoDouble(ACCOUNT_EQUITY)-PortfolioFloor()-InpPortfolioSafetyBuffer-used-InpPortfolioCostReserve;
 return MathMax(0,MathMin(MathMin(requested,InpPortfolioRiskCap),capacity));
}
bool PortfolioInit(ulong ownMagic){
 if(InpPortfolioInitialCapital<=0 || InpPortfolioDailyLoss<=0 || InpPortfolioTotalLoss<=0 || InpPortfolioRiskCap<=0 ||
    InpPortfolioSafetyBuffer<0 || InpPortfolioSafetyBuffer>=MathMin(InpPortfolioDailyLoss,InpPortfolioTotalLoss) || InpPortfolioCostReserve<0 ||
    InpPortfolioMarketCloseBufferMin<1 || InpPortfolioMidnightBufferMin<1)return false;
 if(AccountInfoString(ACCOUNT_CURRENCY)!="EUR"){Print("PORTFOLIO_INIT: preset in EUR, conto non EUR");return false;}
 if(!MQLInfoInteger(MQL_TESTER) && (!InpPortfolioPermitAccountOrders || !InpPortfolioClockConfirmed || !InpPortfolioHistoryConfirmed)){
  Print("PORTFOLIO_ACCOUNT_ORDERS_DISABLED: versione di ricerca, serve attivazione esplicita e verifiche");return false;
 }
 ArrayResize(g_portfolioOwned,1);g_portfolioOwned[0]=(long)ownMagic;
 string magics[];int n=StringSplit(InpPortfolioOwnedMagics,',',magics);
 for(int i=0;i<n;i++){long value=StringToInteger(magics[i]);if(value<=0)continue;int k=ArraySize(g_portfolioOwned);ArrayResize(g_portfolioOwned,k+1);g_portfolioOwned[k]=value;}
 if(!MQLInfoInteger(MQL_TESTER)){
  g_portfolioFailed=GlobalVariableCheck(PortfolioKey("FAILED"));g_portfolioTotalStop=GlobalVariableCheck(PortfolioKey("TOTAL_STOP"));
  if(GlobalVariableCheck(PortfolioKey("LOCKDAY")))g_portfolioLockedDay=(int)GlobalVariableGet(PortfolioKey("LOCKDAY"));
 }
 // Inizializzazione differita al primo tick: nel tester lo storico dei deal
 // puo' essere vuoto durante OnInit, prima del deposito iniziale simulato.
 g_portfolioDay=0;return true;
}
void PortfolioExport(){
 if(g_portfolioMinEquity==DBL_MAX){Print("PORTFOLIO_AUDIT_NO_OBSERVATIONS");return;}
 string name=MQLInfoString(MQL_PROGRAM_NAME)+"_portfolio_daily.csv";
 int file=FileOpen(name,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(file==INVALID_HANDLE)return;
 FileWrite(file,"day_cet","midnight_balance","minimum_equity","daily_loss","total_floor","daily_floor");
 for(int i=0;i<ArraySize(g_portfolioAuditDays);i++)FileWrite(file,g_portfolioAuditDays[i],DoubleToString(g_portfolioAuditMidnight[i],2),DoubleToString(g_portfolioAuditMinimum[i],2),DoubleToString(MathMax(0,g_portfolioAuditMidnight[i]-g_portfolioAuditMinimum[i]),2),DoubleToString(InpPortfolioInitialCapital-InpPortfolioTotalLoss,2),DoubleToString(g_portfolioAuditMidnight[i]-InpPortfolioDailyLoss,2));
 FileClose(file);
 PrintFormat("PORTFOLIO_AUDIT minimum=%.2f dailyWorst=%.2f breach=%s totalStop=%s",g_portfolioMinEquity,g_portfolioWorstDaily,g_portfolioFailed?"true":"false",g_portfolioTotalStop?"true":"false");
}

int    g_atrHandle = INVALID_HANDLE;

double g_lonHigh = 0.0, g_lonLow = 999999.0;
bool   g_lonRangeDefined = false;
bool   g_lonExecutedToday = false;

double g_nyHigh = 0.0, g_nyLow = 999999.0;
bool   g_nyRangeDefined = false;
bool   g_nyExecutedToday = false;

double g_asiaHigh = 0.0, g_asiaLow = 999999.0;
bool   g_asiaRangeDefined = false;
bool   g_asiaExecutedToday = false;

int    g_dailyBias = 0;
double g_daily_swing_highs[];
double g_daily_swing_lows[];
int    InpSwingLen_Daily = 5;

datetime g_currentDay = 0;
double   g_dayStartEquity = 0.0;
int      g_tradesToday = 0;
int      g_lossesToday = 0;
bool     g_dayLocked = false;
long     g_countedLossIds[];

struct SessionMap { long posId; string session; double entryPrice; double slDistance; int dir; bool beDone; bool partialDone; bool pyramidDone; };
SessionMap g_posSessionMap[];

// Stats continuazione
double g_profitLon = 0, g_profitNY = 0, g_profitAsia = 0;
int    g_winsLon = 0, g_winsNY = 0, g_winsAsia = 0;
int    g_totalLon = 0, g_totalNY = 0, g_totalAsia = 0;

// Stats fade (NY/Asia soltanto)
double g_profitNYFade = 0, g_profitAsiaFade = 0;
int    g_winsNYFade = 0, g_winsAsiaFade = 0;
int    g_totalNYFade = 0, g_totalAsiaFade = 0;

// Stats reversal - doppio fakeout (NY soltanto, Londra non ha piu' fade quindi non arma mai il reversal)
double g_profitNYRev = 0;
int    g_winsNYRev = 0;
int    g_totalNYRev = 0;

// Stato per il rientro reversal dopo stop della fade (NY)
int      g_nyFadeDir = 0;    double g_nyFadeSL = 0;
bool     g_nyReArmActive = false;  datetime g_nyReArmDeadline = 0;

// ═══════════════════════════════════════════
// 🚀 INIT & DEINIT
// ══════════════════════════════════════════

input group "Crypto: regime su candele concluse"
input ENUM_TIMEFRAMES InpRegimeTF=PERIOD_H4;
input int InpRegimeFast=12;
input int InpRegimeSlow=36;
input int InpRegimeSlopeBars=3;
input int InpRegimeEfficiencyBars=20;
input double InpRegimeMinEfficiency=0.15;
input double InpRegimeStopATR=0.0;
int g_regimeFast=INVALID_HANDLE,g_regimeSlow=INVALID_HANDLE,g_regimeATR=INVALID_HANDLE;
datetime g_regimeBar=0;
int g_regimeDirection=0;
double g_regimeStopDistance=0;

// Mai barra0 per segnali: refresh solo con buffer disponibili e barre chiuse.
bool CryptoRegimeAllows(int direction){
 datetime bar=iTime(_Symbol,InpRegimeTF,0);
 if(bar<=0)return false;
 if(bar!=g_regimeBar){
  double fast[],slow[],atr[],closes[];
  int nf=InpRegimeSlopeBars+1,nc=InpRegimeEfficiencyBars+1;
  if(CopyBuffer(g_regimeFast,0,1,nf,fast)!=nf ||
     CopyBuffer(g_regimeSlow,0,1,1,slow)!=1 ||
     CopyBuffer(g_regimeATR,0,1,1,atr)!=1 ||
     CopyClose(_Symbol,InpRegimeTF,1,nc,closes)!=nc)return false;
  // CopyBuffer/CopyClose: il valore piu vecchio e' all'indice0.
  double latest=closes[nc-1],effort=0;
  for(int i=1;i<nc;i++)effort+=MathAbs(closes[i]-closes[i-1]);
  double efficiency=effort>0?MathAbs(latest-closes[0])/effort:0;
  int signal=0;
  if(efficiency>=InpRegimeMinEfficiency && atr[0]>0){
   if(latest>fast[nf-1] && fast[nf-1]>slow[0] && fast[nf-1]>fast[0])signal=1;
   if(latest<fast[nf-1] && fast[nf-1]<slow[0] && fast[nf-1]<fast[0])signal=-1;
  }
  g_regimeDirection=signal;g_regimeStopDistance=atr[0]*InpRegimeStopATR;
  g_regimeBar=bar;
 }
 return direction==g_regimeDirection;
}


input group "Crypto: uscita in multipli del rischio iniziale"
input double InpTrailActivateR=4.0;
input double InpTrailGapR=3.0;

// Stop monotono: quote correnti, distanza iniziale registrata al fill.
// Una modifica per minuto; nessun massimo futuro o candela in formazione.
void CryptoTrailInitialRisk(){
 if(InpTrailActivateR<=0)return;
 static datetime checked=0;
 datetime minute=(datetime)(TimeCurrent()/60);
 if(minute==checked)return;
 checked=minute;
 MqlTick quote;if(!SymbolInfoTick(_Symbol,quote))return;
 double grid=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
 double point=SymbolInfoDouble(_Symbol,SYMBOL_POINT);
 double gap=MathMax(SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL))*point;
 if(grid<=0 || point<=0)return;
 int digits=(int)SymbolInfoInteger(_Symbol,SYMBOL_DIGITS);
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  int index=FindSessionMapIndex(PositionGetInteger(POSITION_IDENTIFIER));
  if(index<0)continue;
  double initialRisk=g_posSessionMap[index].slDistance;
  double entry=g_posSessionMap[index].entryPrice;
  int dir=g_posSessionMap[index].dir;
  double price=dir==1?quote.bid:quote.ask;
  if(initialRisk<=0 || dir*(price-entry)<InpTrailActivateR*initialRisk)continue;
  double stop=PositionGetDouble(POSITION_SL),target=PositionGetDouble(POSITION_TP);
  double candidate=price-dir*InpTrailGapR*initialRisk;
  candidate=NormalizeDouble((dir==1?MathFloor(candidate/grid):MathCeil(candidate/grid))*grid,digits);
  if((dir==1 && (candidate<=stop || candidate>=quote.bid-gap || candidate>=target)) ||
     (dir==-1 && ((stop>0 && candidate>=stop) || candidate<=quote.ask+gap || candidate<=target)))continue;
  if(trade.PositionModify(ticket,candidate,target) && trade.ResultRetcode()==TRADE_RETCODE_DONE){
   if(dir*(candidate-entry)>0)g_posSessionMap[index].beDone=true;
  }
 }
}


input group "Crypto: nuove sessioni con posizione gia protetta"
input bool InpAllowProtectedStack=false;
input double InpStackMinLockR=0.50;
input double InpStackExtraRiskMultiplier=0.50;
input double InpStaleExitHours=0.0;
input double InpStaleMaxR=0.25;

bool CryptoCanOpenSession(){
 int count=CryptoPositionCount();
 if(count==0)return true;
 if(!InpAllowProtectedStack || count>=InpCryptoMaxPositions)return false;
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  int index=FindSessionMapIndex(PositionGetInteger(POSITION_IDENTIFIER));
  if(index<0)return false;
  int dir=g_posSessionMap[index].dir;
  double entry=g_posSessionMap[index].entryPrice,initialRisk=g_posSessionMap[index].slDistance;
  double stop=PositionGetDouble(POSITION_SL),locked=0;
  if(stop<=0 || initialRisk<=0 || dir*(stop-entry)<InpStackMinLockR*initialRisk)return false;
  if(!OrderCalcProfit(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,PositionGetDouble(POSITION_VOLUME),entry,stop,locked) ||
     locked+PositionGetDouble(POSITION_SWAP)<=0)return false;
 }
 return true;
}

bool CryptoStackDirectionAllows(int direction){
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
  if(direction!=dir)return false;
 }
 return true;
}

void CryptoExitStaleTrades(){
 if(InpStaleExitHours<=0)return;
 static datetime checked=0;
 datetime minute=(datetime)(TimeCurrent()/60);if(minute==checked)return;checked=minute;
 MqlTick quote;if(!SymbolInfoTick(_Symbol,quote))return;
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  int index=FindSessionMapIndex(PositionGetInteger(POSITION_IDENTIFIER));
  if(index<0 || g_posSessionMap[index].slDistance<=0)continue;
  if(TimeCurrent()-(datetime)PositionGetInteger(POSITION_TIME)<InpStaleExitHours*3600)continue;
  int dir=g_posSessionMap[index].dir;
  double price=dir==1?quote.bid:quote.ask;
  double achieved=dir*(price-g_posSessionMap[index].entryPrice)/g_posSessionMap[index].slDistance;
  // Il regimeH4 legge barre chiuse. Non chiudere un runner ancora in trend.
  if(achieved>=InpStaleMaxR || CryptoRegimeAllows(dir))continue;
  if(!trade.PositionClose(ticket) || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))
   Print("STALE_CLOSE_REJECTED ",ticket," retcode ",trade.ResultRetcode());
 }
}


input group "Crypto: breakout dopo compressione, senza orari fissi"
input ENUM_TIMEFRAMES InpBandSignalTF=PERIOD_H1;
input int InpBandPeriod=20;
input double InpBandDeviation=2.0;
input int InpBandCompressionBars=48;
input double InpBandCompressionRatio=1.0;
input double InpBandStopATR=2.0;
input double InpBandMaxQuoteGapATR=0.5;
input double InpBandMinCloseLocation=0.60;
input bool InpBandRequireRegime=true;
input double InpBandRiskMultiplier=0.0;
input double InpBandTargetR=10.0;
input bool InpBandPriority=false;
int g_bandHandle=INVALID_HANDLE,g_bandATR=INVALID_HANDLE;
datetime g_bandSignalBar=0;
double g_bandProfit=0;int g_bandTrades=0,g_bandWins=0;

int OnInit() {
 if(!PortfolioInit(InpMagicNumber))return INIT_FAILED;

 if(InpBandRiskMultiplier<0 || InpBandTargetR<=0 || InpCryptoMaxPositions!=1 ||
    InpAllowProtectedStack || InpPyramidRiskPercent>0)return INIT_PARAMETERS_INCORRECT;

 if((InpBandSignalTF!=PERIOD_M15 && InpBandSignalTF!=PERIOD_M30 && InpBandSignalTF!=PERIOD_H1) ||
    InpBandPeriod<5 || InpBandDeviation<=0 || InpBandCompressionBars<5 || InpBandCompressionBars>500 ||
    InpBandCompressionRatio<=0 || InpBandStopATR<=0 || InpBandMaxQuoteGapATR<0 ||
    InpBandMinCloseLocation<0 || InpBandMinCloseLocation>1)return INIT_PARAMETERS_INCORRECT;
 g_bandHandle=iBands(_Symbol,InpBandSignalTF,InpBandPeriod,0,InpBandDeviation,PRICE_CLOSE);
 g_bandATR=iATR(_Symbol,InpBandSignalTF,14);
 if(g_bandHandle==INVALID_HANDLE || g_bandATR==INVALID_HANDLE)return INIT_FAILED;

 if(InpStackMinLockR<=0 || InpStackExtraRiskMultiplier<=0 || InpStackExtraRiskMultiplier>1 ||
    InpStaleExitHours<0 || InpStaleMaxR<0)return INIT_PARAMETERS_INCORRECT;

 if(InpTrailActivateR<0 || InpTrailGapR<=0 ||
    (InpTrailActivateR>0 && InpTrailGapR>=InpTrailActivateR))return INIT_PARAMETERS_INCORRECT;

 if(InpRegimeFast<2 || InpRegimeSlow<=InpRegimeFast || InpRegimeSlopeBars<1 ||
    InpRegimeEfficiencyBars<2 || InpRegimeMinEfficiency<0 || InpRegimeMinEfficiency>1 ||
    InpRegimeStopATR<0 || (InpRegimeTF!=PERIOD_H1 && InpRegimeTF!=PERIOD_H4))return INIT_PARAMETERS_INCORRECT;
 g_regimeFast=iMA(_Symbol,InpRegimeTF,InpRegimeFast,0,MODE_EMA,PRICE_CLOSE);
 g_regimeSlow=iMA(_Symbol,InpRegimeTF,InpRegimeSlow,0,MODE_EMA,PRICE_CLOSE);
 g_regimeATR=iATR(_Symbol,InpRegimeTF,14);
 if(g_regimeFast==INVALID_HANDLE || g_regimeSlow==INVALID_HANDLE || g_regimeATR==INVALID_HANDLE)return INIT_FAILED;

    if(InpCryptoWeekendRiskMult<0 || InpCryptoWeekendRiskMult>1)return INIT_PARAMETERS_INCORRECT;
    if(InpCryptoMaxHoldHours<0 || InpCryptoMaxPositions<1 || InpCryptoMaxStopATR<0 || InpCryptoLongRiskMult<0 || InpCryptoShortRiskMult<0)return INIT_PARAMETERS_INCORRECT;
    if(InpCryptoMaxPositions>1 && AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)return INIT_PARAMETERS_INCORRECT;
    if(InpRequireWeekendSessions) {
        datetime from,to;
        if(!SymbolInfoSessionTrade(_Symbol,SATURDAY,0,from,to) || !SymbolInfoSessionTrade(_Symbol,SUNDAY,0,from,to)) {
            Print("CRYPTO_INIT_FAILED: il simbolo non ha sessioni sia sabato sia domenica: ",_Symbol);
            return INIT_PARAMETERS_INCORRECT;
        }
    }
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFillingBySymbol(_Symbol);

    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    ArrayResize(g_posSessionMap, 0);

    g_atrHandle = iATR(_Symbol, PERIOD_D1, InpATRPeriod);
    if(g_atrHandle == INVALID_HANDLE) Print("ATTENZIONE: impossibile creare handle ATR, filtro volatilità disattivato di fatto.");

    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

    RestoreDailyStats();

    for(int p=PositionsTotal()-1;p>=0;p--){
     ulong ticket=PositionGetTicket(p);if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
     long id=PositionGetInteger(POSITION_IDENTIFIER);double entry=PositionGetDouble(POSITION_PRICE_OPEN),initial=0;
     int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
     if(HistorySelectByPosition(id)){
      for(int j=0;j<HistoryOrdersTotal();j++){
       ulong order=HistoryOrderGetTicket(j);long type=HistoryOrderGetInteger(order,ORDER_TYPE);
       if(type!=ORDER_TYPE_BUY && type!=ORDER_TYPE_SELL)continue;
       double stop=HistoryOrderGetDouble(order,ORDER_SL);
       if(stop>0 && dir*(entry-stop)>0){initial=MathAbs(entry-stop);break;}
      }
     }
     if(initial<=0){Print("PORTFOLIO_RESTART_INITIAL_RISK_UNKNOWN: nuovi ingressi bloccati");return INIT_FAILED;}
     string comment=PositionGetString(POSITION_COMMENT),parts[];StringSplit(comment,'|',parts);
     RegisterPosition(id,ArraySize(parts)>0?parts[0]:"RESTORED",entry,initial,dir);
    }

    Print("Tri-Session Master v5.5 (Londra senza fade) Ready.");
    return(INIT_SUCCEEDED);
}

void OnDeinit(const int reason) {
 PortfolioExport();

 if(g_bandHandle!=INVALID_HANDLE)IndicatorRelease(g_bandHandle);
 if(g_bandATR!=INVALID_HANDLE)IndicatorRelease(g_bandATR);

 if(g_regimeFast!=INVALID_HANDLE)IndicatorRelease(g_regimeFast);
 if(g_regimeSlow!=INVALID_HANDLE)IndicatorRelease(g_regimeSlow);
 if(g_regimeATR!=INVALID_HANDLE)IndicatorRelease(g_regimeATR);

    PrintFinalReport();
    if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
    Comment("");
}

// ═══════════════════════════════════════════
// 🔧 UTILS & STATS
// ═══════════════════════════════════════════
void AddToArray(double &arr[], double val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }
void ShiftArray(double &arr[]) {
    if(ArraySize(arr)<=1){ArrayResize(arr,0);return;}
    for(int i=0;i<ArraySize(arr)-1;i++) arr[i]=arr[i+1];
    ArrayResize(arr,ArraySize(arr)-1);
}
void AddToLongArray(long &arr[], long val) { int s=ArraySize(arr); ArrayResize(arr,s+1); arr[s]=val; }

double GetRiskAmount() {
    if(InpUseEquityRisk) return AccountInfoDouble(ACCOUNT_EQUITY) * (InpRiskPercent/100.0);
    return InpFixedRiskAmount;
}

double GetATRValue() {
    if(g_atrHandle == INVALID_HANDLE) return -1;
    double buf[];
    ArraySetAsSeries(buf, true);
    if(CopyBuffer(g_atrHandle, 0, 1, 1, buf) <= 0) return -1;
    return buf[0];
}

void RegisterPosition(long posId, string session, double entryPrice, double slDistance, int dir) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) if(g_posSessionMap[i].posId == posId) ArrayRemove(g_posSessionMap, i, 1);
    int s = ArraySize(g_posSessionMap); ArrayResize(g_posSessionMap, s+1);
    g_posSessionMap[s].posId = posId;
    g_posSessionMap[s].session = session;
    g_posSessionMap[s].entryPrice = entryPrice;
    g_posSessionMap[s].slDistance = slDistance;
    g_posSessionMap[s].dir = dir;
    g_posSessionMap[s].beDone = false;
    g_posSessionMap[s].partialDone = false;
    g_posSessionMap[s].pyramidDone = false;
}

string GetSessionByPosId(long posId) {
    for(int i=0; i<ArraySize(g_posSessionMap); i++) if(g_posSessionMap[i].posId == posId) return g_posSessionMap[i].session;
    return "UNKNOWN";
}

int FindSessionMapIndex(long posId) {
    for(int i=0; i<ArraySize(g_posSessionMap); i++) if(g_posSessionMap[i].posId == posId) return i;
    return -1;
}

void UnregisterPosition(long posId) {
    for(int i=ArraySize(g_posSessionMap)-1; i>=0; i--) if(g_posSessionMap[i].posId == posId) { ArrayRemove(g_posSessionMap, i, 1); return; }
}

bool IsIdAlreadyCounted(long posId) {
    for(int i=0; i<ArraySize(g_countedLossIds); i++) if(g_countedLossIds[i] == posId) return true;
    return false;
}

bool HasOpenPosition() {
    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == _Symbol && PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) return true;
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
            bool stillOpen = false;
            for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetTicket(p)>0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
            if(!stillOpen && profit < 0 && !IsIdAlreadyCounted(posId)) { g_lossesToday++; AddToLongArray(g_countedLossIds, posId); }
        }
    }
    if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;
}

void PrintSessionStats(string name, double prof, int wins, int total) {
    double wr = (total > 0) ? (double)wins / total * 100.0 : 0.0;
    Print(name, " | Profit: ", DoubleToString(prof, 2), " | Wins: ", IntegerToString(wins), "/", IntegerToString(total), " | WR: ", DoubleToString(wr, 1), "%");
}

void PrintFinalReport() {
 PrintSessionStats("BAND_BREAK",g_bandProfit,g_bandWins,g_bandTrades);
    Print("--------------------------------------------------");
    Print("TRI-SESSION MASTER v5.5 (LONDON NO-FADE) - FINAL REPORT");
    Print("--------------------------------------------------");
    Print("--- Continuazione ---");
    PrintSessionStats("LONDON      ", g_profitLon, g_winsLon, g_totalLon);
    PrintSessionStats("NY          ", g_profitNY, g_winsNY, g_totalNY);
    PrintSessionStats("ASIA        ", g_profitAsia, g_winsAsia, g_totalAsia);
    Print("--- Fade Falsi Breakout (solo NY/Asia) ---");
    PrintSessionStats("NY_FADE     ", g_profitNYFade, g_winsNYFade, g_totalNYFade);
    PrintSessionStats("ASIA_FADE   ", g_profitAsiaFade, g_winsAsiaFade, g_totalAsiaFade);
    Print("--- Reversal dopo stop fade (solo NY) ---");
    PrintSessionStats("NY_REV      ", g_profitNYRev, g_winsNYRev, g_totalNYRev);
    PrintSessionStats("PYRAMID",g_profitPyramid,g_winsPyramid,g_totalPyramid);
    Print("--- Totale ---");
    double totProfit = g_profitLon + g_profitNY + g_profitAsia + g_profitNYFade + g_profitAsiaFade + g_profitNYRev + g_profitPyramid;
    int totWins = g_winsLon + g_winsNY + g_winsAsia + g_winsNYFade + g_winsAsiaFade + g_winsNYRev + g_winsPyramid;
    int totTrades = g_totalLon + g_totalNY + g_totalAsia + g_totalNYFade + g_totalAsiaFade + g_totalNYRev + g_totalPyramid;
    PrintSessionStats("TOTALE      ", totProfit, totWins, totTrades);
    Print("--------------------------------------------------");
}

// ══════════════════════════════════════════
// 🎯 GESTIONE DINAMICA POSIZIONI
// ══════════════════════════════════════════
void ManageOpenPositions() {
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != _Symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        long posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
        int idx = FindSessionMapIndex(posId);
        if(idx < 0) continue;

        double entry  = g_posSessionMap[idx].entryPrice;
        double slDist = g_posSessionMap[idx].slDistance;
        int    dir    = g_posSessionMap[idx].dir;
        if(slDist <= 0) continue;

        double currentPrice = (dir == 1) ? SymbolInfoDouble(_Symbol, SYMBOL_BID) : SymbolInfoDouble(_Symbol, SYMBOL_ASK);
        double rMultiple = (dir == 1) ? (currentPrice - entry) / slDist : (entry - currentPrice) / slDist;

        double volume    = PositionGetDouble(POSITION_VOLUME);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);

        // --- CHIUSURA PARZIALE ---
        if(InpUsePartialClose && !g_posSessionMap[idx].partialDone && rMultiple >= InpPartialCloseAtR) {
            double volMin  = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
            double volStep = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
            double closeVol = MathFloor((volume * (InpPartialClosePct/100.0)) / volStep) * volStep;
            if(closeVol >= volMin && closeVol < volume) {
                if(trade.PositionClosePartial(ticket, closeVol) && (trade.ResultRetcode()==TRADE_RETCODE_DONE || trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL)) {
                    g_posSessionMap[idx].partialDone = true;
                    Print("PARTIAL CLOSE: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Vol: ", DoubleToString(closeVol,2), " @ R=", DoubleToString(rMultiple,2));
                }
            } else {
                g_posSessionMap[idx].partialDone = true;
            }
        }

        // --- BREAK-EVEN ---
        if(InpUseBreakEven && !g_posSessionMap[idx].beDone && rMultiple >= InpBreakEvenTriggerR) {
            double beSL = (dir == 1) ? entry + (InpBreakEvenLockR * slDist) : entry - (InpBreakEvenLockR * slDist);
            double grid=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
            if(grid<=0) continue;
            beSL=NormalizeDouble((dir==1?MathFloor(beSL/grid):MathCeil(beSL/grid))*grid,digits);
            MqlTick live; if(!SymbolInfoTick(_Symbol,live)) continue;
            double minGap=MathMax(SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(_Symbol,SYMBOL_TRADE_FREEZE_LEVEL))*SymbolInfoDouble(_Symbol,SYMBOL_POINT);
            if((dir==1 && beSL>=live.bid-minGap) || (dir==-1 && beSL<=live.ask+minGap)) continue;
            bool improves = (dir == 1) ? (beSL > currentSL) : (currentSL == 0 || beSL < currentSL);
            if(improves) {
                if(trade.PositionModify(ticket, beSL, currentTP) && trade.ResultRetcode()==TRADE_RETCODE_DONE) {
                    g_posSessionMap[idx].beDone = true;
                    Print("BREAK-EVEN: ", g_posSessionMap[idx].session, " | PosID: ", posId, " | Nuovo SL: ", DoubleToString(beSL,digits));
                }
            }
        }
    }
}

// ══════════════════════════════════════════
// 🔄 MAIN LOGIC
// ═══════════════════════════════════════════
// Un solo incremento su una posizione originale gia protetta; nessuna media in perdita.
void ManagePyramids() {
    static datetime checkedBar=0;
    datetime bar=iTime(_Symbol,PERIOD_M1,0);
    if(bar==checkedBar) return;
    checkedBar=bar;
    if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING || g_dayLocked || g_tradesToday>=InpMaxTradesDay) return;
    MqlTick tick; if(!SymbolInfoTick(_Symbol,tick)) return;
    int ownCount=0; double existingStopRisk=0;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        ownCount++;
        int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
        double stop=PositionGetDouble(POSITION_SL),price=dir==1?tick.bid:tick.ask,pnl=0;
        if(stop<=0) return;
        if(!OrderCalcProfit(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,PositionGetDouble(POSITION_VOLUME),price,stop,pnl)) return;
        existingStopRisk+=MathMax(0,-pnl);
    }
    if(ownCount!=1) return;
    double equity=AccountInfoDouble(ACCOUNT_EQUITY);
    double budget=MathMin(equity*InpPyramidRiskPercent/100.0,equity*InpTotalOpenStopRiskPct/100.0-existingStopRisk);
    if(budget<=0) return;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        long id=PositionGetInteger(POSITION_IDENTIFIER);
        int idx=FindSessionMapIndex(id);
        if(idx<0 || g_posSessionMap[idx].pyramidDone || !g_posSessionMap[idx].beDone || StringFind(g_posSessionMap[idx].session,"FADE")>=0 || g_posSessionMap[idx].session=="PYRAMID") continue;
        int dir=g_posSessionMap[idx].dir;
        double entry=g_posSessionMap[idx].entryPrice,distance=g_posSessionMap[idx].slDistance;
        double price=dir==1?tick.bid:tick.ask;
        double achieved=dir==1?price-entry:entry-price;
        double stop=PositionGetDouble(POSITION_SL),target=PositionGetDouble(POSITION_TP);
        if(distance<=0 || achieved<InpPyramidAtR*distance || (dir==1 && stop<=entry) || (dir==-1 && stop>=entry)) continue;
        target+=dir*distance*InpPyramidExtraTargetR;
        int previous=g_tradesToday;
        ExecuteTrade(dir,dir==1?tick.ask:tick.bid,stop,"PYRAMID",budget,target);
        idx=FindSessionMapIndex(id);
        if(idx>=0 && g_tradesToday>previous) g_posSessionMap[idx].pyramidDone=true;
        return;
    }
}

// Range esatto delle M1 concluse: non include la candela precedente alla finestra.
void LoadCryptoORB(datetime start,int minutes,bool &loaded,bool &rangeDefined,double &high,double &low) {
    if(loaded || minutes<=0 || TimeCurrent()<start+minutes*60) return;
    MqlRates bars[];
    int n=CopyRates(_Symbol,PERIOD_M1,start,start+minutes*60-1,bars);
    if(n<=0) return;
    double h=-DBL_MAX,l=DBL_MAX;
    for(int i=0;i<n;i++){h=MathMax(h,bars[i].high);l=MathMin(l,bars[i].low);}
    if(h<=l || l<=0) return;
    high=h;low=l;rangeDefined=true;loaded=true;
}


int CryptoPositionCount(){
 int n=0;for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket>0 && PositionGetString(POSITION_SYMBOL)==_Symbol && PositionGetInteger(POSITION_MAGIC)==(long)InpMagicNumber)n++;
 }return n;
}
bool CryptoSessionOpen(string session){
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket>0 && PositionGetString(POSITION_SYMBOL)==_Symbol && PositionGetInteger(POSITION_MAGIC)==(long)InpMagicNumber && StringFind(PositionGetString(POSITION_COMMENT),session+"|")==0)return true;
 }return false;
}
double CryptoOpenStopRisk(){
 MqlTick quote;if(!SymbolInfoTick(_Symbol,quote))return DBL_MAX;
 double total=0;
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
  double stop=PositionGetDouble(POSITION_SL),pnl=0;
  if(stop<=0 || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,stop,pnl))return DBL_MAX;
  total+=MathMax(0,-pnl);
 }return total;
}
void CryptoExpirePositions(){
 if(InpCryptoMaxHoldHours<=0)return;
 static datetime lastMinute=0;
 datetime minute=(datetime)(TimeCurrent()/60);if(minute==lastMinute)return;lastMinute=minute;
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);
  if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=_Symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  if(TimeCurrent()-(datetime)PositionGetInteger(POSITION_TIME)>=InpCryptoMaxHoldHours*3600){
   if(!trade.PositionClose(ticket) || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))Print("CRYPTO_CLOSE_FAILED ",ticket," retcode ",trade.ResultRetcode());
  }
 }
}
void CryptoWarmDailyBias(){
 if(!InpCryptoInitializeDailyBias)return;
 ArrayResize(g_daily_swing_highs,0);ArrayResize(g_daily_swing_lows,0);
 int oldest=MathMin(300,Bars(_Symbol,PERIOD_D1)-InpSwingLen_Daily-1);
 for(int shift=oldest;shift>=InpSwingLen_Daily+1;shift--){
  double high=GetPivot(PERIOD_D1,InpSwingLen_Daily,shift,true),low=GetPivot(PERIOD_D1,InpSwingLen_Daily,shift,false);
  if(high!=EMPTY_VALUE){AddToArray(g_daily_swing_highs,high);if(ArraySize(g_daily_swing_highs)>10)ShiftArray(g_daily_swing_highs);}
  if(low!=EMPTY_VALUE){AddToArray(g_daily_swing_lows,low);if(ArraySize(g_daily_swing_lows)>10)ShiftArray(g_daily_swing_lows);}
 }g_dailyBias=GetDailyBias();
}


void CryptoTryComplementaryBand(){
 if(g_dayLocked || g_tradesToday>=InpMaxTradesDay || g_lossesToday>=InpMaxLossesDay ||
    CryptoPositionCount()>0 || InpBandRiskMultiplier<=0)return;
 datetime current=iTime(_Symbol,InpBandSignalTF,0);
 if(current<=0 || current==g_bandSignalBar)return;
 MqlRates signal[];ArraySetAsSeries(signal,true);
 if(CopyRates(_Symbol,InpBandSignalTF,1,2,signal)!=2)return;
 int count=InpBandCompressionBars+2;
 double upper[],lower[],atr[];
 if(CopyBuffer(g_bandHandle,1,1,count,upper)!=count ||
    CopyBuffer(g_bandHandle,2,1,count,lower)!=count || CopyBuffer(g_bandATR,0,1,1,atr)!=1)return;
 if(atr[0]==EMPTY_VALUE || !MathIsValidNumber(atr[0]) || atr[0]<=0)return;
 double priorMeanWidth=0;
 for(int i=0;i<count;i++){
  if(upper[i]==EMPTY_VALUE || lower[i]==EMPTY_VALUE || !MathIsValidNumber(upper[i]) ||
     !MathIsValidNumber(lower[i]) || upper[i]<=lower[i])return;
  if(i<count-2)priorMeanWidth+=upper[i]-lower[i];
 }
 priorMeanWidth/=InpBandCompressionBars;
 // Buffer normale:count-1=barra1,count-2=barra2. Media solo barre3..lookback+2.
 // La compressione e' misurata PRIMA della candela che conferma il breakout.
 g_bandSignalBar=current;
 if(upper[count-2]-lower[count-2]>InpBandCompressionRatio*priorMeanWidth)return;
 int direction=0;
 double body=signal[0].close-signal[0].open;
 if(signal[1].close<=upper[count-2] && signal[0].close>upper[count-1] && body>0)direction=1;
 if(signal[1].close>=lower[count-2] && signal[0].close<lower[count-1] && body<0)direction=-1;
 if(direction==0)return;
 double height=signal[0].high-signal[0].low;if(height<=0)return;
 double location=direction==1?(signal[0].close-signal[0].low)/height:(signal[0].high-signal[0].close)/height;
 if(location<InpBandMinCloseLocation)return;
 MqlTick quote;if(!SymbolInfoTick(_Symbol,quote))return;
 double entry=direction==1?quote.ask:quote.bid;
 if(MathAbs(entry-signal[0].close)>atr[0]*InpBandMaxQuoteGapATR)return;
 double stop=entry-direction*atr[0]*InpBandStopATR;
 ExecuteTrade(direction,entry,stop,"BAND_BREAK",GetRiskAmount()*InpBandRiskMultiplier,entry+direction*MathAbs(entry-stop)*InpBandTargetR);
}

void OnTick() {
 if(PortfolioNewsBlocked(_Symbol))return;
 if(!PortfolioUpdate())return;

    CryptoExpirePositions();
    CryptoExitStaleTrades();
    ManageOpenPositions();
    CryptoTrailInitialRisk();

    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(_Symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;

    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(today != g_currentDay) {
        g_currentDay = today;
        g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
        g_lonLoaded=false;g_nyLoaded=false;g_asiaLoaded=false;
        g_lonRangeDefined = false; g_nyRangeDefined = false; g_asiaRangeDefined = false;
        g_lonExecutedToday = false; g_nyExecutedToday = false; g_asiaExecutedToday = false;
        g_lonHigh = 0.0; g_lonLow = 999999.0;
        g_nyHigh = 0.0; g_nyLow = 999999.0;
        g_asiaHigh = 0.0; g_asiaLow = 999999.0;
        g_nyReArmActive = false;
        ArrayResize(g_daily_swing_highs, 0); ArrayResize(g_daily_swing_lows, 0);
        CryptoWarmDailyBias();
        ArrayResize(g_countedLossIds, 0);
    }

    if(InpUseEquityDailyLock && g_dayStartEquity > 0) {
        double eq = AccountInfoDouble(ACCOUNT_EQUITY);
        double lossPct = (g_dayStartEquity - eq) / g_dayStartEquity * 100.0;
        if(lossPct >= InpMaxDailyLossPct) g_dayLocked = true;
    }

    if(!InpTradeWeekend && (dt.day_of_week==0 || dt.day_of_week==6)) return;
    LoadCryptoORB(today+InpLon_StartHour*3600+InpLon_StartMin*60,InpLon_DurationMin,g_lonLoaded,g_lonRangeDefined,g_lonHigh,g_lonLow);
    LoadCryptoORB(today+InpNY_StartHour*3600+InpNY_StartMin*60,InpNY_DurationMin,g_nyLoaded,g_nyRangeDefined,g_nyHigh,g_nyLow);
    if(InpUseAsia) LoadCryptoORB(today+InpAsia_StartHour*3600+InpAsia_StartMin*60,InpAsia_DurationMin,g_asiaLoaded,g_asiaRangeDefined,g_asiaHigh,g_asiaLow);
    ManagePyramids();
    if(g_dayLocked) { if(InpShowComments) Comment("STOP TRADING: Max Losses / Daily Loss Limit Reached"); return; }
    if(g_tradesToday >= InpMaxTradesDay) return;
    if(!CryptoCanOpenSession()) return;

    if(InpBandPriority){CryptoTryComplementaryBand();if(CryptoPositionCount()>0)return;}

    static datetime lastD1 = 0;
    if(iTime(_Symbol, PERIOD_D1, 0) != lastD1) {
        lastD1 = iTime(_Symbol, PERIOD_D1, 0);
        double ph = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, true);
        double pl = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, false);
        if(ph != EMPTY_VALUE) { AddToArray(g_daily_swing_highs, ph); if(ArraySize(g_daily_swing_highs)>10) ShiftArray(g_daily_swing_highs); }
        if(pl != EMPTY_VALUE) { AddToArray(g_daily_swing_lows, pl); if(ArraySize(g_daily_swing_lows)>10) ShiftArray(g_daily_swing_lows); }
        g_dailyBias = GetDailyBias();
    }

    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);

    // Gestione rientro REVERSAL dopo stop di una fade (solo NY; Londra non apre piu' fade)
    if(g_nyReArmActive) {
        if(TimeCurrent() > g_nyReArmDeadline) {
            g_nyReArmActive = false;
        } else if(!HasOpenPosition()) {
            double price = (g_nyFadeDir == -1) ? SymbolInfoDouble(_Symbol, SYMBOL_ASK) : SymbolInfoDouble(_Symbol, SYMBOL_BID);
            int revDir = -g_nyFadeDir;
            double sl = g_nyFadeSL;
            ExecuteTrade(revDir, price, sl, "NY_REVERSAL", GetRiskAmount(), 0);
            g_nyReArmActive = false;
        }
    }

    // --- LONDON SESSION LOGIC (v5.5: SENZA FADE) ---
    datetime lonStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpLon_StartHour, InpLon_StartMin));
    datetime lonEnd   = lonStart + (InpLon_DurationMin * 60);
    datetime lonScanLimit = lonEnd + (InpMaxEntryDelayMin * 60);

    if(!g_lonExecutedToday && TimeCurrent() >= lonStart) {
        if(TimeCurrent() < lonEnd) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_lonHigh) g_lonHigh = h;
            if(l < g_lonLow) g_lonLow = l;
            g_lonRangeDefined = true;
        } else if(TimeCurrent() < lonScanLimit && g_lonRangeDefined) {

            bool atrOkLon = true;
            if(InpUseATRFilter) {
                double atrVal = GetATRValue();
                double sessionRange = g_lonHigh - g_lonLow;
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMult) atrOkLon = false;
            }

            if(!atrOkLon) {
                g_lonExecutedToday = true;
            } else {
                double m5_close = iClose(_Symbol, InpEntryTF, 1);
                double m5_open  = iOpen(_Symbol, InpEntryTF, 1);
                double m5_high  = iHigh(_Symbol, InpEntryTF, 1);
                double m5_low   = iLow(_Symbol, InpEntryTF, 1);

                double bodySize = MathAbs(m5_close - m5_open);
                double totalSize = m5_high - m5_low;
                bool validBody = (totalSize > 0) && ((bodySize / totalSize) * 100.0 > InpBodyFilterPct);

                bool bull = (m5_close > g_lonHigh);
                bool bear = (m5_close < g_lonLow);

                if(!InpLon_IgnoreBias && InpLon_UseDailyFilter) {
                    if(bull && g_dailyBias != 1) bull = false;
                    if(bear && g_dailyBias != -1) bear = false;
                }

                if(bull || bear) {
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                        // Continuazione: unica logica attiva su Londra
                        if(bull) ExecuteTrade(1, m5_close, g_lonLow, "LONDON", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_lonHigh, "LONDON", riskAmt, 0);
                    }
                    // v5.5: RIMOSSO il ramo "else if(InpFadeEnabled)" -> Londra non apre mai
                    // un LONDON_FADE. Se il corpo e' debole, non si fa nulla e la scansione
                    // prosegue sulle candele successive (g_lonExecutedToday resta false,
                    // dato che sotto viene impostato solo se una posizione risulta aperta).
                    if(CryptoSessionOpen("LONDON")) g_lonExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= lonScanLimit) {
             g_lonExecutedToday = true;
        }
    }

    // --- NEW YORK SESSION LOGIC (invariata, fade attiva) ---
    datetime nyStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpNY_StartHour, InpNY_StartMin));
    datetime nyEnd   = nyStart + (InpNY_DurationMin * 60);
    datetime nyScanLimit = nyEnd + (InpMaxEntryDelayMin * 60);

    if(!g_nyExecutedToday && TimeCurrent() >= nyStart) {
        if(TimeCurrent() < nyEnd) {
            double h = iHigh(_Symbol, PERIOD_M1, 1);
            double l = iLow(_Symbol, PERIOD_M1, 1);
            if(h > g_nyHigh) g_nyHigh = h;
            if(l < g_nyLow) g_nyLow = l;
            g_nyRangeDefined = true;
        } else if(TimeCurrent() < nyScanLimit && g_nyRangeDefined) {

            bool atrOkNY = true;
            if(InpUseATRFilter) {
                double atrVal = GetATRValue();
                double sessionRange = g_nyHigh - g_nyLow;
                if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMultNY) atrOkNY = false;
            }

            if(!atrOkNY) {
                g_nyExecutedToday = true;
            } else {
                double m5_close = iClose(_Symbol, InpEntryTF, 1);
                double m5_open  = iOpen(_Symbol, InpEntryTF, 1);
                double m5_high  = iHigh(_Symbol, InpEntryTF, 1);
                double m5_low   = iLow(_Symbol, InpEntryTF, 1);

                double bodySize = MathAbs(m5_close - m5_open);
                double totalSize = m5_high - m5_low;
                bool validBody = (totalSize > 0) && ((bodySize / totalSize) * 100.0 > InpBodyFilterPct);

                bool bull = (m5_close > g_nyHigh);
                bool bear = (m5_close < g_nyLow);

                if(InpNY_UseDailyFilter) {
                    if(bull && g_dailyBias == -1) bull = false;
                    if(bear && g_dailyBias == 1) bear = false;
                }

                if(bull || bear) {
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                        if(bull) ExecuteTrade(1, m5_close, g_nyLow, "NY", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_nyHigh, "NY", riskAmt, 0);
                    } else if(InpFadeEnabled) {
                        double fadeRisk = riskAmt * (InpFadeRiskPct / 100.0);
                        double buffer = InpFadeSLBufferPoints * point;
                        if(bull) {
                            double sl = m5_high + buffer;
                            g_nyFadeDir = -1; g_nyFadeSL = sl;
                            ExecuteTrade(-1, m5_close, sl, "NY_FADE", fadeRisk, g_nyLow);
                        } else if(bear) {
                            double sl = m5_low - buffer;
                            g_nyFadeDir = 1; g_nyFadeSL = sl;
                            ExecuteTrade(1, m5_close, sl, "NY_FADE", fadeRisk, g_nyHigh);
                        }
                    }
                    if(CryptoSessionOpen("NY")) g_nyExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= nyScanLimit) {
             g_nyExecutedToday = true;
        }
    }

    // --- ASIAN SESSION LOGIC (invariata, fade attiva) ---
    if(InpUseAsia) {
        datetime asiaStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpAsia_StartHour, InpAsia_StartMin));
        datetime asiaEnd   = asiaStart + (InpAsia_DurationMin * 60);
        datetime asiaScanLimit = asiaEnd + (InpMaxEntryDelayMin * 60);

        if(!g_asiaExecutedToday && TimeCurrent() >= asiaStart) {
            if(TimeCurrent() < asiaEnd) {
                double h = iHigh(_Symbol, PERIOD_M1, 1);
                double l = iLow(_Symbol, PERIOD_M1, 1);
                if(h > g_asiaHigh) g_asiaHigh = h;
                if(l < g_asiaLow) g_asiaLow = l;
                g_asiaRangeDefined = true;
            } else if(TimeCurrent() < asiaScanLimit && g_asiaRangeDefined) {

                bool atrOkAsia = true;
                if(InpUseATRFilter) {
                    double atrVal = GetATRValue();
                    double sessionRange = g_asiaHigh - g_asiaLow;
                    if(atrVal <= 0 || sessionRange < atrVal * InpATRMinRangeMultAsia) atrOkAsia = false;
                }

                if(!atrOkAsia) {
                    g_asiaExecutedToday = true;
                } else {
                    double m5_close = iClose(_Symbol, InpEntryTF, 1);
                    double m5_open  = iOpen(_Symbol, InpEntryTF, 1);
                    double m5_high  = iHigh(_Symbol, InpEntryTF, 1);
                    double m5_low   = iLow(_Symbol, InpEntryTF, 1);

                    double bodySize = MathAbs(m5_close - m5_open);
                    double totalSize = m5_high - m5_low;
                    bool validBody = (totalSize > 0) && ((bodySize / totalSize) * 100.0 > InpBodyFilterPct);

                    bool bull = (m5_close > g_asiaHigh);
                    bool bear = (m5_close < g_asiaLow);

                    if(InpAsia_UseDailyFilter) {
                        if(bull && g_dailyBias == -1) bull = false;
                        if(bear && g_dailyBias == 1) bear = false;
                    }

                    if(bull || bear) {
                        double riskAmt = GetRiskAmount();
                        if(validBody) {
                            if(bull) ExecuteTrade(1, m5_close, g_asiaLow, "ASIA", riskAmt, 0);
                            else if(bear) ExecuteTrade(-1, m5_close, g_asiaHigh, "ASIA", riskAmt, 0);
                        } else if(InpFadeEnabled) {
                            double fadeRisk = riskAmt * (InpFadeRiskPct / 100.0);
                            double buffer = InpFadeSLBufferPoints * point;
                            if(bull) {
                                double sl = m5_high + buffer;
                                ExecuteTrade(-1, m5_close, sl, "ASIA_FADE", fadeRisk, g_asiaLow);
                            } else if(bear) {
                                double sl = m5_low - buffer;
                                ExecuteTrade(1, m5_close, sl, "ASIA_FADE", fadeRisk, g_asiaHigh);
                            }
                        }
                        if(CryptoSessionOpen("ASIA")) g_asiaExecutedToday = true;
                    }
                }
            } else if(TimeCurrent() >= asiaScanLimit) {
                g_asiaExecutedToday = true;
            }
        }
    }

    if(!InpBandPriority)CryptoTryComplementaryBand();

    if(InpShowComments) {
        Comment(StringFormat("Tri-Session v5.5 (Lon no-fade) | Bias: %s | Lon H/L: %.2f/%.2f | NY H/L: %.2f/%.2f | Asia H/L: %.2f/%.2f | Trades: %d",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")),
              g_lonHigh, g_lonLow, g_nyHigh, g_nyLow, g_asiaHigh, g_asiaLow, g_tradesToday));
    }
}

// ══════════════════════════════════════════
// 💼 EXECUTION
// ══════════════════════════════════════════
void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
 if(!PortfolioUpdate() || PortfolioNewsBlocked(_Symbol))return;
 if(InpNASOnlyNY && session!="NY")return;

    bool additional=CryptoPositionCount()>0;
    if(additional){
      if(!CryptoCanOpenSession() || !CryptoStackDirectionAllows(dir))return;
      riskAmount*=InpStackExtraRiskMultiplier;
    }

    if((session!="BAND_BREAK" || InpBandRequireRegime) && !CryptoRegimeAllows(dir))return;
    if((dir==1 && !InpCryptoAllowLong) || (dir==-1 && !InpCryptoAllowShort))return;
    if(CryptoPositionCount()>=InpCryptoMaxPositions || CryptoSessionOpen(session))return;
    riskAmount*=dir==1?InpCryptoLongRiskMult:InpCryptoShortRiskMult;
    MqlDateTime riskClock;TimeToStruct(TimeCurrent(),riskClock);
    if(riskClock.day_of_week==0 || riskClock.day_of_week==6)riskAmount*=InpCryptoWeekendRiskMult;

    if(StringFind(session,"ASIA")==0) riskAmount*=InpAsiaRiskMultiplier;
    else if(StringFind(session,"NY")==0) riskAmount*=InpNYRiskMultiplier;
    else if(session=="LONDON") riskAmount*=InpLondonRiskMultiplier;

    double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);

    // Rischio e target calcolati sul prezzo negoziabile corrente.
    MqlTick quote;
    if(!SymbolInfoTick(_Symbol,quote)) return;
    entry=dir==1?quote.ask:quote.bid;
    if(InpRegimeStopATR>0){
      if(g_regimeStopDistance<=0)return;
      sl_ref=entry-dir*g_regimeStopDistance;
    }
    double tickSize=SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
    if(tickSize<=0 || riskAmount<=0 || entry<=0) return;
    double sl_final=NormalizeDouble(MathRound(sl_ref/tickSize)*tickSize,digits);
    double minStop=SymbolInfoInteger(_Symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
    if((dir==1 && (sl_final>=quote.bid-minStop)) || (dir==-1 && sl_final<=quote.ask+minStop)) return;

    double stopDistance=MathAbs(entry-sl_final);
    if(InpCryptoMaxStopATR>0){double atr=GetATRValue();if(atr<=0 || stopDistance>atr*InpCryptoMaxStopATR)return;}
    double aggregateBudget=AccountInfoDouble(ACCOUNT_EQUITY)*InpTotalOpenStopRiskPct/100.0-CryptoOpenStopRisk();
    riskAmount=MathMin(riskAmount,aggregateBudget);if(riskAmount<=0)return;
    riskAmount=PortfolioEntryRisk(riskAmount,_Symbol);if(riskAmount<=0)return;


    if(InpMaxSpreadToStopPct>0 && quote.ask-quote.bid>stopDistance*InpMaxSpreadToStopPct/100.0) return;
    double tp_final = 0;
    if(customTP > 0) {
        tp_final = NormalizeDouble(customTP, digits);
    } else {
        double dist = MathAbs(entry - sl_final);
        if(dist == 0) return;
        tp_final = NormalizeDouble((dir == 1) ? entry + (dist * InpRR_Target) : entry - (dist * InpRR_Target), digits);
    }

    tp_final=NormalizeDouble(MathRound(tp_final/tickSize)*tickSize,digits);
    if(dir == 1 && tp_final <= entry) {
        Print("DEBUG ExecuteTrade: SCARTATO (", session, ") - TP ", DoubleToString(tp_final,digits),
              " <= entry ", DoubleToString(entry,digits));
        return;
    }
    if(dir == -1 && tp_final >= entry) {
        Print("DEBUG ExecuteTrade: SCARTATO (", session, ") - TP ", DoubleToString(tp_final,digits),
              " >= entry ", DoubleToString(entry,digits));
        return;
    }

    double minLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(_Symbol, SYMBOL_VOLUME_STEP);
    double profitPerLot = 0;

    bool calcRes = false;
    if(dir == 1) calcRes = OrderCalcProfit(ORDER_TYPE_BUY, _Symbol, 1.0, entry, sl_final, profitPerLot);
    else calcRes = OrderCalcProfit(ORDER_TYPE_SELL, _Symbol, 1.0, entry, sl_final, profitPerLot);

    if(!calcRes || profitPerLot >= 0) return;

    if(stepLot<=0) return;
    double lots = MathFloor((riskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
    // Nessun arrotondamento verso l'alto che superi il rischio scelto.
    if(lots < minLot || stepLot<=0) return;
    lots=MathMin(lots,MathFloor(SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX)/stepLot)*stepLot);
    double margin=0;
    if(!OrderCalcMargin(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,_Symbol,lots,entry,margin) || margin<=0) return;
    double availableMargin=AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.80;
    if(margin>availableMargin) lots=MathFloor(lots*availableMargin/margin/stepLot)*stepLot;
    if(lots<minLot) return;

    string cmt = StringFormat("%s|v5.5|R:%.2f", session, riskAmount);
    bool res = false;

    if(dir == 1) res = trade.Buy(lots, _Symbol, 0.0, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, _Symbol, 0.0, sl_final, tp_final, cmt);

    if(res && (trade.ResultRetcode()==TRADE_RETCODE_DONE || trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL)) {
        g_tradesToday++;
        if(additional)Print("CRYPTO_STACK_OPENED ticket ",trade.ResultOrder()," session ",session," nominalBudget ",riskAmount);


        long posId=0; double filledEntry=entry;
        ulong filledDeal=trade.ResultDeal();
        if(filledDeal>0 && HistoryDealSelect(filledDeal)) {
            posId=HistoryDealGetInteger(filledDeal,DEAL_POSITION_ID);
            filledEntry=HistoryDealGetDouble(filledDeal,DEAL_PRICE);
        }
        if(posId > 0) RegisterPosition(posId, session, filledEntry, MathAbs(filledEntry - sl_final), dir);

        if(session=="BAND_BREAK")g_bandTrades++;
        if(session == "LONDON") g_totalLon++;
        else if(session == "NY") g_totalNY++;
        else if(session == "ASIA") g_totalAsia++;
        else if(session == "NY_FADE") g_totalNYFade++;
        else if(session == "ASIA_FADE") g_totalAsiaFade++;
        else if(session == "NY_REVERSAL") g_totalNYRev++;
        else if(session == "PYRAMID") g_totalPyramid++;

        Print("TRADE OPENED: ", session, " | PosID: ", posId, " | Risk: ", DoubleToString(riskAmount,2), " | TP: ", DoubleToString(tp_final, digits));
    }
}

// ══════════════════════════════════════════
// 📊 STATS TRACKING
// ══════════════════════════════════════════
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != _Symbol) return;

    long posId = HistoryDealGetInteger(trans.deal, DEAL_POSITION_ID);
    int dealEntry = (int)HistoryDealGetInteger(trans.deal, DEAL_ENTRY);

    if(dealEntry == DEAL_ENTRY_OUT) {
        string session = GetSessionByPosId(posId);
        if(session == "UNKNOWN") return;

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
        for(int p=PositionsTotal()-1; p>=0; p--) if(PositionGetTicket(p)>0 && PositionGetInteger(POSITION_IDENTIFIER) == posId) stillOpen = true;
        if(stillOpen) return;

        if(!IsIdAlreadyCounted(posId)) {
            if(session=="BAND_BREAK"){g_bandProfit+=totalPnL;if(totalPnL>0)g_bandWins++;}
            if(session == "LONDON") { g_profitLon += totalPnL; if(totalPnL > 0) g_winsLon++; }
            else if(session == "NY") { g_profitNY += totalPnL; if(totalPnL > 0) g_winsNY++; }
            else if(session == "ASIA") { g_profitAsia += totalPnL; if(totalPnL > 0) g_winsAsia++; }
            else if(session == "NY_FADE") { g_profitNYFade += totalPnL; if(totalPnL > 0) g_winsNYFade++; }
            else if(session == "ASIA_FADE") { g_profitAsiaFade += totalPnL; if(totalPnL > 0) g_winsAsiaFade++; }
            else if(session == "NY_REVERSAL") { g_profitNYRev += totalPnL; if(totalPnL > 0) g_winsNYRev++; }

            if(session=="PYRAMID") {g_profitPyramid+=totalPnL;if(totalPnL>0) g_winsPyramid++;}
            AddToLongArray(g_countedLossIds, posId);
            if(totalPnL < 0) {
                g_lossesToday++;
                if(g_lossesToday >= InpMaxLossesDay) g_dayLocked = true;

                if(session == "NY_FADE") {
                    g_nyReArmActive = true;
                    g_nyReArmDeadline = TimeCurrent() + 30*60;
                }
            }
            UnregisterPosition(posId);
        }
    }
}
//+------------------------------------------------------------------+
// Esporta una riga per posizione, unendo tutti i parziali e includendo costi e swap.
double OnTester() {
    if(!HistorySelect(0,TimeCurrent())) return TesterStatistics(STAT_PROFIT);
    long ids[];
    for(int i=0;i<HistoryDealsTotal();i++) {
        ulong deal=HistoryDealGetTicket(i);
        if(HistoryDealGetString(deal,DEAL_SYMBOL)!=_Symbol || HistoryDealGetInteger(deal,DEAL_MAGIC)!=(long)InpMagicNumber) continue;
        long id=HistoryDealGetInteger(deal,DEAL_POSITION_ID);
        bool found=false;
        for(int j=0;j<ArraySize(ids);j++) if(ids[j]==id) {found=true;break;}
        if(id>0 && !found) {int n=ArraySize(ids);ArrayResize(ids,n+1);ids[n]=id;}
    }
    string filename=MQLInfoString(MQL_PROGRAM_NAME)+"_trades.csv";
    int handle=FileOpen(filename,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);
    if(handle==INVALID_HANDLE) {Print("CSV non creato: ",GetLastError());return TesterStatistics(STAT_PROFIT);}
    FileWrite(handle,"position_id","session","open_time","close_time","initial_volume","entry_price","initial_sl","initial_tp","net_profit");
    for(int i=0;i<ArraySize(ids);i++) {
        if(!HistorySelectByPosition(ids[i])) continue;
        string session="";datetime opened=0,closed=0;double volume=0,entry=0,sl=0,tp=0,net=0;
        for(int j=0;j<HistoryDealsTotal();j++) {
            ulong deal=HistoryDealGetTicket(j);
            net+=HistoryDealGetDouble(deal,DEAL_PROFIT)+HistoryDealGetDouble(deal,DEAL_COMMISSION)+HistoryDealGetDouble(deal,DEAL_SWAP)+HistoryDealGetDouble(deal,DEAL_FEE);
            if(HistoryDealGetInteger(deal,DEAL_ENTRY)==DEAL_ENTRY_IN && opened==0) {
                opened=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);
                session=HistoryDealGetString(deal,DEAL_COMMENT);int sep=StringFind(session,"|");if(sep>=0) session=StringSubstr(session,0,sep);
                volume=HistoryDealGetDouble(deal,DEAL_VOLUME);entry=HistoryDealGetDouble(deal,DEAL_PRICE);
                sl=HistoryDealGetDouble(deal,DEAL_SL);tp=HistoryDealGetDouble(deal,DEAL_TP);
            }
            if(HistoryDealGetInteger(deal,DEAL_ENTRY)==DEAL_ENTRY_OUT) closed=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);
        }
        FileWrite(handle,ids[i],session,TimeToString(opened,TIME_DATE|TIME_SECONDS),TimeToString(closed,TIME_DATE|TIME_SECONDS),DoubleToString(volume,4),DoubleToString(entry,5),DoubleToString(sl,5),DoubleToString(tp,5),DoubleToString(net,2));
    }
    FileClose(handle);
    Print("CSV creato: ",TerminalInfoString(TERMINAL_COMMONDATA_PATH),"\\Files\\",filename);
    return TesterStatistics(STAT_PROFIT);
}

// Ricerca FTMO Standard: calendario UTC incorporato solo nel tester.
input bool InpPortfolioNewsEnabled=true;
input int InpPortfolioNewsCloseAheadMin=15;
input int InpPortfolioNewsResumeAfterMin=3;
long g_newsHistorical[]={1736362800,1736515800,1736947800,1738177200,1738243800,1738935000,1739367000,1739991600,1740663000,1741354200,1741782600,1742407200,1743078600,1743769800,1744221600,1744288200,1746016200,1746189000,1746640800,1747139400,1748455200,1748521800,1749213000,1749645000,1750269600,1750941000,1751545800,1752084000,1752582600,1753878600,1753898400,1754051400,1755001800,1755712800,1756384200,1757075400,1757593800,1758132000,1758803400,1759946400,1761309000,1761760800,1763578800,1763641800,1765393200,1765891800,1766064600,1766496600,1767207600,1767965400,1768311000,1769088600,1769626800,1770816600,1770989400,1771441200,1771594200,1772803800,1773232200,1773405000,1773856800,1775219400,1775671200,1775737800,1775824200,1777485600,1777552200,1778243400,1778589000,1779300000,1779971400,1780662600,1781094600,1781719200,1782390600,1782995400,1783533600,1784032200,1785348000,1785414600,1786105800,1786537800,1787162400,1787747400,1788525000,1789129800,1789581600,1790771400,1790944200,1791396000,1791981000,1793210400,1793277000};
long g_newsLive[];
datetime g_newsLoaded=0;
bool g_newsReady=false,g_newsOpenInRestrictedWindow=false;
bool PortfolioRestrictedEvent(string code){
 return code=="fed-interest-rate-decision" || code=="fomc-meeting-statement" || code=="fomc-minutes" ||
 code=="nonfarm-payrolls" || code=="unemployment-rate" || code=="average-hourly-earnings-mm" ||
 code=="average-hourly-earnings-yy" || code=="consumer-price-index-yy" || code=="gross-domestic-product-qq";
}
bool PortfolioNewsLoadLive(){
 datetime now=TimeCurrent();
 if(g_newsReady && now>=g_newsLoaded && now-g_newsLoaded<3600)return true;
 g_newsReady=false;
 int offset=(int)MathRound((double)(TimeTradeServer()-TimeGMT())/3600.0);
 if(offset!=PortfolioServerOffset(now)){Print("NEWS_CLOCK_MISMATCH: XAU/NAS bloccati");return false;}
 MqlCalendarValue values[];ResetLastError();
 int count=CalendarValueHistory(values,now-86400,now+7*86400,"US","USD");
 if(count<=0 || GetLastError()!=0){Print("NEWS_CALENDAR_UNAVAILABLE: XAU/NAS bloccati");return false;}
 ArrayResize(g_newsLive,0);
 for(int i=0;i<count;i++){
  MqlCalendarEvent event;
  if(!CalendarEventById(values[i].event_id,event))return false;
  if(!PortfolioRestrictedEvent(event.event_code))continue;
  int n=ArraySize(g_newsLive);ArrayResize(g_newsLive,n+1);g_newsLive[n]=(long)values[i].time-offset*3600;
 }
 ArraySort(g_newsLive);g_newsLoaded=now;g_newsReady=true;return true;
}
bool PortfolioNewsInit(){
 return InpPortfolioNewsEnabled && InpPortfolioNewsCloseAheadMin>=5 && InpPortfolioNewsResumeAfterMin>=3;
}
long PortfolioNearestNews(long utc,long &events[]){
 int left=0,right=ArraySize(events);
 while(left<right){int mid=(left+right)/2;if(events[mid]<utc-InpPortfolioNewsResumeAfterMin*60)left=mid+1;else right=mid;}
 return left<ArraySize(events)?events[left]:0;
}
bool PortfolioNewsBlocked(string symbol){
 if(symbol==InpBTCsymbol)return false; // BTC non e' nella tabella USD FTMO verificata.
 if(!InpPortfolioNewsEnabled)return true;
 datetime now=TimeCurrent();long utc=(long)now-PortfolioServerOffset(now)*3600;
 if(MQLInfoInteger(MQL_TESTER)){
  if(utc<1735689600 || utc>=1793491200)return true;
 }else if(!PortfolioNewsLoadLive())return true;
 long event=MQLInfoInteger(MQL_TESTER)?PortfolioNearestNews(utc,g_newsHistorical):PortfolioNearestNews(utc,g_newsLive);
 bool blocked=event>0 && utc>=event-InpPortfolioNewsCloseAheadMin*60 && utc<=event+InpPortfolioNewsResumeAfterMin*60;
 if(!blocked)return false;
 for(int i=PositionsTotal()-1;i>=0;i--){
  ulong ticket=PositionGetTicket(i);
  if(ticket==0 || !PortfolioOwn(PositionGetInteger(POSITION_MAGIC)) || PositionGetString(POSITION_SYMBOL)!=symbol)continue;
  // Non inviare chiusure dentro +/-2 minuti: una chiusura sarebbe anch'essa soggetta alla restrizione.
  if(utc>=event-120 && utc<=event+120){
   if(!g_newsOpenInRestrictedWindow)Print("NEWS_POSITION_STILL_OPEN_IN_RESTRICTED_WINDOW: verifica necessaria ",symbol);
   g_newsOpenInRestrictedWindow=true;continue;
  }
  g_portfolioTrade.SetTypeFillingBySymbol(symbol);
  bool sent=g_portfolioTrade.PositionClose(ticket);
  if(!sent || (g_portfolioTrade.ResultRetcode()!=TRADE_RETCODE_DONE && g_portfolioTrade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))PrintFormat("NEWS_PRE_CLOSE_FAILED %I64u ret=%u",ticket,g_portfolioTrade.ResultRetcode());
 }
 return true;
}
