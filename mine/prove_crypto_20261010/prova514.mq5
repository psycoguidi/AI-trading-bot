// prova514: NAS100 ORB XAU183, NY09:30ET corretta, cap50EUR; solo NY, solo long, 2R. Screening 2025, non attivato
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
#property strict
#include <Trade\Trade.mqh>
input ulong InpMagicNumber=20261010514;
input bool InpShowComments=false;
input string InpBTCsymbol="BTCUSD";
input string InpNASsymbol="NAS100";
input bool InpStandardNewsValidated=false;
input bool InpNASOnlyNY=true;
// Protezione comune FTMO 2-Step. Bozza di ricerca: ordini fuori tester disabilitati.
// Le soglie riguardano l'intero conto, incluse altre posizioni e commissioni.
enum ENUM_PORTFOLIO_DST { PORTFOLIO_DST_NONE=0, PORTFOLIO_DST_EU=1, PORTFOLIO_DST_US=2 };
input group "FTMO 2-Step: controllo comune del conto"
input double InpPortfolioInitialCapital=10000.0;
input double InpPortfolioDailyLoss=500.0;
input double InpPortfolioTotalLoss=1000.0;
input double InpPortfolioSafetyBuffer=100.0;
input double InpPortfolioCostReserve=8.0;
input double InpPortfolioRiskCap=50.0;
input string InpPortfolioOwnedMagics="20261013514";
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
enum ENUM_FTMO_SERVER_DST { FTMO_DST_EU=0, FTMO_DST_US=1, FTMO_DST_NONE=2 };
input int NAS_InpLon_StartHour=10;
input int NAS_InpLon_StartMin=0;
input int NAS_InpLon_DurationMin=60;
input bool NAS_InpLon_UseDailyFilter=true;
input bool NAS_InpLon_IgnoreBias=true;
input int NAS_InpNY_StartHour=16;
input int NAS_InpNY_StartMin=30;
input int NAS_InpNY_DurationMin=30;
input bool NAS_InpNY_UseDailyFilter=true;
input bool NAS_InpUseAsia=false;
input int NAS_InpAsia_StartHour=3;
input int NAS_InpAsia_StartMin=0;
input int NAS_InpAsia_DurationMin=60;
input bool NAS_InpAsia_UseDailyFilter=true;
input ENUM_TIMEFRAMES NAS_InpEntryTF=PERIOD_M5;
input int NAS_InpMaxEntryDelayMin=360;
input double NAS_InpBodyFilterPct=25.0;
input bool NAS_InpFadeEnabled=false;
input double NAS_InpFadeRiskPct=50.0;
input double NAS_InpFadeSLBufferPoints=30;
input double NAS_InpATRMinRangeMultAsia=0.1;
input double NAS_InpATRMinRangeMultNY=0.15;
input bool NAS_InpUseATRFilter=true;
input int NAS_InpATRPeriod=14;
input double NAS_InpATRMinRangeMult=0.15;
input bool NAS_InpUseBreakEven=true;
input double NAS_InpBreakEvenTriggerR=1.0;
input double NAS_InpBreakEvenLockR=0.1;
input bool NAS_InpUsePartialClose=false;
input double NAS_InpPartialClosePct=50.0;
input double NAS_InpPartialCloseAtR=1.0;
input double NAS_InpFixedRiskAmount=80.0;
input bool NAS_InpUseEquityRisk=true;
input double NAS_InpRiskPercent=2.0;
input double NAS_InpRR_Target=2;
input int NAS_InpMaxLossesDay=4;
input int NAS_InpMaxTradesDay=12;
input ulong NAS_InpMagicNumber=20261013514;
input bool NAS_InpShowComments=false;
input bool NAS_InpUseEquityDailyLock=false;
input double NAS_InpMaxDailyLossPct=3.0;
input double NAS_InpPyramidAtR=1.2;
input double NAS_InpPyramidRiskPercent=0.4;
input double NAS_InpPyramidExtraTargetR=1.0;
input double NAS_InpTotalOpenStopRiskPct=4.0;
input double NAS_InpAsiaRiskMultiplier=1.05;
input double NAS_InpNYRiskMultiplier=0.8;
input double NAS_InpLondonRiskMultiplier=1.1;
input double NAS_InpFTMOInitialCapital=10000.0;
input double NAS_InpFTMOTargetPercent=10.0;
input bool NAS_InpFTMOStopAtTarget=false;
input int NAS_InpFTMOMinTradingDays=4;
input double NAS_InpFTMODailyLossEUR=500.0;
input double NAS_InpFTMOTotalLossEUR=1000.0;
input double NAS_InpFTMOBufferEUR=150.0;
input double NAS_InpFTMOReservePerTradeEUR=5.0;
input double NAS_InpFTMOQualificationRiskPct=1.0;
input int NAS_InpFTMOServerUTCWinter=2;
input int NAS_InpFTMOServerUTCSummer=3;
input ENUM_FTMO_SERVER_DST NAS_InpFTMOServerDST=FTMO_DST_US;
input bool NAS_InpFTMOBrokerClockConfirmed=false;
input int NAS_InpResearchMonth=0;
input long NAS_InpResearchStopServer=0;
input int NAS_InpMaximumPositions=3;
input int NAS_InpSessionMaxEntries=2;
input int NAS_InpReentryCooldownMin=20;
input bool NAS_InpAllowLong=true;
input bool NAS_InpAllowShort=false;
input bool NAS_InpCloseFriday=true;
input int NAS_InpRolloverCheckServerHour=22;
input int NAS_InpRolloverCheckServerMinute=45;
input int NAS_InpDailyDirectionPeriod=20;
input bool NAS_InpDailyFilterLong=false;
input bool NAS_InpDailyFilterShort=true;
input double NAS_InpBelowInitialRiskPercent=1.25;
input bool NAS_InpOvernightOnlyProtected=true;
input bool NAS_InpDailyFlat=false;class NASEngine{
public:
 string m_symbol;
int InpLon_StartHour;
int InpLon_StartMin;
int InpLon_DurationMin;
bool InpLon_UseDailyFilter;
bool InpLon_IgnoreBias;
int InpNY_StartHour;
int InpNY_StartMin;
int InpNY_DurationMin;
bool InpNY_UseDailyFilter;
bool InpUseAsia;
int InpAsia_StartHour;
int InpAsia_StartMin;
int InpAsia_DurationMin;
bool InpAsia_UseDailyFilter;
ENUM_TIMEFRAMES InpEntryTF;
int InpMaxEntryDelayMin;
double InpBodyFilterPct;
bool InpFadeEnabled;
double InpFadeRiskPct;
double InpFadeSLBufferPoints;
double InpATRMinRangeMultAsia;
double InpATRMinRangeMultNY;
bool InpUseATRFilter;
int InpATRPeriod;
double InpATRMinRangeMult;
bool InpUseBreakEven;
double InpBreakEvenTriggerR;
double InpBreakEvenLockR;
bool InpUsePartialClose;
double InpPartialClosePct;
double InpPartialCloseAtR;
double InpFixedRiskAmount;
bool InpUseEquityRisk;
double InpRiskPercent;
double InpRR_Target;
int InpMaxLossesDay;
int InpMaxTradesDay;
ulong InpMagicNumber;
bool InpShowComments;
bool InpUseEquityDailyLock;
double InpMaxDailyLossPct;
double InpPyramidAtR;
double InpPyramidRiskPercent;
double InpPyramidExtraTargetR;
double InpTotalOpenStopRiskPct;
double g_profitPyramid;
int g_winsPyramid,g_totalPyramid;
double InpAsiaRiskMultiplier;
double InpNYRiskMultiplier;
double InpLondonRiskMultiplier;
double InpFTMOInitialCapital;
double InpFTMOTargetPercent;
bool InpFTMOStopAtTarget;
int InpFTMOMinTradingDays;
double InpFTMODailyLossEUR;
double InpFTMOTotalLossEUR;
double InpFTMOBufferEUR;
double InpFTMOReservePerTradeEUR;
double InpFTMOQualificationRiskPct;
int InpFTMOServerUTCWinter;
int InpFTMOServerUTCSummer;
ENUM_FTMO_SERVER_DST InpFTMOServerDST;
bool InpFTMOBrokerClockConfirmed;
int InpResearchMonth;
long InpResearchStopServer;
bool g_ftmoReady,g_ftmoFailed,g_ftmoPassed,g_ftmoTotalStopped;
int g_ftmoDay,g_ftmoLockedDay,g_ftmoDays[],g_ftmoMonth;
double g_ftmoMidnight,g_ftmoMinEquity,g_ftmoMinBalance,g_ftmoWorstDaily;
datetime g_ftmoStartCET,g_ftmoEndCET,g_ftmoPassedServer,g_ftmoFirstEntry;
string g_ftmoFailureReason;
int InpMaximumPositions;
int InpSessionMaxEntries;
int InpReentryCooldownMin;
int g_lonEntries,g_nyEntries,g_asiaEntries;
datetime g_lonLastEntry,g_nyLastEntry,g_asiaLastEntry;
int g_auditKeys[],g_auditIndex;
double g_auditMidnight[],g_auditMinimum[];
bool InpAllowLong;
bool InpAllowShort;
bool InpCloseFriday;
int InpRolloverCheckServerHour;
int InpRolloverCheckServerMinute;
int InpDailyDirectionPeriod;
bool InpDailyFilterLong;
bool InpDailyFilterShort;
int g_dailyDirectionMA;
double InpBelowInitialRiskPercent;
bool InpOvernightOnlyProtected;
bool InpDailyFlat;
datetime g_ftmoElapsedStartCET;
CTrade trade;
bool g_lonLoaded,g_nyLoaded,g_asiaLoaded;
int g_atrHandle;
double g_lonHigh,g_lonLow;
bool g_lonRangeDefined;
bool g_lonExecutedToday;
double g_nyHigh,g_nyLow;
bool g_nyRangeDefined;
bool g_nyExecutedToday;
double g_asiaHigh,g_asiaLow;
bool g_asiaRangeDefined;
bool g_asiaExecutedToday;
int g_dailyBias;
double g_daily_swing_highs[];
double g_daily_swing_lows[];
int InpSwingLen_Daily;
datetime g_currentDay;
double g_dayStartEquity;
int g_tradesToday;
int g_lossesToday;
bool g_dayLocked;
long g_countedLossIds[];


struct SessionMap { long posId; string session; double entryPrice; double slDistance; int dir; bool beDone; bool partialDone; bool pyramidDone; };
SessionMap g_posSessionMap[];
double g_profitLon,g_profitNY,g_profitAsia;
int g_winsLon,g_winsNY,g_winsAsia;
int g_totalLon,g_totalNY,g_totalAsia;
double g_profitNYFade,g_profitAsiaFade;
int g_winsNYFade,g_winsAsiaFade;
int g_totalNYFade,g_totalAsiaFade;
double g_profitNYRev;
int g_winsNYRev;
int g_totalNYRev;
int g_nyFadeDir;
double g_nyFadeSL;
bool g_nyReArmActive;
datetime g_nyReArmDeadline;


 
 
 
int OnInit() {
 if(InpDailyDirectionPeriod<=0)return INIT_PARAMETERS_INCORRECT;
 g_dailyDirectionMA=iMA(m_symbol,PERIOD_D1,InpDailyDirectionPeriod,0,MODE_EMA,PRICE_CLOSE);
 if(g_dailyDirectionMA==INVALID_HANDLE)return INIT_FAILED;

    if(InpFTMOInitialCapital<=0 || (InpFTMOStopAtTarget && InpFTMOTargetPercent<=0) || InpFTMOMinTradingDays<4 || InpFTMOBufferEUR<0 || InpFTMODailyLossEUR!=500 || InpFTMOTotalLossEUR!=1000)return INIT_PARAMETERS_INCORRECT;
    if(!MQLInfoInteger(MQL_TESTER) && !InpFTMOBrokerClockConfirmed){Print("Verificare orario/DST del broker FTMO e impostare InpFTMOBrokerClockConfirmed. Nessun ordine inviato.");return INIT_PARAMETERS_INCORRECT;}

    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFilling(ORDER_FILLING_IOC);

    ArrayResize(g_daily_swing_highs, 0);
    ArrayResize(g_daily_swing_lows, 0);
    ArrayResize(g_countedLossIds, 0);
    ArrayResize(g_posSessionMap, 0);

    g_atrHandle = iATR(m_symbol, PERIOD_D1, InpATRPeriod);
    if(g_atrHandle == INVALID_HANDLE) Print("ATTENZIONE: impossibile creare handle ATR, filtro volatilità disattivato di fatto.");

    MqlDateTime dtInit;
    TimeCurrent(dtInit);
    g_currentDay = StringToTime(StringFormat("%04d.%02d.%02d", dtInit.year, dtInit.mon, dtInit.day));
    g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);

    RestoreDailyStats();
 if(!RestorePortfolioPositions())return INIT_FAILED;
    Print("Tri-Session Master v5.5 (Londra senza fade) Ready.");
    return(INIT_SUCCEEDED);
}


void OnDeinit(const int reason) {
 if(g_dailyDirectionMA!=INVALID_HANDLE)IndicatorRelease(g_dailyDirectionMA);
    PrintFinalReport();
    if(g_atrHandle != INVALID_HANDLE) IndicatorRelease(g_atrHandle);
    Comment("");
}


 
 
 
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
        if(ticket > 0 && PositionGetString(POSITION_SYMBOL) == m_symbol && PositionGetInteger(POSITION_MAGIC) == (long)InpMagicNumber) return true;
    }
    return false;
}


double GetPivot(ENUM_TIMEFRAMES tf, int len, int shift, bool isHigh) {
    double val = isHigh ? iHigh(m_symbol, tf, shift) : iLow(m_symbol, tf, shift);
    for(int i=1; i<=len; i++) {
        double c1 = isHigh ? iHigh(m_symbol, tf, shift+i) : iLow(m_symbol, tf, shift+i);
        double c2 = isHigh ? iHigh(m_symbol, tf, shift-i) : iLow(m_symbol, tf, shift-i);
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
    double close = iClose(m_symbol, PERIOD_D1, 1);
    if(close > lastHH || (lastHH > prevHH && lastLL > prevLL)) return 1;
    if(close < lastLL || (lastHH < prevHH && lastLL < prevLL)) return -1;
    return 0;
}


void RestoreDailyStats() {
    datetime todayStart = StringToTime(TimeToString(TimeCurrent(), TIME_DATE));
    g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
g_lonEntries=0;g_nyEntries=0;g_asiaEntries=0;g_lonLastEntry=0;g_nyLastEntry=0;g_asiaLastEntry=0;
    ArrayResize(g_countedLossIds, 0);
    if(!HistorySelect(todayStart, TimeCurrent())) return;
    int totalDeals = HistoryDealsTotal();
    for(int i=0; i<totalDeals; i++) {
        ulong ticket = HistoryDealGetTicket(i);
        if(HistoryDealGetInteger(ticket, DEAL_MAGIC) != (long)InpMagicNumber) continue;
        if(HistoryDealGetString(ticket, DEAL_SYMBOL) != m_symbol) continue;
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


 
 
 
void ManageOpenPositions() {
    int digits = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);

    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        long posId = (long)PositionGetInteger(POSITION_IDENTIFIER);
        int idx = FindSessionMapIndex(posId);
        if(idx < 0) continue;

        double entry  = g_posSessionMap[idx].entryPrice;
        double slDist = g_posSessionMap[idx].slDistance;
        int    dir    = g_posSessionMap[idx].dir;
        if(slDist <= 0) continue;

        double currentPrice = (dir == 1) ? SymbolInfoDouble(m_symbol, SYMBOL_BID) : SymbolInfoDouble(m_symbol, SYMBOL_ASK);
        double rMultiple = (dir == 1) ? (currentPrice - entry) / slDist : (entry - currentPrice) / slDist;

        double volume    = PositionGetDouble(POSITION_VOLUME);
        double currentSL = PositionGetDouble(POSITION_SL);
        double currentTP = PositionGetDouble(POSITION_TP);

         
        if(InpUsePartialClose && !g_posSessionMap[idx].partialDone && rMultiple >= InpPartialCloseAtR) {
            double volMin  = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
            double volStep = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
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

         
        if(InpUseBreakEven && !g_posSessionMap[idx].beDone && rMultiple >= InpBreakEvenTriggerR) {
            double beSL = (dir == 1) ? entry + (InpBreakEvenLockR * slDist) : entry - (InpBreakEvenLockR * slDist);
            beSL = NormalizeDouble(beSL, digits);
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


 
 
 
 
void ManagePyramids() {
    static datetime checkedBar=0;
    datetime bar=iTime(m_symbol,PERIOD_M1,0);
    if(bar==checkedBar) return;
    checkedBar=bar;
    if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING || g_dayLocked || g_tradesToday>=InpMaxTradesDay) return;
    MqlTick tick; if(!SymbolInfoTick(m_symbol,tick)) return;
    int ownCount=0; double existingStopRisk=0;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=m_symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        ownCount++;
        int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
        double stop=PositionGetDouble(POSITION_SL),price=dir==1?tick.bid:tick.ask,pnl=0;
        if(stop<=0) return;
        if(!OrderCalcProfit(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,m_symbol,PositionGetDouble(POSITION_VOLUME),price,stop,pnl)) return;
        existingStopRisk+=MathMax(0,-pnl);
    }
    if(ownCount!=1) return;
    double equity=AccountInfoDouble(ACCOUNT_EQUITY);
    double budget=MathMin(equity*InpPyramidRiskPercent/100.0,equity*InpTotalOpenStopRiskPct/100.0-existingStopRisk);
    if(budget<=0) return;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=m_symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
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


int FTMOLastSunday(int year,int month) {
   MqlDateTime d={};d.year=year;d.mon=month+1;d.day=1;
   if(month==12){d.year++;d.mon=1;}
   datetime t=StructToTime(d)-86400;TimeToStruct(t,d);return d.day-d.day_of_week;
}

datetime FTMOEUChange(int year,int month) {
   MqlDateTime d={};d.year=year;d.mon=month;d.day=FTMOLastSunday(year,month);d.hour=1;return StructToTime(d);
}

datetime FTMOUSChange(int year,int month) {
   MqlDateTime d={};d.year=year;d.mon=month;d.day=1;TimeToStruct(StructToTime(d),d);
   d.day=1+(7-d.day_of_week)%7+(month==3?7:0);d.hour=month==3?7:6;return StructToTime(d);
}

int FTMOServerOffset(datetime server) {
   if(InpFTMOServerDST==FTMO_DST_NONE)return InpFTMOServerUTCWinter;
   MqlDateTime d;TimeToStruct(server,d);
   datetime a=InpFTMOServerDST==FTMO_DST_US?FTMOUSChange(d.year,3):FTMOEUChange(d.year,3);
   datetime b=InpFTMOServerDST==FTMO_DST_US?FTMOUSChange(d.year,11):FTMOEUChange(d.year,10);
   a+=InpFTMOServerUTCWinter*3600;b+=InpFTMOServerUTCSummer*3600;
   return server>=a && server<b?InpFTMOServerUTCSummer:InpFTMOServerUTCWinter;
}

datetime FTMOClock(datetime server) {
   static datetime hourStart=0;static int correction=0;
   datetime currentHour=(datetime)(((long)server/3600)*3600);
   if(currentHour!=hourStart){
      hourStart=currentHour;datetime utc=server-FTMOServerOffset(server)*3600;MqlDateTime d;TimeToStruct(utc,d);
      int offset=utc>=FTMOEUChange(d.year,3)&&utc<FTMOEUChange(d.year,10)?2:1;correction=offset*3600-FTMOServerOffset(server)*3600;
   }
   return server+correction;
}

int FTMODayKey(datetime server){
   static datetime dayStart=0;static int key=0;datetime local=FTMOClock(server);
   if(local<dayStart || local>=dayStart+86400){MqlDateTime d;TimeToStruct(local,d);key=d.year*10000+d.mon*100+d.day;d.hour=0;d.min=0;d.sec=0;dayStart=StructToTime(d);}
   return key;
}

datetime FTMODayStart(datetime cet){MqlDateTime d;TimeToStruct(cet,d);d.hour=0;d.min=0;d.sec=0;return StructToTime(d);}

string FTMOStateKey(string suffix){return StringFormat("F10_%I64d_%I64u_%s",AccountInfoInteger(ACCOUNT_LOGIN),InpMagicNumber,suffix);}

void FTMOStoreFailure(){if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("FAILED"),1);}

double FTMODealDelta(ulong ticket){return HistoryDealGetDouble(ticket,DEAL_PROFIT)+HistoryDealGetDouble(ticket,DEAL_SWAP)+HistoryDealGetDouble(ticket,DEAL_COMMISSION)+HistoryDealGetDouble(ticket,DEAL_FEE);}

bool FTMOInEvaluation(datetime server) {
   datetime cet=FTMOClock(server);
   if(cet<g_ftmoStartCET || (g_ftmoEndCET>0 && cet>=g_ftmoEndCET))return false;
   if(MQLInfoInteger(MQL_TESTER) && InpResearchStopServer>0 && (long)server>=InpResearchStopServer)return false;
   return true;
}

void FTMOAddTradingDay(int key,datetime entered) {
   if(g_ftmoFirstEntry==0 || entered<g_ftmoFirstEntry)g_ftmoFirstEntry=entered;
   for(int i=0;i<ArraySize(g_ftmoDays);i++)if(g_ftmoDays[i]==key)return;
   int n=ArraySize(g_ftmoDays);ArrayResize(g_ftmoDays,n+1);g_ftmoDays[n]=key;
}

bool FTMORebuildDay() {
   datetime now=TimeCurrent();if(!HistorySelect(0,now))return false;
   int today=FTMODayKey(now);double delta=0,allDelta=0;
   for(int i=0;i<HistoryDealsTotal();i++) {
      ulong deal=HistoryDealGetTicket(i);datetime t=(datetime)HistoryDealGetInteger(deal,DEAL_TIME);
      if(!FTMOInEvaluation(t))continue;
      long type=HistoryDealGetInteger(deal,DEAL_TYPE);
      if(type==DEAL_TYPE_BALANCE || type==DEAL_TYPE_CREDIT)continue;
      double pnl=FTMODealDelta(deal);allDelta+=pnl;
      if(FTMODayKey(t)==today)delta+=pnl;
      long entry=HistoryDealGetInteger(deal,DEAL_ENTRY);
      if((type==DEAL_TYPE_BUY || type==DEAL_TYPE_SELL) && (entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT))FTMOAddTradingDay(FTMODayKey(t),t);
   }
   double balance=AccountInfoDouble(ACCOUNT_BALANCE);
   if(!MQLInfoInteger(MQL_TESTER) && MathAbs(balance-allDelta-InpFTMOInitialCapital)>.03){Print("FTMO: storico incompleto o capitale iniziale diverso; ingressi bloccati.");return false;}
   g_ftmoMidnight=balance-delta;g_ftmoDay=today;
   if(!MQLInfoInteger(MQL_TESTER)) {
      if(GlobalVariableCheck(FTMOStateKey("DAY")) && (int)GlobalVariableGet(FTMOStateKey("DAY"))==today && GlobalVariableCheck(FTMOStateKey("MIDNIGHT")))g_ftmoMidnight=MathMax(g_ftmoMidnight,GlobalVariableGet(FTMOStateKey("MIDNIGHT")));
      GlobalVariableSet(FTMOStateKey("DAY"),today);GlobalVariableSet(FTMOStateKey("MIDNIGHT"),g_ftmoMidnight);
      if(GlobalVariableCheck(FTMOStateKey("LOCKDAY")))g_ftmoLockedDay=(int)GlobalVariableGet(FTMOStateKey("LOCKDAY"));
   }
   return true;
}

bool FTMOEnsureReady() {
   if(g_ftmoReady)return true;
   datetime now=TimeCurrent();MqlDateTime d;TimeToStruct(FTMOClock(now),d);
   g_ftmoMonth=MQLInfoInteger(MQL_TESTER)&&InpResearchMonth>0?InpResearchMonth:d.year*100+d.mon;
   MqlDateTime a={};a.year=g_ftmoMonth/100;a.mon=g_ftmoMonth%100;a.day=1;g_ftmoStartCET=StructToTime(a);
   if(MQLInfoInteger(MQL_TESTER)&&InpResearchMonth>0){a.mon++;if(a.mon>12){a.year++;a.mon=1;}g_ftmoEndCET=StructToTime(a);}
   else g_ftmoStartCET=0;  
   if(!FTMORebuildDay())return false;
   if(!MQLInfoInteger(MQL_TESTER)) {
      g_ftmoFailed=GlobalVariableCheck(FTMOStateKey("FAILED"));
      if(InpFTMOStopAtTarget && GlobalVariableCheck(FTMOStateKey("PASSED"))){g_ftmoPassed=true;g_ftmoPassedServer=(datetime)GlobalVariableGet(FTMOStateKey("PASSED"));}
      g_ftmoTotalStopped=GlobalVariableCheck(FTMOStateKey("TOTAL_STOP"));
   }
   if(MQLInfoInteger(MQL_TESTER) && InpResearchMonth>0)g_ftmoElapsedStartCET=g_ftmoStartCET;
   else g_ftmoElapsedStartCET=FTMODayStart(FTMOClock(g_ftmoFirstEntry>0?g_ftmoFirstEntry:now));
   g_ftmoMinEquity=AccountInfoDouble(ACCOUNT_EQUITY);g_ftmoMinBalance=AccountInfoDouble(ACCOUNT_BALANCE);g_ftmoReady=true;return true;
}

double FTMOTotalFloor(){return InpFTMOInitialCapital-InpFTMOTotalLossEUR;}

double FTMODailyFloor(){return g_ftmoMidnight-InpFTMODailyLossEUR;}

double FTMOFloor(){return MathMax(FTMOTotalFloor(),FTMODailyFloor());}

double FTMOTarget(){return InpFTMOInitialCapital*(1+InpFTMOTargetPercent/100.0);}

void FTMOCloseOwn() {
   for(int p=PositionsTotal()-1;p>=0;p--) {
      ulong t=PositionGetTicket(p);if(t==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=m_symbol)continue;
      bool sent=trade.PositionClose(t);
      if(!sent || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))PrintFormat("FTMO: chiusura non completata %I64u, retcode %u",t,trade.ResultRetcode());
   }
}

void FTMOAuditObserve() {
   if(g_auditIndex<0 || g_auditKeys[g_auditIndex]!=g_ftmoDay){
      g_auditIndex=ArraySize(g_auditKeys);ArrayResize(g_auditKeys,g_auditIndex+1);ArrayResize(g_auditMidnight,g_auditIndex+1);ArrayResize(g_auditMinimum,g_auditIndex+1);
      g_auditKeys[g_auditIndex]=g_ftmoDay;g_auditMidnight[g_auditIndex]=g_ftmoMidnight;g_auditMinimum[g_auditIndex]=AccountInfoDouble(ACCOUNT_EQUITY);
   }
   g_auditMinimum[g_auditIndex]=MathMin(g_auditMinimum[g_auditIndex],AccountInfoDouble(ACCOUNT_EQUITY));
}

void FTMOExportDaily() {
   string name=(MQLInfoString(MQL_PROGRAM_NAME)+"_"+m_symbol)+"_FTMO_"+IntegerToString(g_ftmoMonth)+"_daily.csv";
   int h=FileOpen(name,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE)return;
   FileWrite(h,"day","midnight_balance","minimum_equity","daily_loss","total_floor","daily_floor");
   for(int i=0;i<ArraySize(g_auditKeys);i++)FileWrite(h,g_auditKeys[i],DoubleToString(g_auditMidnight[i],2),DoubleToString(g_auditMinimum[i],2),DoubleToString(MathMax(0,g_auditMidnight[i]-g_auditMinimum[i]),2),DoubleToString(FTMOTotalFloor(),2),DoubleToString(g_auditMidnight[i]-InpFTMODailyLossEUR,2));
   FileClose(h);
}

void FTMOObserve() {
   if(!g_ftmoReady)return;
   double eq=AccountInfoDouble(ACCOUNT_EQUITY),bal=AccountInfoDouble(ACCOUNT_BALANCE);
   g_ftmoMinEquity=MathMin(g_ftmoMinEquity,eq);g_ftmoMinBalance=MathMin(g_ftmoMinBalance,bal);
   g_ftmoWorstDaily=MathMax(g_ftmoWorstDaily,MathMax(0,g_ftmoMidnight-eq));
   FTMOAuditObserve();
   if(eq<FTMOTotalFloor() || eq<FTMODailyFloor()) {
      if(!g_ftmoFailed){g_ftmoFailureReason=eq<FTMOTotalFloor()?"limite_totale":"limite_giornaliero";PrintFormat("FTMO VIOLAZIONE: equity %.2f, limite %.2f",eq,FTMOFloor());}
      g_ftmoFailed=true;g_ftmoPassed=false;FTMOStoreFailure();
   }
}

void FTMOMarkPassed() {
   if(!InpFTMOStopAtTarget || g_ftmoFailed || g_ftmoPassed || !FTMOInEvaluation(TimeCurrent()))return;
   if(AccountInfoDouble(ACCOUNT_BALANCE)+.00001<FTMOTarget() || ArraySize(g_ftmoDays)<InpFTMOMinTradingDays || PositionsTotal()>0 || OrdersTotal()>0)return;
   g_ftmoPassed=true;g_ftmoPassedServer=TimeCurrent();
   if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("PASSED"),(double)g_ftmoPassedServer);
   PrintFormat("FTMO SUPERATA: saldo %.2f, giorni trading %d",AccountInfoDouble(ACCOUNT_BALANCE),ArraySize(g_ftmoDays));
}

bool FTMOPreRollover(){
 MqlDateTime now;TimeToStruct(TimeCurrent(),now);
 if(now.hour*60+now.min<InpRolloverCheckServerHour*60+InpRolloverCheckServerMinute)return false;
 if(InpCloseFriday && now.day_of_week==5){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();return true;}
 if(InpDailyFlat){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();return true;}
 if(InpOvernightOnlyProtected){
   for(int p=PositionsTotal()-1;p>=0;p--){
     ulong ticket=PositionGetTicket(p);if(ticket==0 || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber || PositionGetString(POSITION_SYMBOL)!=m_symbol)continue;
     bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
     double entry=PositionGetDouble(POSITION_PRICE_OPEN),sl=PositionGetDouble(POSITION_SL);
     if(sl<=0 || (buy && sl<=entry) || (!buy && sl>=entry)){
       bool sent=trade.PositionClose(ticket);
       if(!sent || (trade.ResultRetcode()!=TRADE_RETCODE_DONE && trade.ResultRetcode()!=TRADE_RETCODE_DONE_PARTIAL))PrintFormat("FTMO rollover: chiusura non completata %I64u",ticket);
     }
   }
   FTMOObserve();FTMOMarkPassed();
 }

 double used=0;
 for(int p=PositionsTotal()-1;p>=0;p--){
   ulong ticket=PositionGetTicket(p);if(ticket==0)continue;
   string symbol=PositionGetString(POSITION_SYMBOL);double sl=PositionGetDouble(POSITION_SL),pnl=0;MqlTick quote;
   bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
   if(sl<=0 || !SymbolInfoTick(symbol,quote) || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,sl,pnl)){FTMOCloseOwn();return true;}
   used+=MathMax(0,-pnl)+InpFTMOReservePerTradeEUR;
 }
 double futureFloor=MathMax(FTMOTotalFloor(),AccountInfoDouble(ACCOUNT_BALANCE)-InpFTMODailyLossEUR);
 if(AccountInfoDouble(ACCOUNT_EQUITY)-used<=futureFloor+InpFTMOBufferEUR){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();}
 return true;  
}

bool FTMOUpdate() {
   if(!FTMOEnsureReady())return false;
   if(FTMODayKey(TimeCurrent())!=g_ftmoDay && !FTMORebuildDay())return false;
   FTMOObserve();
   if(g_ftmoFailed || g_ftmoTotalStopped || (InpFTMOStopAtTarget && g_ftmoPassed)){FTMOCloseOwn();return false;}
   if(!FTMOInEvaluation(TimeCurrent())){if(FTMOClock(TimeCurrent())>=g_ftmoStartCET)FTMOCloseOwn();return false;}
   double eq=AccountInfoDouble(ACCOUNT_EQUITY);
   if(eq<=FTMOTotalFloor()+InpFTMOBufferEUR){g_ftmoTotalStopped=true;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("TOTAL_STOP"),1);}
   if(eq<=FTMOFloor()+InpFTMOBufferEUR){g_ftmoLockedDay=g_ftmoDay;if(!MQLInfoInteger(MQL_TESTER))GlobalVariableSet(FTMOStateKey("LOCKDAY"),g_ftmoLockedDay);FTMOCloseOwn();return false;}
   if(InpFTMOStopAtTarget && eq>=FTMOTarget()+InpFTMOReservePerTradeEUR && ArraySize(g_ftmoDays)>=InpFTMOMinTradingDays){FTMOCloseOwn();FTMOObserve();FTMOMarkPassed();return false;}
   FTMOMarkPassed();
   if(FTMOPreRollover())return false;
   return (!InpFTMOStopAtTarget || !g_ftmoPassed) && !g_ftmoTotalStopped && g_ftmoLockedDay!=g_ftmoDay;
}

double FTMORiskCapacity() {
   if(!g_ftmoReady || g_ftmoFailed || (InpFTMOStopAtTarget && g_ftmoPassed) || g_ftmoTotalStopped || g_ftmoLockedDay==g_ftmoDay)return 0;
   MqlTick tick;if(!SymbolInfoTick(m_symbol,tick))return 0;
   double used=0;int positions=0;
   for(int p=PositionsTotal()-1;p>=0;p--) {
      ulong t=PositionGetTicket(p);if(t==0)continue;
      string symbol=PositionGetString(POSITION_SYMBOL);bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
      double sl=PositionGetDouble(POSITION_SL),pnl=0;MqlTick quote;
      if(sl<=0 || !SymbolInfoTick(symbol,quote) || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,symbol,PositionGetDouble(POSITION_VOLUME),buy?quote.bid:quote.ask,sl,pnl))return 0;
      used+=MathMax(0,-pnl);positions++;
   }
   if(OrdersTotal()>0)return 0;
   return MathMax(0,AccountInfoDouble(ACCOUNT_EQUITY)-FTMOFloor()-InpFTMOBufferEUR-used-InpFTMOReservePerTradeEUR*(positions+1));
}

double FTMOEntryRisk(double requested) {
   if(MathMin(AccountInfoDouble(ACCOUNT_BALANCE),AccountInfoDouble(ACCOUNT_EQUITY))<InpFTMOInitialCapital)
      requested=MathMin(requested,InpFTMOInitialCapital*InpBelowInitialRiskPercent/100.0);
   if(InpFTMOStopAtTarget && AccountInfoDouble(ACCOUNT_BALANCE)>=FTMOTarget() && ArraySize(g_ftmoDays)<InpFTMOMinTradingDays)requested=MathMin(requested,InpFTMOInitialCapital*InpFTMOQualificationRiskPct/100.0);
   return MathMin(requested,FTMORiskCapacity());
}

void FTMOOnDeal(const MqlTradeTransaction &trans) {
   if(trans.type!=TRADE_TRANSACTION_DEAL_ADD || !g_ftmoReady || !HistoryDealSelect(trans.deal))return;
   datetime t=(datetime)HistoryDealGetInteger(trans.deal,DEAL_TIME);long type=HistoryDealGetInteger(trans.deal,DEAL_TYPE),entry=HistoryDealGetInteger(trans.deal,DEAL_ENTRY);
   if(FTMODayKey(t)!=g_ftmoDay && !FTMORebuildDay())return;
   if(FTMOInEvaluation(t) && (type==DEAL_TYPE_BUY || type==DEAL_TYPE_SELL) && (entry==DEAL_ENTRY_IN || entry==DEAL_ENTRY_INOUT))FTMOAddTradingDay(FTMODayKey(t),t);
   FTMOObserve();FTMOMarkPassed();
}

void FTMOExport() {
   if(!g_ftmoReady)FTMOEnsureReady();
   FTMOObserve();FTMOMarkPassed();
   FTMOExportDaily();
   double actualMinimum=TesterStatistics(STAT_EQUITYMIN);
   if(actualMinimum<FTMOTotalFloor()){g_ftmoFailed=true;g_ftmoPassed=false;g_ftmoFailureReason="limite_totale_statistiche_MT5";}
   string status=g_ftmoFailed?"violazione":g_ftmoPassed?"superato":g_ftmoTotalStopped?"arresto_protezione_totale":InpFTMOStopAtTarget?"non_superato":"continuo_completato";
   int calendarDays=g_ftmoPassed?1+(int)((FTMODayStart(FTMOClock(g_ftmoPassedServer))-g_ftmoElapsedStartCET)/86400):0;
   string filename=(MQLInfoString(MQL_PROGRAM_NAME)+"_"+m_symbol)+"_FTMO_"+IntegerToString(g_ftmoMonth)+"_result.csv";
   int h=FileOpen(filename,FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);if(h==INVALID_HANDLE){Print("FTMO CSV fallito: ",GetLastError());return;}
   FileWrite(h,"month","status","balance","equity","min_balance","min_equity","max_daily_loss","trading_days","calendar_days_to_pass","passed_server","first_entry_server","failure_reason","target_percent","daily_loss_limit","total_floor","buffer","clock_dst","stop_at_target");
   FileWrite(h,g_ftmoMonth,status,DoubleToString(AccountInfoDouble(ACCOUNT_BALANCE),2),DoubleToString(AccountInfoDouble(ACCOUNT_EQUITY),2),DoubleToString(TesterStatistics(STAT_BALANCEMIN),2),DoubleToString(actualMinimum,2),DoubleToString(g_ftmoWorstDaily,2),ArraySize(g_ftmoDays),calendarDays,g_ftmoPassed?TimeToString(g_ftmoPassedServer,TIME_DATE|TIME_SECONDS):"",g_ftmoFirstEntry>0?TimeToString(g_ftmoFirstEntry,TIME_DATE|TIME_SECONDS):"",g_ftmoFailureReason,InpFTMOTargetPercent,InpFTMODailyLossEUR,FTMOTotalFloor(),InpFTMOBufferEUR,(int)InpFTMOServerDST,InpFTMOStopAtTarget);
   FileClose(h);
}


void LoadCompletedORB(datetime start,int duration,bool &loaded,bool &isDefined,double &high,double &low){
 if(loaded || duration<=0 || TimeCurrent()<start+duration*60)return;
 MqlRates rates[];int copied=CopyRates(m_symbol,PERIOD_M1,start,start+duration*60-1,rates);if(copied<duration)return;
 double h=-DBL_MAX,l=DBL_MAX;for(int i=0;i<copied;i++){h=MathMax(h,rates[i].high);l=MathMin(l,rates[i].low);}
 if(h<=l || l<=0)return;high=h;low=l;isDefined=true;loaded=true;
}


int ResearchOwnPositionCount() {
    int count=0;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t>0 && PositionGetString(POSITION_SYMBOL)==m_symbol && PositionGetInteger(POSITION_MAGIC)==(long)InpMagicNumber) count++;
    }
    return count;
}

double ResearchOpenStopRisk() {
    double risk=0;
    MqlTick tick;if(!SymbolInfoTick(m_symbol,tick)) return DBL_MAX;
    for(int p=PositionsTotal()-1;p>=0;p--) {
        ulong t=PositionGetTicket(p);
        if(t==0 || PositionGetString(POSITION_SYMBOL)!=m_symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
        bool buy=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY;
        double stop=PositionGetDouble(POSITION_SL),pnl=0;
        if(stop<=0 || !OrderCalcProfit(buy?ORDER_TYPE_BUY:ORDER_TYPE_SELL,m_symbol,PositionGetDouble(POSITION_VOLUME),buy?tick.bid:tick.ask,stop,pnl)) return DBL_MAX;
        risk+=MathMax(0,-pnl);
    }
    return risk;
}


void OnTick() {
    if(!FTMOUpdate()) return;
    ManageOpenPositions();
    ManagePyramids();

    static datetime lastBarTime = 0;
    datetime currentBarTime = iTime(m_symbol, PERIOD_M1, 0);
    if(currentBarTime == lastBarTime) return;
    lastBarTime = currentBarTime;

    MqlDateTime dt; TimeCurrent(dt);
    datetime today = StringToTime(StringFormat("%04d.%02d.%02d", dt.year, dt.mon, dt.day));

    if(today != g_currentDay) {
        g_currentDay = today;
        g_dayStartEquity = AccountInfoDouble(ACCOUNT_EQUITY);
        g_tradesToday = 0; g_lossesToday = 0; g_dayLocked = false;
g_lonEntries=0;g_nyEntries=0;g_asiaEntries=0;g_lonLastEntry=0;g_nyLastEntry=0;g_asiaLastEntry=0;
        g_lonLoaded=false;g_nyLoaded=false;g_asiaLoaded=false;
        g_lonRangeDefined = false; g_nyRangeDefined = false; g_asiaRangeDefined = false;
        g_lonExecutedToday = false; g_nyExecutedToday = false; g_asiaExecutedToday = false;
        g_lonHigh = 0.0; g_lonLow = 999999.0;
        g_nyHigh = 0.0; g_nyLow = 999999.0;
        g_asiaHigh = 0.0; g_asiaLow = 999999.0;
        g_nyReArmActive = false;
        ArrayResize(g_daily_swing_highs, 0); ArrayResize(g_daily_swing_lows, 0);
        ArrayResize(g_countedLossIds, 0);
    }

    if(InpUseEquityDailyLock && g_dayStartEquity > 0) {
        double eq = AccountInfoDouble(ACCOUNT_EQUITY);
        double lossPct = (g_dayStartEquity - eq) / g_dayStartEquity * 100.0;
        if(lossPct >= InpMaxDailyLossPct) g_dayLocked = true;
    }

    LoadCompletedORB(today+InpLon_StartHour*3600+InpLon_StartMin*60,InpLon_DurationMin,g_lonLoaded,g_lonRangeDefined,g_lonHigh,g_lonLow);
 LoadCompletedORB(today+InpNY_StartHour*3600+InpNY_StartMin*60,InpNY_DurationMin,g_nyLoaded,g_nyRangeDefined,g_nyHigh,g_nyLow);
 if(InpUseAsia)LoadCompletedORB(today+InpAsia_StartHour*3600+InpAsia_StartMin*60,InpAsia_DurationMin,g_asiaLoaded,g_asiaRangeDefined,g_asiaHigh,g_asiaLow);
 if(g_dayLocked) { if(InpShowComments) Comment("STOP TRADING: Max Losses / Daily Loss Limit Reached"); return; }
    if(g_tradesToday >= InpMaxTradesDay) return;
    if(ResearchOwnPositionCount()>=InpMaximumPositions) return;

    double rearmClose=iClose(m_symbol,InpEntryTF,1);datetime rearmTime=iTime(m_symbol,InpEntryTF,1);
if(g_lonExecutedToday && g_lonEntries>0 && g_lonEntries<InpSessionMaxEntries && rearmTime>g_lonLastEntry+InpReentryCooldownMin*60 && rearmClose>g_lonLow && rearmClose<g_lonHigh)g_lonExecutedToday=false;
if(g_nyExecutedToday && g_nyEntries>0 && g_nyEntries<InpSessionMaxEntries && rearmTime>g_nyLastEntry+InpReentryCooldownMin*60 && rearmClose>g_nyLow && rearmClose<g_nyHigh)g_nyExecutedToday=false;
if(g_asiaExecutedToday && g_asiaEntries>0 && g_asiaEntries<InpSessionMaxEntries && rearmTime>g_asiaLastEntry+InpReentryCooldownMin*60 && rearmClose>g_asiaLow && rearmClose<g_asiaHigh)g_asiaExecutedToday=false;
static datetime lastD1 = 0;
    if(iTime(m_symbol, PERIOD_D1, 0) != lastD1) {
        lastD1 = iTime(m_symbol, PERIOD_D1, 0);
        double ph = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, true);
        double pl = GetPivot(PERIOD_D1, InpSwingLen_Daily, InpSwingLen_Daily+1, false);
        if(ph != EMPTY_VALUE) { AddToArray(g_daily_swing_highs, ph); if(ArraySize(g_daily_swing_highs)>10) ShiftArray(g_daily_swing_highs); }
        if(pl != EMPTY_VALUE) { AddToArray(g_daily_swing_lows, pl); if(ArraySize(g_daily_swing_lows)>10) ShiftArray(g_daily_swing_lows); }
        g_dailyBias = GetDailyBias();
    }

    double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);

     
    if(g_nyReArmActive) {
        if(TimeCurrent() > g_nyReArmDeadline) {
            g_nyReArmActive = false;
        } else if(!HasOpenPosition()) {
            double price = (g_nyFadeDir == -1) ? SymbolInfoDouble(m_symbol, SYMBOL_ASK) : SymbolInfoDouble(m_symbol, SYMBOL_BID);
            int revDir = -g_nyFadeDir;
            double sl = g_nyFadeSL;
            ExecuteTrade(revDir, price, sl, "NY_REVERSAL", GetRiskAmount(), 0);
            g_nyReArmActive = false;
        }
    }

     
    datetime lonStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpLon_StartHour, InpLon_StartMin));
    datetime lonEnd   = lonStart + (InpLon_DurationMin * 60);
    datetime lonScanLimit = lonEnd + (InpMaxEntryDelayMin * 60);

    if(!g_lonExecutedToday && TimeCurrent() >= lonStart) {
        if(TimeCurrent() < lonEnd) {
            double h = iHigh(m_symbol, PERIOD_M1, 1);
            double l = iLow(m_symbol, PERIOD_M1, 1);
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
                double m5_close = iClose(m_symbol, InpEntryTF, 1);
                double m5_open  = iOpen(m_symbol, InpEntryTF, 1);
                double m5_high  = iHigh(m_symbol, InpEntryTF, 1);
                double m5_low   = iLow(m_symbol, InpEntryTF, 1);

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
                    int beforeEntries=g_tradesToday;
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                         
                        if(bull) ExecuteTrade(1, m5_close, g_lonLow, "LONDON", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_lonHigh, "LONDON", riskAmt, 0);
                    }
                     
                     
                     
                     
                    if(g_tradesToday>beforeEntries) g_lonExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= lonScanLimit) {
             g_lonExecutedToday = true;
        }
    }

     
    datetime nyStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpNY_StartHour, InpNY_StartMin));
    datetime nyEnd   = nyStart + (InpNY_DurationMin * 60);
    datetime nyScanLimit = nyEnd + (InpMaxEntryDelayMin * 60);

    if(!g_nyExecutedToday && TimeCurrent() >= nyStart) {
        if(TimeCurrent() < nyEnd) {
            double h = iHigh(m_symbol, PERIOD_M1, 1);
            double l = iLow(m_symbol, PERIOD_M1, 1);
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
                double m5_close = iClose(m_symbol, InpEntryTF, 1);
                double m5_open  = iOpen(m_symbol, InpEntryTF, 1);
                double m5_high  = iHigh(m_symbol, InpEntryTF, 1);
                double m5_low   = iLow(m_symbol, InpEntryTF, 1);

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
                    int beforeEntries=g_tradesToday;
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
                    if(g_tradesToday>beforeEntries) g_nyExecutedToday = true;
                }
            }
        } else if(TimeCurrent() >= nyScanLimit) {
             g_nyExecutedToday = true;
        }
    }

     
    if(InpUseAsia) {
        datetime asiaStart = StringToTime(StringFormat("%04d.%02d.%02d %02d:%02d", dt.year, dt.mon, dt.day, InpAsia_StartHour, InpAsia_StartMin));
        datetime asiaEnd   = asiaStart + (InpAsia_DurationMin * 60);
        datetime asiaScanLimit = asiaEnd + (InpMaxEntryDelayMin * 60);

        if(!g_asiaExecutedToday && TimeCurrent() >= asiaStart) {
            if(TimeCurrent() < asiaEnd) {
                double h = iHigh(m_symbol, PERIOD_M1, 1);
                double l = iLow(m_symbol, PERIOD_M1, 1);
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
                    double m5_close = iClose(m_symbol, InpEntryTF, 1);
                    double m5_open  = iOpen(m_symbol, InpEntryTF, 1);
                    double m5_high  = iHigh(m_symbol, InpEntryTF, 1);
                    double m5_low   = iLow(m_symbol, InpEntryTF, 1);

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
                        int beforeEntries=g_tradesToday;
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
                        if(g_tradesToday>beforeEntries) g_asiaExecutedToday = true;
                    }
                }
            } else if(TimeCurrent() >= asiaScanLimit) {
                g_asiaExecutedToday = true;
            }
        }
    }

    if(InpShowComments) {
        Comment(StringFormat("Tri-Session v5.5 (Lon no-fade) | Bias: %s | Lon H/L: %.2f/%.2f | NY H/L: %.2f/%.2f | Asia H/L: %.2f/%.2f | Trades: %d",
              (g_dailyBias==1?"BULL":(g_dailyBias==-1?"BEAR":"NEUT")),
              g_lonHigh, g_lonLow, g_nyHigh, g_nyLow, g_asiaHigh, g_asiaLow, g_tradesToday));
    }
}


 
 
 
bool FTMODailyAllows(int dir){
 if((dir==1 && !InpDailyFilterLong) || (dir==-1 && !InpDailyFilterShort))return true;
 double ema[];if(CopyBuffer(g_dailyDirectionMA,0,1,1,ema)!=1)return false;
 double closed=iClose(m_symbol,PERIOD_D1,1);
 return dir==1?closed>ema[0]:closed<ema[0];
}

void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
 if(InpNASOnlyNY && session!="NY")return;
 if(!PortfolioUpdate() || PortfolioNewsBlocked(m_symbol))return;

    if(!FTMODailyAllows(dir))return;
    if((dir==1 && !InpAllowLong) || (dir==-1 && !InpAllowShort))return;
    if(ResearchOwnPositionCount()>=InpMaximumPositions) return;
    double availableStopRisk=AccountInfoDouble(ACCOUNT_EQUITY)*InpTotalOpenStopRiskPct/100.0-ResearchOpenStopRisk();
    riskAmount=MathMin(riskAmount,availableStopRisk);
    if(riskAmount<=0) return;

    if(StringFind(session,"ASIA")==0) riskAmount*=InpAsiaRiskMultiplier;
    else if(StringFind(session,"NY")==0) riskAmount*=InpNYRiskMultiplier;
    else if(session=="LONDON") riskAmount*=InpLondonRiskMultiplier;

    double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);

     
    MqlTick quote;
    if(!SymbolInfoTick(m_symbol,quote)) return;
    entry=dir==1?quote.ask:quote.bid;
    double tickSize=SymbolInfoDouble(m_symbol,SYMBOL_TRADE_TICK_SIZE);
    riskAmount=FTMOEntryRisk(riskAmount);
 riskAmount=PortfolioEntryRisk(riskAmount,m_symbol);if(riskAmount<=0)return;
    if(tickSize<=0 || riskAmount<=0 || entry<=0) return;
    double sl_final=NormalizeDouble(MathRound(sl_ref/tickSize)*tickSize,digits);
    double minStop=SymbolInfoInteger(m_symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
    if((dir==1 && (sl_final>=quote.bid-minStop)) || (dir==-1 && sl_final<=quote.ask+minStop)) return;

    double tp_final = 0;
    if(customTP > 0) {
        tp_final = NormalizeDouble(customTP, digits);
    } else {
        double dist = MathAbs(entry - sl_final);
        if(dist == 0) return;
        tp_final = NormalizeDouble((dir == 1) ? entry + (dist * InpRR_Target) : entry - (dist * InpRR_Target), digits);
    }

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

    double minLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
    double profitPerLot = 0;

    bool calcRes = false;
    if(dir == 1) calcRes = OrderCalcProfit(ORDER_TYPE_BUY, m_symbol, 1.0, entry, sl_final, profitPerLot);
    else calcRes = OrderCalcProfit(ORDER_TYPE_SELL, m_symbol, 1.0, entry, sl_final, profitPerLot);

    if(!calcRes || profitPerLot >= 0) return;

    double lots = MathFloor((riskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
     
    if(lots < minLot || stepLot<=0) return;
    lots=MathMin(lots,MathFloor(SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_MAX)/stepLot)*stepLot);
    double margin=0;
    if(!OrderCalcMargin(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,m_symbol,lots,entry,margin) || margin<=0) return;
    double availableMargin=AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.80;
    if(margin>availableMargin) lots=MathFloor(lots*availableMargin/margin/stepLot)*stepLot;
    if(lots<minLot) return;

    string cmt = StringFormat("%s|v5.5|R:%.2f", session, riskAmount);
    bool res = false;

    if(dir == 1) res = trade.Buy(lots, m_symbol, 0.0, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, m_symbol, 0.0, sl_final, tp_final, cmt);

    if(res && (trade.ResultRetcode()==TRADE_RETCODE_DONE || trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL)) {
        g_tradesToday++;
if(session=="LONDON"){g_lonEntries++;g_lonLastEntry=TimeCurrent();}
else if(StringFind(session,"NY")==0){g_nyEntries++;g_nyLastEntry=TimeCurrent();}
else if(StringFind(session,"ASIA")==0){g_asiaEntries++;g_asiaLastEntry=TimeCurrent();}

        long posId=0; double filledEntry=entry;
        ulong filledDeal=trade.ResultDeal();
        if(filledDeal>0 && HistoryDealSelect(filledDeal)) {
            posId=HistoryDealGetInteger(filledDeal,DEAL_POSITION_ID);
            filledEntry=HistoryDealGetDouble(filledDeal,DEAL_PRICE);
        }
        if(posId > 0) RegisterPosition(posId, session, filledEntry, MathAbs(filledEntry - sl_final), dir);

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


 
 
 
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
    FTMOOnDeal(trans);
    if(trans.type != TRADE_TRANSACTION_DEAL_ADD) return;
    if(!HistoryDealSelect(trans.deal)) return;
    if(HistoryDealGetInteger(trans.deal, DEAL_MAGIC) != (long)InpMagicNumber) return;
    if(HistoryDealGetString(trans.deal, DEAL_SYMBOL) != m_symbol) return;

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

 
 
double OnTester() {
    FTMOExport();
    if(!HistorySelect(0,TimeCurrent())) return TesterStatistics(STAT_PROFIT);
    long ids[];
    for(int i=0;i<HistoryDealsTotal();i++) {
        ulong deal=HistoryDealGetTicket(i);
        if(HistoryDealGetString(deal,DEAL_SYMBOL)!=m_symbol || HistoryDealGetInteger(deal,DEAL_MAGIC)!=(long)InpMagicNumber) continue;
        long id=HistoryDealGetInteger(deal,DEAL_POSITION_ID);
        bool found=false;
        for(int j=0;j<ArraySize(ids);j++) if(ids[j]==id) {found=true;break;}
        if(id>0 && !found) {int n=ArraySize(ids);ArrayResize(ids,n+1);ids[n]=id;}
    }
    string filename=(MQLInfoString(MQL_PROGRAM_NAME)+"_"+m_symbol)+"_"+IntegerToString(g_ftmoMonth)+"_trades.csv";
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
bool RestorePortfolioPositions(){
 for(int p=PositionsTotal()-1;p>=0;p--){
  ulong ticket=PositionGetTicket(p);if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=m_symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber)continue;
  long id=PositionGetInteger(POSITION_IDENTIFIER);double entry=PositionGetDouble(POSITION_PRICE_OPEN),initial=0;
  double initialVolume=0,closedVolume=0;bool partial=false;
  int dir=PositionGetInteger(POSITION_TYPE)==POSITION_TYPE_BUY?1:-1;
  string session="RESTORED",comment=PositionGetString(POSITION_COMMENT),segments[];
  if(StringSplit(comment,'|',segments)>0)session=segments[0];
  if(!HistorySelectByPosition(id))return false;
  for(int j=0;j<HistoryOrdersTotal();j++){
   ulong order=HistoryOrderGetTicket(j);long type=HistoryOrderGetInteger(order,ORDER_TYPE);
   if(type!=ORDER_TYPE_BUY && type!=ORDER_TYPE_SELL)continue;
   double stop=HistoryOrderGetDouble(order,ORDER_SL);
   if(stop>0 && dir*(entry-stop)>0){initial=MathAbs(entry-stop);break;}
  }
  for(int j=0;j<HistoryDealsTotal();j++){
   ulong deal=HistoryDealGetTicket(j);long type=HistoryDealGetInteger(deal,DEAL_TYPE);
   if(type!=DEAL_TYPE_BUY && type!=DEAL_TYPE_SELL)continue;
   long side=HistoryDealGetInteger(deal,DEAL_ENTRY);
   if(side==DEAL_ENTRY_IN)initialVolume+=HistoryDealGetDouble(deal,DEAL_VOLUME);
   else if(side==DEAL_ENTRY_OUT)closedVolume+=HistoryDealGetDouble(deal,DEAL_VOLUME);
  }
  if(initial<=0 || initialVolume<=0){Print("PORTFOLIO_RESTART_RISK_UNKNOWN ",m_symbol);return false;}
  partial=closedVolume>0;
  if(!PositionSelectByTicket(ticket))return false;
  double currentStop=PositionGetDouble(POSITION_SL);
  RegisterPosition(id,session,entry,initial,dir);int idx=FindSessionMapIndex(id);
  if(idx<0)return false;
  g_posSessionMap[idx].partialDone=partial;
  g_posSessionMap[idx].beDone=currentStop>0 && dir*(currentStop-entry)>=InpBreakEvenLockR*initial;
 }
 return true;
}

NASEngine(){InpLon_StartHour=NAS_InpLon_StartHour;InpLon_StartMin=NAS_InpLon_StartMin;InpLon_DurationMin=NAS_InpLon_DurationMin;InpLon_UseDailyFilter=NAS_InpLon_UseDailyFilter;InpLon_IgnoreBias=NAS_InpLon_IgnoreBias;InpNY_StartHour=NAS_InpNY_StartHour;InpNY_StartMin=NAS_InpNY_StartMin;InpNY_DurationMin=NAS_InpNY_DurationMin;InpNY_UseDailyFilter=NAS_InpNY_UseDailyFilter;InpUseAsia=NAS_InpUseAsia;InpAsia_StartHour=NAS_InpAsia_StartHour;InpAsia_StartMin=NAS_InpAsia_StartMin;InpAsia_DurationMin=NAS_InpAsia_DurationMin;InpAsia_UseDailyFilter=NAS_InpAsia_UseDailyFilter;InpEntryTF=NAS_InpEntryTF;InpMaxEntryDelayMin=NAS_InpMaxEntryDelayMin;InpBodyFilterPct=NAS_InpBodyFilterPct;InpFadeEnabled=NAS_InpFadeEnabled;InpFadeRiskPct=NAS_InpFadeRiskPct;InpFadeSLBufferPoints=NAS_InpFadeSLBufferPoints;InpATRMinRangeMultAsia=NAS_InpATRMinRangeMultAsia;InpATRMinRangeMultNY=NAS_InpATRMinRangeMultNY;InpUseATRFilter=NAS_InpUseATRFilter;InpATRPeriod=NAS_InpATRPeriod;InpATRMinRangeMult=NAS_InpATRMinRangeMult;InpUseBreakEven=NAS_InpUseBreakEven;InpBreakEvenTriggerR=NAS_InpBreakEvenTriggerR;InpBreakEvenLockR=NAS_InpBreakEvenLockR;InpUsePartialClose=NAS_InpUsePartialClose;InpPartialClosePct=NAS_InpPartialClosePct;InpPartialCloseAtR=NAS_InpPartialCloseAtR;InpFixedRiskAmount=NAS_InpFixedRiskAmount;InpUseEquityRisk=NAS_InpUseEquityRisk;InpRiskPercent=NAS_InpRiskPercent;InpRR_Target=NAS_InpRR_Target;InpMaxLossesDay=NAS_InpMaxLossesDay;InpMaxTradesDay=NAS_InpMaxTradesDay;InpMagicNumber=NAS_InpMagicNumber;InpShowComments=NAS_InpShowComments;InpUseEquityDailyLock=NAS_InpUseEquityDailyLock;InpMaxDailyLossPct=NAS_InpMaxDailyLossPct;InpPyramidAtR=NAS_InpPyramidAtR;InpPyramidRiskPercent=NAS_InpPyramidRiskPercent;InpPyramidExtraTargetR=NAS_InpPyramidExtraTargetR;InpTotalOpenStopRiskPct=NAS_InpTotalOpenStopRiskPct;g_profitPyramid=0;g_winsPyramid=0;g_totalPyramid=0;InpAsiaRiskMultiplier=NAS_InpAsiaRiskMultiplier;InpNYRiskMultiplier=NAS_InpNYRiskMultiplier;InpLondonRiskMultiplier=NAS_InpLondonRiskMultiplier;InpFTMOInitialCapital=NAS_InpFTMOInitialCapital;InpFTMOTargetPercent=NAS_InpFTMOTargetPercent;InpFTMOStopAtTarget=NAS_InpFTMOStopAtTarget;InpFTMOMinTradingDays=NAS_InpFTMOMinTradingDays;InpFTMODailyLossEUR=NAS_InpFTMODailyLossEUR;InpFTMOTotalLossEUR=NAS_InpFTMOTotalLossEUR;InpFTMOBufferEUR=NAS_InpFTMOBufferEUR;InpFTMOReservePerTradeEUR=NAS_InpFTMOReservePerTradeEUR;InpFTMOQualificationRiskPct=NAS_InpFTMOQualificationRiskPct;InpFTMOServerUTCWinter=NAS_InpFTMOServerUTCWinter;InpFTMOServerUTCSummer=NAS_InpFTMOServerUTCSummer;InpFTMOServerDST=NAS_InpFTMOServerDST;InpFTMOBrokerClockConfirmed=NAS_InpFTMOBrokerClockConfirmed;InpResearchMonth=NAS_InpResearchMonth;InpResearchStopServer=NAS_InpResearchStopServer;g_ftmoReady=false;g_ftmoFailed=false;g_ftmoPassed=false;g_ftmoTotalStopped=false;g_ftmoDay=0;g_ftmoLockedDay=0;g_ftmoMonth=0;g_ftmoMidnight=10000;g_ftmoMinEquity=10000;g_ftmoMinBalance=10000;g_ftmoWorstDaily=0;g_ftmoStartCET=0;g_ftmoEndCET=0;g_ftmoPassedServer=0;g_ftmoFirstEntry=0;g_ftmoFailureReason="";InpMaximumPositions=NAS_InpMaximumPositions;InpSessionMaxEntries=NAS_InpSessionMaxEntries;InpReentryCooldownMin=NAS_InpReentryCooldownMin;g_lonEntries=0;g_nyEntries=0;g_asiaEntries=0;g_lonLastEntry=0;g_nyLastEntry=0;g_asiaLastEntry=0;g_auditIndex=-1;InpAllowLong=NAS_InpAllowLong;InpAllowShort=NAS_InpAllowShort;InpCloseFriday=NAS_InpCloseFriday;InpRolloverCheckServerHour=NAS_InpRolloverCheckServerHour;InpRolloverCheckServerMinute=NAS_InpRolloverCheckServerMinute;InpDailyDirectionPeriod=NAS_InpDailyDirectionPeriod;InpDailyFilterLong=NAS_InpDailyFilterLong;InpDailyFilterShort=NAS_InpDailyFilterShort;g_dailyDirectionMA=INVALID_HANDLE;InpBelowInitialRiskPercent=NAS_InpBelowInitialRiskPercent;InpOvernightOnlyProtected=NAS_InpOvernightOnlyProtected;InpDailyFlat=NAS_InpDailyFlat;g_ftmoElapsedStartCET=0;g_lonLoaded=false;g_nyLoaded=false;g_asiaLoaded=false;g_atrHandle=INVALID_HANDLE;g_lonHigh=0.0;g_lonLow=999999.0;g_lonRangeDefined=false;g_lonExecutedToday=false;g_nyHigh=0.0;g_nyLow=999999.0;g_nyRangeDefined=false;g_nyExecutedToday=false;g_asiaHigh=0.0;g_asiaLow=999999.0;g_asiaRangeDefined=false;g_asiaExecutedToday=false;g_dailyBias=0;InpSwingLen_Daily=5;g_currentDay=0;g_dayStartEquity=0.0;g_tradesToday=0;g_lossesToday=0;g_dayLocked=false;g_profitLon=0;g_profitNY=0;g_profitAsia=0;g_winsLon=0;g_winsNY=0;g_winsAsia=0;g_totalLon=0;g_totalNY=0;g_totalAsia=0;g_profitNYFade=0;g_profitAsiaFade=0;g_winsNYFade=0;g_winsAsiaFade=0;g_totalNYFade=0;g_totalAsiaFade=0;g_profitNYRev=0;g_winsNYRev=0;g_totalNYRev=0;g_nyFadeDir=0;g_nyFadeSL=0;g_nyReArmActive=false;g_nyReArmDeadline=0;}
};

NASEngine nas;
bool g_nasInitialized=false;
int OnInit(){
 if(!MQLInfoInteger(MQL_TESTER) && !InpStandardNewsValidated)return INIT_FAILED;
 if(!PortfolioInit(InpMagicNumber) || !PortfolioNewsInit())return INIT_PARAMETERS_INCORRECT;
 if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)return INIT_PARAMETERS_INCORRECT;
 if(!SymbolSelect(InpNASsymbol,true))return INIT_FAILED;
 nas.m_symbol=InpNASsymbol;if(nas.OnInit()!=INIT_SUCCEEDED)return INIT_FAILED;g_nasInitialized=true;
 EventSetTimer(5);return INIT_SUCCEEDED;
}
void RunNAS(){
 if(!PortfolioUpdate() || PortfolioNewsBlocked(InpNASsymbol))return;
 MqlTick q;if(!g_nasInitialized || !SymbolInfoTick(InpNASsymbol,q) || q.bid<=0 || q.ask<q.bid || q.time>TimeCurrent() || TimeCurrent()-q.time>60)return;
 nas.OnTick();
}
void OnTick(){RunNAS();}
void OnTimer(){RunNAS();}
void OnTradeTransaction(const MqlTradeTransaction &t,const MqlTradeRequest &r,const MqlTradeResult &s){if(g_nasInitialized)nas.OnTradeTransaction(t,r,s);PortfolioUpdate();}
void OnDeinit(const int reason){EventKillTimer();if(g_nasInitialized)nas.OnDeinit(reason);PortfolioExport();}
double OnTester() {
    if(!HistorySelect(0,TimeCurrent())) return TesterStatistics(STAT_PROFIT);
    long ids[];
    for(int i=0;i<HistoryDealsTotal();i++) {
        ulong deal=HistoryDealGetTicket(i);
        if(!PortfolioOwn(HistoryDealGetInteger(deal,DEAL_MAGIC)))continue;
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
