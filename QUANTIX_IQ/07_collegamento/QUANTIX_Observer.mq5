// Lettore passivo dei commenti del grafico. Nessun ordine o modifica al bot.
#property strict
#property version "1.01"
#property indicator_chart_window
#property indicator_plots 0
#property indicator_buffers 0

input long InpExpectedLogin=0;
input string InpExpectedServer="";
input string InpObservedSymbol="XAUUSD";
input string InpObservedBot="prova298";

string JsonText(string value){
   StringReplace(value,"\\","\\\\");StringReplace(value,"\"","\\\"");
   StringReplace(value,"\r","\\r");StringReplace(value,"\n","\\n");StringReplace(value,"\t","\\t");
   return "\""+value+"\"";
}

void Publish(){
   long account=AccountInfoInteger(ACCOUNT_LOGIN);
   string server=AccountInfoString(ACCOUNT_SERVER);
   if(account!=InpExpectedLogin || server!=InpExpectedServer)return;
   long target=-1;
   for(long chart=ChartFirst();chart>=0;chart=ChartNext(chart)){
      if(ChartSymbol(chart)==InpObservedSymbol && ChartGetString(chart,CHART_EXPERT_NAME)==InpObservedBot){
         target=chart;break;
      }
   }
   string text=target>=0?ChartGetString(target,CHART_COMMENT):"";
   string data="{\"version\":1,\"read_only\":true,\"account\":"+IntegerToString(account)+",\"server\":"+JsonText(server)+",\"symbol\":"+JsonText(InpObservedSymbol)+",\"bot\":"+JsonText(InpObservedBot)+",\"chart_found\":"+(target>=0?"true":"false")+",\"observed_utc_epoch\":"+IntegerToString((long)TimeGMT())+",\"server_epoch\":"+IntegerToString((long)TimeCurrent())+",\"chart_id\":"+JsonText(IntegerToString(target))+",\"comment\":"+JsonText(text)+"}";
   FolderCreate("QuantixIQ",FILE_COMMON);
   int file=FileOpen("QuantixIQ\\observer.json",FILE_WRITE|FILE_TXT|FILE_ANSI|FILE_COMMON|FILE_SHARE_READ,0,CP_UTF8);
   if(file==INVALID_HANDLE)return;
   FileWriteString(file,data);FileClose(file);
}

int OnInit(){
   if(InpExpectedLogin<=0 || StringLen(InpExpectedServer)==0 || StringLen(InpObservedSymbol)==0 || StringLen(InpObservedBot)==0){
      Print("Configurare conto, server, simbolo e bot attesi negli input dell indicatore.");
      return INIT_PARAMETERS_INCORRECT;
   }
   IndicatorSetString(INDICATOR_SHORTNAME,"QUANTIX Observer (sola lettura)");
   EventSetTimer(1);Publish();return INIT_SUCCEEDED;
}
void OnTimer(){Publish();}
void OnDeinit(const int reason){EventKillTimer();}
int OnCalculate(const int rates_total,const int prev_calculated,const int begin,const double &price[]){return rates_total;}
