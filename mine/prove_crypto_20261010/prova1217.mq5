// prova1217: Portfolio NAS1211 + XAU183: tutti gli input e corpi dei metodi originali preservati; magics separati; equity/margine condivisi; solo tester, nessuna attivazione. Motori abilitati nel controllo: nas
// Ricerca crypto; non attivato sul conto. Ore delle sessioni in tempo server.
#property strict
#include <Trade\Trade.mqh>
input ulong InpMagicNumber=20261011217;
input bool InpShowComments=false;
input int NAS_InpLon_StartHour=10;
input int NAS_InpLon_StartMin=0;
input int NAS_InpLon_DurationMin=60;
input bool NAS_InpLon_UseDailyFilter=false;
input bool NAS_InpLon_IgnoreBias=true;
input int NAS_InpNY_StartHour=16;
input int NAS_InpNY_StartMin=30;
input int NAS_InpNY_DurationMin=30;
input bool NAS_InpNY_UseDailyFilter=false;
input bool NAS_InpUseAsia=true;
input int NAS_InpAsia_StartHour=3;
input int NAS_InpAsia_StartMin=0;
input int NAS_InpAsia_DurationMin=120;
input bool NAS_InpAsia_UseDailyFilter=false;
input ENUM_TIMEFRAMES NAS_InpEntryTF=PERIOD_M5;
input int NAS_InpMaxEntryDelayMin=360;
input double NAS_InpBodyFilterPct=35;
input bool NAS_InpFadeEnabled=false;
input double NAS_InpFadeRiskPct=50.0;
input double NAS_InpFadeSLBufferPoints=30;
input double NAS_InpATRMinRangeMultAsia=0.1;
input double NAS_InpATRMinRangeMultNY=0.15;
input bool NAS_InpUseATRFilter=true;
input int NAS_InpATRPeriod=14;
input double NAS_InpATRMinRangeMult=0.15;
input bool NAS_InpUseBreakEven=false;
input double NAS_InpBreakEvenTriggerR=1.0;
input double NAS_InpBreakEvenLockR=0.1;
input bool NAS_InpUsePartialClose=false;
input double NAS_InpPartialClosePct=50.0;
input double NAS_InpPartialCloseAtR=1.0;
input double NAS_InpFixedRiskAmount=80.0;
input bool NAS_InpUseEquityRisk=true;
input double NAS_InpRiskPercent=0.64;
input double NAS_InpRR_Target=4;
input int NAS_InpMaxLossesDay=6;
input int NAS_InpMaxTradesDay=9;
input ulong NAS_InpMagicNumber=20261011211;
input bool NAS_InpShowComments=false;
input bool NAS_InpUseEquityDailyLock=false;
input double NAS_InpMaxDailyLossPct=3.0;
input double NAS_InpPyramidAtR=1.2;
input double NAS_InpPyramidRiskPercent=0;
input double NAS_InpPyramidExtraTargetR=1.0;
input double NAS_InpTotalOpenStopRiskPct=0.64;
input double NAS_InpAsiaRiskMultiplier=1.5;
input double NAS_InpNYRiskMultiplier=0.5;
input double NAS_InpLondonRiskMultiplier=1;
input bool NAS_InpTradeWeekend=false;
input double NAS_InpMaxSpreadToStopPct=10.0;
input bool NAS_InpRequireWeekendSessions=false;
input bool NAS_InpNasOnlyNY=false;
input bool NAS_InpNasCloseEndOfDay=true;
input int NAS_InpNasCloseHour=23;
input int NAS_InpNasCloseMinute=45;
input int NAS_InpNasMaxHoldMinutes=240;
input bool NAS_InpNasAllowLong=true;
input bool NAS_InpNasAllowShort=true;
input bool NAS_InpNasTrendFilter=true;
input int NAS_InpNasTrendFast=20;
input int NAS_InpNasTrendSlow=100;
input double NAS_InpNasStopFraction=1;
input bool NAS_InpNasReentry=false;
input int NAS_InpNasMaxSessionEntries=2;
input bool NAS_InpNasAdaptiveRisk=false;
input double NAS_InpNasAdaptiveSoftDDPct=2;
input double NAS_InpNasAdaptiveHardDDPct=9.2;
input double NAS_InpNasAdaptiveCurve=2;
input double NAS_InpNasAdaptiveMinFraction=0.10;
input double NAS_InpNasAdaptiveRoomFraction=0.3;
input double NAS_InpNasMaxBreakoutExtension=0.8;
input bool NAS_InpNasQualityStrongOnly=true;
input bool NAS_InpNasQualityCandleDirection=true;
input double NAS_InpNasQualityMinBodyPct=35;
input double NAS_InpNasAsiaLongRisk=2.0;
input double NAS_InpNasAsiaShortRisk=0.35;
input double NAS_InpNasLondonLongRisk=0.35;
input double NAS_InpNasLondonShortRisk=7;
input double NAS_InpNasNYLongRisk=1.5;
input double NAS_InpNasNYShortRisk=0.35;
input double NAS_InpNasTrailStartR=1;
input double NAS_InpNasTrailGapR=0.5;
input double NAS_InpNasTrailStepR=0.25;
input double NAS_InpNasClockWeakRisk=0.01;
input int NAS_InpNasAsiaLongSoftHoursMask=512;
input int NAS_InpNasLondonShortSoftHoursMask=40960;
input int NAS_InpNasNYLongSoftHoursMask=3145728;
enum ENUM_FTMO_SERVER_DST { FTMO_DST_EU=0, FTMO_DST_US=1, FTMO_DST_NONE=2 };
input int XAU_InpLon_StartHour=8;
input int XAU_InpLon_StartMin=0;
input int XAU_InpLon_DurationMin=60;
input bool XAU_InpLon_UseDailyFilter=true;
input bool XAU_InpLon_IgnoreBias=true;
input int XAU_InpNY_StartHour=15;
input int XAU_InpNY_StartMin=30;
input int XAU_InpNY_DurationMin=30;
input bool XAU_InpNY_UseDailyFilter=true;
input bool XAU_InpUseAsia=true;
input int XAU_InpAsia_StartHour=1;
input int XAU_InpAsia_StartMin=0;
input int XAU_InpAsia_DurationMin=60;
input bool XAU_InpAsia_UseDailyFilter=true;
input ENUM_TIMEFRAMES XAU_InpEntryTF=PERIOD_M5;
input int XAU_InpMaxEntryDelayMin=360;
input double XAU_InpBodyFilterPct=25.0;
input bool XAU_InpFadeEnabled=false;
input double XAU_InpFadeRiskPct=50.0;
input double XAU_InpFadeSLBufferPoints=30;
input double XAU_InpATRMinRangeMultAsia=0.1;
input double XAU_InpATRMinRangeMultNY=0.15;
input bool XAU_InpUseATRFilter=true;
input int XAU_InpATRPeriod=14;
input double XAU_InpATRMinRangeMult=0.15;
input bool XAU_InpUseBreakEven=true;
input double XAU_InpBreakEvenTriggerR=1.0;
input double XAU_InpBreakEvenLockR=0.1;
input bool XAU_InpUsePartialClose=false;
input double XAU_InpPartialClosePct=50.0;
input double XAU_InpPartialCloseAtR=1.0;
input double XAU_InpFixedRiskAmount=80.0;
input bool XAU_InpUseEquityRisk=true;
input double XAU_InpRiskPercent=2.0;
input double XAU_InpRR_Target=2;
input int XAU_InpMaxLossesDay=4;
input int XAU_InpMaxTradesDay=12;
input ulong XAU_InpMagicNumber=2026101083;
input bool XAU_InpShowComments=true;
input bool XAU_InpUseEquityDailyLock=false;
input double XAU_InpMaxDailyLossPct=3.0;
input double XAU_InpPyramidAtR=1.2;
input double XAU_InpPyramidRiskPercent=0.4;
input double XAU_InpPyramidExtraTargetR=1.0;
input double XAU_InpTotalOpenStopRiskPct=4.0;
input double XAU_InpAsiaRiskMultiplier=1.05;
input double XAU_InpNYRiskMultiplier=0.8;
input double XAU_InpLondonRiskMultiplier=1.1;
input double XAU_InpFTMOInitialCapital=10000.0;
input double XAU_InpFTMOTargetPercent=10.0;
input bool XAU_InpFTMOStopAtTarget=false;
input int XAU_InpFTMOMinTradingDays=4;
input double XAU_InpFTMODailyLossEUR=500.0;
input double XAU_InpFTMOTotalLossEUR=1000.0;
input double XAU_InpFTMOBufferEUR=150.0;
input double XAU_InpFTMOReservePerTradeEUR=5.0;
input double XAU_InpFTMOQualificationRiskPct=1.0;
input int XAU_InpFTMOServerUTCWinter=2;
input int XAU_InpFTMOServerUTCSummer=3;
input ENUM_FTMO_SERVER_DST XAU_InpFTMOServerDST=FTMO_DST_EU;
input bool XAU_InpFTMOBrokerClockConfirmed=false;
input int XAU_InpResearchMonth=0;
input long XAU_InpResearchStopServer=0;
input int XAU_InpMaximumPositions=3;
input int XAU_InpSessionMaxEntries=2;
input int XAU_InpReentryCooldownMin=20;
input bool XAU_InpAllowLong=true;
input bool XAU_InpAllowShort=true;
input bool XAU_InpCloseFriday=true;
input int XAU_InpRolloverCheckServerHour=22;
input int XAU_InpRolloverCheckServerMinute=45;
input int XAU_InpDailyDirectionPeriod=20;
input bool XAU_InpDailyFilterLong=false;
input bool XAU_InpDailyFilterShort=true;
input double XAU_InpBelowInitialRiskPercent=1.25;
input bool XAU_InpOvernightOnlyProtected=true;
input bool XAU_InpDailyFlat=false;
class NASEngine{
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
bool InpTradeWeekend;
double InpMaxSpreadToStopPct;
bool InpRequireWeekendSessions;
bool g_lonLoaded,g_nyLoaded,g_asiaLoaded;
bool InpNasOnlyNY;
bool InpNasCloseEndOfDay;
int InpNasCloseHour;
int InpNasCloseMinute;
int InpNasMaxHoldMinutes;
bool InpNasAllowLong;
bool InpNasAllowShort;
bool InpNasTrendFilter;
int InpNasTrendFast;
int InpNasTrendSlow;
double InpNasStopFraction;
int g_nasTrendFast,g_nasTrendSlow;
bool InpNasReentry;
int InpNasMaxSessionEntries;
int g_nasLonEntries,g_nasNYEntries,g_nasAsiaEntries;
datetime g_nasEntryDay,g_nasRearmBar,g_nasLonFillBar,g_nasNYFillBar,g_nasAsiaFillBar;
bool InpNasAdaptiveRisk;
double InpNasAdaptiveSoftDDPct;
double InpNasAdaptiveHardDDPct;
double InpNasAdaptiveCurve;
double InpNasAdaptiveMinFraction;
double InpNasAdaptiveRoomFraction;
double g_nasAdaptivePeak;
bool g_nasAdaptiveHalted;

 
void NasAdaptiveObserve() {
    if(!InpNasAdaptiveRisk) return;
    double eq=AccountInfoDouble(ACCOUNT_EQUITY);
    g_nasAdaptivePeak=MathMax(g_nasAdaptivePeak,eq);
    if(g_nasAdaptivePeak<=0) return;
    double floorEquity=g_nasAdaptivePeak*(1-InpNasAdaptiveHardDDPct/100.0);
    if(eq<=floorEquity) g_nasAdaptiveHalted=true;
}

double NasAdaptiveBudget(double requested) {
    if(!InpNasAdaptiveRisk) return requested;
    if(g_nasAdaptiveHalted || g_nasAdaptivePeak<=0) return 0;
    double eq=AccountInfoDouble(ACCOUNT_EQUITY);
    double dd=MathMax(0,(g_nasAdaptivePeak-eq)/g_nasAdaptivePeak*100.0);
    double progress=MathMax(0,MathMin(1,(dd-InpNasAdaptiveSoftDDPct)/(InpNasAdaptiveHardDDPct-InpNasAdaptiveSoftDDPct)));
    double scale=InpNasAdaptiveMinFraction+(1-InpNasAdaptiveMinFraction)*MathPow(1-progress,InpNasAdaptiveCurve);
    double room=MathMax(0,eq-g_nasAdaptivePeak*(1-InpNasAdaptiveHardDDPct/100.0));
    return MathMin(requested*scale,room*InpNasAdaptiveRoomFraction);
}
double InpNasMaxBreakoutExtension;
bool InpNasQualityStrongOnly;
bool InpNasQualityCandleDirection;
double InpNasQualityMinBodyPct;
double InpNasAsiaLongRisk;
double InpNasAsiaShortRisk;
double InpNasLondonLongRisk;
double InpNasLondonShortRisk;
double InpNasNYLongRisk;
double InpNasNYShortRisk;

 
double NasDirectionRisk(string session,int dir) {
    if(StringFind(session,"ASIA")==0) return dir==1?InpNasAsiaLongRisk:InpNasAsiaShortRisk;
    if(session=="LONDON") return dir==1?InpNasLondonLongRisk:InpNasLondonShortRisk;
    if(StringFind(session,"NY")==0) return dir==1?InpNasNYLongRisk:InpNasNYShortRisk;
    return 1;
}
double InpNasTrailStartR;
double InpNasTrailGapR;
double InpNasTrailStepR;
double InpNasClockWeakRisk;
int InpNasAsiaLongSoftHoursMask;
int InpNasLondonShortSoftHoursMask;
int InpNasNYLongSoftHoursMask;

double NasClockRisk(string session,int dir,int hour) {
    int mask=0;
    if(session=="ASIA" && dir==1) mask=InpNasAsiaLongSoftHoursMask;
    else if(session=="LONDON" && dir==-1) mask=InpNasLondonShortSoftHoursMask;
    else if(session=="NY" && dir==1) mask=InpNasNYLongSoftHoursMask;
    return ((mask>>hour)&1)==1?InpNasClockWeakRisk:1.0;
}
CTrade trade;
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
    if(InpNasClockWeakRisk<=0 || InpNasClockWeakRisk>1 ||
       InpNasAsiaLongSoftHoursMask<0 || InpNasAsiaLongSoftHoursMask>16777215 ||
       InpNasLondonShortSoftHoursMask<0 || InpNasLondonShortSoftHoursMask>16777215 ||
       InpNasNYLongSoftHoursMask<0 || InpNasNYLongSoftHoursMask>16777215 ||
       !InpUseEquityRisk || InpNasAdaptiveRisk || InpPyramidRiskPercent!=0) return INIT_PARAMETERS_INCORRECT;

    if(InpNasQualityMinBodyPct<0 || InpNasQualityMinBodyPct>=100) return INIT_PARAMETERS_INCORRECT;
    if(InpNasMaxBreakoutExtension<=0 || InpNasMaxBreakoutExtension>2) return INIT_PARAMETERS_INCORRECT;

    if(!MQLInfoInteger(MQL_TESTER)) {
        Print("NAS_RESEARCH_ONLY: questa variante si avvia solo nel tester.");
        return INIT_FAILED;
    }
    if(m_symbol!="NAS100") {Print("NAS_RESEARCH_ONLY: simbolo richiesto NAS100");return INIT_PARAMETERS_INCORRECT;}

    if(InpRequireWeekendSessions) {
        datetime from,to;
        if(!SymbolInfoSessionTrade(m_symbol,SATURDAY,0,from,to) || !SymbolInfoSessionTrade(m_symbol,SUNDAY,0,from,to)) {
            Print("CRYPTO_INIT_FAILED: il simbolo non ha sessioni sia sabato sia domenica: ",m_symbol);
            return INIT_PARAMETERS_INCORRECT;
        }
    }
    if(InpNasAdaptiveRisk && (InpNasAdaptiveSoftDDPct<0 || InpNasAdaptiveHardDDPct<=InpNasAdaptiveSoftDDPct || InpNasAdaptiveHardDDPct>=10 || InpNasAdaptiveCurve<=0 || InpNasAdaptiveMinFraction<=0 || InpNasAdaptiveMinFraction>1 || InpNasAdaptiveRoomFraction<=0 || InpNasAdaptiveRoomFraction>=1)) return INIT_PARAMETERS_INCORRECT;
    g_nasAdaptivePeak=AccountInfoDouble(ACCOUNT_EQUITY);
    if(InpNasAsiaLongRisk<=0 || InpNasAsiaShortRisk<=0 || InpNasLondonLongRisk<=0 || InpNasLondonShortRisk<=0 || InpNasNYLongRisk<=0 || InpNasNYShortRisk<=0) return INIT_PARAMETERS_INCORRECT;
    if(InpNasTrailStartR<=0 || InpNasTrailGapR<=0 || InpNasTrailGapR>InpNasTrailStartR || InpNasTrailStepR<=0) return INIT_PARAMETERS_INCORRECT;
    trade.SetExpertMagicNumber(InpMagicNumber);
    trade.SetDeviationInPoints(20);
    trade.SetTypeFillingBySymbol(m_symbol);

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
    Print("Tri-Session Master v5.5 (Londra senza fade) Ready.");
    if(InpNasStopFraction<=0 || InpNasStopFraction>1 || InpNasTrendFast<1 || InpNasTrendSlow<=InpNasTrendFast) return INIT_PARAMETERS_INCORRECT;
    if(InpNasTrendFilter) {
        g_nasTrendFast=iMA(m_symbol,PERIOD_H1,InpNasTrendFast,0,MODE_EMA,PRICE_CLOSE);
        g_nasTrendSlow=iMA(m_symbol,PERIOD_H1,InpNasTrendSlow,0,MODE_EMA,PRICE_CLOSE);
        if(g_nasTrendFast==INVALID_HANDLE || g_nasTrendSlow==INVALID_HANDLE) return INIT_FAILED;
    }
    return(INIT_SUCCEEDED);
}


void OnDeinit(const int reason) {
    Print("NAS_ADAPTIVE_FINAL peak=",DoubleToString(g_nasAdaptivePeak,2)," halted=",g_nasAdaptiveHalted);
    if(g_nasTrendFast!=INVALID_HANDLE) IndicatorRelease(g_nasTrendFast);
    if(g_nasTrendSlow!=INVALID_HANDLE) IndicatorRelease(g_nasTrendSlow);

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
    MqlDateTime now;TimeToStruct(TimeCurrent(),now);
    bool closeDay=InpNasCloseEndOfDay && now.hour*60+now.min>=InpNasCloseHour*60+InpNasCloseMinute;

    int digits = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);

    for(int i=PositionsTotal()-1; i>=0; i--) {
        ulong ticket = PositionGetTicket(i);
        if(ticket == 0) continue;
        if(PositionGetString(POSITION_SYMBOL) != m_symbol) continue;
        if(PositionGetInteger(POSITION_MAGIC) != (long)InpMagicNumber) continue;

        datetime opened=(datetime)PositionGetInteger(POSITION_TIME);
        if(closeDay || (InpNasMaxHoldMinutes>0 && TimeCurrent()-opened>=InpNasMaxHoldMinutes*60)) {
            if(!trade.PositionClose(ticket) || trade.ResultRetcode()!=TRADE_RETCODE_DONE)
                Print("NAS_TIMED_CLOSE_RETRY: ",ticket," retcode ",trade.ResultRetcode());
            continue;
        }
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

         
        if(rMultiple>=InpNasTrailStartR) {
            double lockR=InpNasTrailStartR-InpNasTrailGapR+MathFloor((rMultiple-InpNasTrailStartR)/InpNasTrailStepR)*InpNasTrailStepR;
            double wanted=entry+dir*lockR*slDist;
            double grid=SymbolInfoDouble(m_symbol,SYMBOL_TRADE_TICK_SIZE);
            MqlTick live;
            if(grid>0 && SymbolInfoTick(m_symbol,live)) {
                wanted=NormalizeDouble((dir==1?MathFloor(wanted/grid):MathCeil(wanted/grid))*grid,digits);
                double minGap=MathMax(SymbolInfoInteger(m_symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(m_symbol,SYMBOL_TRADE_FREEZE_LEVEL))*SymbolInfoDouble(m_symbol,SYMBOL_POINT);
                bool improves=dir==1?wanted>currentSL+grid/2:(currentSL==0 || wanted<currentSL-grid/2);
                bool allowed=dir==1?wanted<live.bid-minGap:wanted>live.ask+minGap;
                if(improves && allowed && trade.PositionModify(ticket,wanted,currentTP) && trade.ResultRetcode()==TRADE_RETCODE_DONE) {
                    currentSL=wanted;
                    Print("NAS_TRAIL_UPDATE position=",posId," R=",DoubleToString(rMultiple,4)," lockR=",DoubleToString(lockR,4)," stop=",DoubleToString(wanted,digits));
                }
            }
        }
         
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
            double grid=SymbolInfoDouble(m_symbol,SYMBOL_TRADE_TICK_SIZE);
            if(grid<=0) continue;
            beSL=NormalizeDouble((dir==1?MathFloor(beSL/grid):MathCeil(beSL/grid))*grid,digits);
            MqlTick live; if(!SymbolInfoTick(m_symbol,live)) continue;
            double minGap=MathMax(SymbolInfoInteger(m_symbol,SYMBOL_TRADE_STOPS_LEVEL),SymbolInfoInteger(m_symbol,SYMBOL_TRADE_FREEZE_LEVEL))*SymbolInfoDouble(m_symbol,SYMBOL_POINT);
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


 
void LoadCryptoORB(datetime start,int minutes,bool &loaded,bool &rangeDefined,double &high,double &low) {
    if(loaded || minutes<=0 || TimeCurrent()<start+minutes*60) return;
    MqlRates bars[];
    int n=CopyRates(m_symbol,PERIOD_M1,start,start+minutes*60-1,bars);
    if(n<=0) return;
    double h=-DBL_MAX,l=DBL_MAX;
    for(int i=0;i<n;i++){h=MathMax(h,bars[i].high);l=MathMin(l,bars[i].low);}
    if(h<=l || l<=0) return;
    high=h;low=l;rangeDefined=true;loaded=true;
}


void NasResetEntryCounters() {
    datetime now=TimeCurrent(),day=now-now%86400;
    if(day!=g_nasEntryDay) {
        g_nasEntryDay=day;g_nasLonEntries=0;g_nasNYEntries=0;g_nasAsiaEntries=0;
        g_nasRearmBar=0;g_nasLonFillBar=0;g_nasNYFillBar=0;g_nasAsiaFillBar=0;
    }
}

void NasRearmBreakouts() {
    NasResetEntryCounters();
    if(!InpNasReentry || HasOpenPosition()) return;
    datetime bar=iTime(m_symbol,InpEntryTF,1);
    if(bar<=0 || bar==g_nasRearmBar) return;
    g_nasRearmBar=bar;
    double closed=iClose(m_symbol,InpEntryTF,1);
    if(g_nasLonEntries>0 && g_nasLonEntries<InpNasMaxSessionEntries && bar>g_nasLonFillBar && closed>=g_lonLow && closed<=g_lonHigh) g_lonExecutedToday=false;
    if(g_nasNYEntries>0 && g_nasNYEntries<InpNasMaxSessionEntries && bar>g_nasNYFillBar && closed>=g_nyLow && closed<=g_nyHigh) g_nyExecutedToday=false;
    if(g_nasAsiaEntries>0 && g_nasAsiaEntries<InpNasMaxSessionEntries && bar>g_nasAsiaFillBar && closed>=g_asiaLow && closed<=g_asiaHigh) g_asiaExecutedToday=false;
}


void OnTick() {
    NasAdaptiveObserve();
    if(g_nasAdaptiveHalted) {
         
        for(int i=PositionsTotal()-1;i>=0;i--) {
            ulong ticket=PositionGetTicket(i);
            if(ticket==0 || PositionGetString(POSITION_SYMBOL)!=m_symbol || PositionGetInteger(POSITION_MAGIC)!=(long)InpMagicNumber) continue;
            if(!trade.PositionClose(ticket) || trade.ResultRetcode()!=TRADE_RETCODE_DONE)
                Print("NAS_ADAPTIVE_CLOSE_FAILED ",ticket," retcode=",trade.ResultRetcode());
        }
        return;
    }
    ManageOpenPositions();

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

    if(!InpTradeWeekend && (dt.day_of_week==0 || dt.day_of_week==6)) return;
    LoadCryptoORB(today+InpLon_StartHour*3600+InpLon_StartMin*60,InpLon_DurationMin,g_lonLoaded,g_lonRangeDefined,g_lonHigh,g_lonLow);
    LoadCryptoORB(today+InpNY_StartHour*3600+InpNY_StartMin*60,InpNY_DurationMin,g_nyLoaded,g_nyRangeDefined,g_nyHigh,g_nyLow);
    if(InpUseAsia) LoadCryptoORB(today+InpAsia_StartHour*3600+InpAsia_StartMin*60,InpAsia_DurationMin,g_asiaLoaded,g_asiaRangeDefined,g_asiaHigh,g_asiaLow);
    ManagePyramids();
    NasRearmBreakouts();
    if(g_dayLocked) { if(InpShowComments) Comment("STOP TRADING: Max Losses / Daily Loss Limit Reached"); return; }
    if(g_tradesToday >= InpMaxTradesDay) return;
    if(HasOpenPosition()) return;

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
                    double riskAmt = GetRiskAmount();
                    if(validBody) {
                         
                        if(bull) ExecuteTrade(1, m5_close, g_lonLow, "LONDON", riskAmt, 0);
                        else if(bear) ExecuteTrade(-1, m5_close, g_lonHigh, "LONDON", riskAmt, 0);
                    }
                     
                     
                     
                     
                    if(HasOpenPosition()) g_lonExecutedToday = true;
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
                    if(HasOpenPosition()) g_nyExecutedToday = true;
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
                        if(HasOpenPosition()) g_asiaExecutedToday = true;
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


 
 
 
void ExecuteTrade(int dir, double entry, double sl_ref, string session, double riskAmount, double customTP) {
    double nasUnweightedRisk=riskAmount;
    NasResetEntryCounters();
    if(InpNasReentry && ((session=="LONDON" && g_nasLonEntries>=InpNasMaxSessionEntries) || (session=="NY" && g_nasNYEntries>=InpNasMaxSessionEntries) || (session=="ASIA" && g_nasAsiaEntries>=InpNasMaxSessionEntries))) return;

    MqlDateTime now;TimeToStruct(TimeCurrent(),now);
    if(InpNasCloseEndOfDay && now.hour*60+now.min>=InpNasCloseHour*60+InpNasCloseMinute) return;

    if((dir==1 && !InpNasAllowLong) || (dir==-1 && !InpNasAllowShort)) return;
     
    if(InpNasOnlyNY && session!="NY" && session!="NY_FADE" && session!="NY_REVERSAL") return;

    if(StringFind(session,"ASIA")==0) riskAmount*=InpAsiaRiskMultiplier;
    else if(StringFind(session,"NY")==0) riskAmount*=InpNYRiskMultiplier;
    else if(session=="LONDON") riskAmount*=InpLondonRiskMultiplier;
    riskAmount*=NasDirectionRisk(session,dir);
    double nasClockFactor=NasClockRisk(session,dir,now.hour);
    riskAmount*=nasClockFactor;

    riskAmount=NasAdaptiveBudget(riskAmount);
    if(riskAmount<=0) return;

    double point = SymbolInfoDouble(m_symbol, SYMBOL_POINT);
    int digits = (int)SymbolInfoInteger(m_symbol, SYMBOL_DIGITS);

     
    MqlTick quote;
    if(!SymbolInfoTick(m_symbol,quote)) return;
    entry=dir==1?quote.ask:quote.bid;
    bool qualityApplied=!InpNasQualityStrongOnly || (session=="ASIA" && dir==1) || (session=="LONDON" && dir==-1);
    double qualityHigh=0,qualityLow=0,qualityExtension=0,qualityClose=0,qualityOpen=0;
    double qualityCandleHigh=0,qualityCandleLow=0,qualityBodyPct=0;
    if(qualityApplied) {
        if(session=="ASIA") {qualityHigh=g_asiaHigh;qualityLow=g_asiaLow;}
        else if(session=="LONDON") {qualityHigh=g_lonHigh;qualityLow=g_lonLow;}
        else if(session=="NY") {qualityHigh=g_nyHigh;qualityLow=g_nyLow;}
        else return;
        if(qualityHigh<=qualityLow || qualityLow<=0) return;
        qualityClose=iClose(m_symbol,InpEntryTF,1);qualityOpen=iOpen(m_symbol,InpEntryTF,1);
        if(qualityClose<=0 || qualityOpen<=0) return;
        qualityCandleHigh=iHigh(m_symbol,InpEntryTF,1);qualityCandleLow=iLow(m_symbol,InpEntryTF,1);
        if(qualityCandleHigh<=qualityCandleLow || qualityCandleLow<=0) return;
        qualityBodyPct=100*MathAbs(qualityClose-qualityOpen)/(qualityCandleHigh-qualityCandleLow);
        if(qualityBodyPct<=InpNasQualityMinBodyPct) return;

        if(InpNasQualityCandleDirection && ((dir==1 && qualityClose<=qualityOpen) || (dir==-1 && qualityClose>=qualityOpen))) return;
        double edge=dir==1?qualityHigh:qualityLow;
        qualityExtension=dir*(entry-edge)/(qualityHigh-qualityLow);
        if(qualityExtension<0 || qualityExtension>InpNasMaxBreakoutExtension) return;
    }

    if(InpNasTrendFilter) {
        double fast[1],slow[1];
        if(CopyBuffer(g_nasTrendFast,0,1,1,fast)!=1 || CopyBuffer(g_nasTrendSlow,0,1,1,slow)!=1) return;
        double closedH1=iClose(m_symbol,PERIOD_H1,1);
        if(closedH1<=0 || fast[0]<=0 || slow[0]<=0) return;
        if(dir==1 && !(fast[0]>slow[0] && closedH1>fast[0])) return;
        if(dir==-1 && !(fast[0]<slow[0] && closedH1<fast[0])) return;
    }
    sl_ref=entry+(sl_ref-entry)*InpNasStopFraction;

    double tickSize=SymbolInfoDouble(m_symbol,SYMBOL_TRADE_TICK_SIZE);
    if(tickSize<=0 || riskAmount<=0 || entry<=0) return;
    double sl_final=NormalizeDouble(MathRound(sl_ref/tickSize)*tickSize,digits);
    double minStop=SymbolInfoInteger(m_symbol,SYMBOL_TRADE_STOPS_LEVEL)*point;
    if((dir==1 && (sl_final>=quote.bid-minStop)) || (dir==-1 && sl_final<=quote.ask+minStop)) return;

    double stopDistance=MathAbs(entry-sl_final);
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

    double minLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_MIN);
    double stepLot = SymbolInfoDouble(m_symbol, SYMBOL_VOLUME_STEP);
    double profitPerLot = 0;

    bool calcRes = false;
    if(dir == 1) calcRes = OrderCalcProfit(ORDER_TYPE_BUY, m_symbol, 1.0, entry, sl_final, profitPerLot);
    else calcRes = OrderCalcProfit(ORDER_TYPE_SELL, m_symbol, 1.0, entry, sl_final, profitPerLot);

    if(!calcRes || profitPerLot >= 0) return;

    if(stepLot<=0) return;
    double lots = MathFloor((riskAmount / MathAbs(profitPerLot)) / stepLot) * stepLot;
     
    if(lots < minLot || stepLot<=0) return;
    lots=MathMin(lots,MathFloor(SymbolInfoDouble(m_symbol,SYMBOL_VOLUME_MAX)/stepLot)*stepLot);
    double margin=0;
    if(!OrderCalcMargin(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,m_symbol,lots,entry,margin) || margin<=0) return;
    double availableMargin=AccountInfoDouble(ACCOUNT_MARGIN_FREE)*0.80;
    if(margin>availableMargin) lots=MathFloor(lots*availableMargin/margin/stepLot)*stepLot;
    if(lots<minLot) return;

     
    double nasFullPnl=0;
    bool nasFullRiskValid=false;
    for(int correction=0;correction<4;correction++) {
        if(!OrderCalcProfit(dir==1?ORDER_TYPE_BUY:ORDER_TYPE_SELL,m_symbol,lots,entry,sl_final,nasFullPnl) || nasFullPnl>=0) return;
        if(-nasFullPnl<=riskAmount) {nasFullRiskValid=true;break;}
        lots=MathFloor((lots-stepLot+stepLot*1e-8)/stepLot)*stepLot;
        if(lots<minLot) return;
    }
    if(!nasFullRiskValid) return;

    string cmt = StringFormat("%s|v5.5|R:%.2f", session, riskAmount);
    bool res = false;

    if(dir == 1) res = trade.Buy(lots, m_symbol, 0.0, sl_final, tp_final, cmt);
    else res = trade.Sell(lots, m_symbol, 0.0, sl_final, tp_final, cmt);

    if(res && (trade.ResultRetcode()==TRADE_RETCODE_DONE || trade.ResultRetcode()==TRADE_RETCODE_DONE_PARTIAL)) {
        g_tradesToday++;
        datetime fillBar=iTime(m_symbol,InpEntryTF,0);
        if(session=="LONDON") {g_nasLonEntries++;g_nasLonFillBar=fillBar;}
        else if(session=="NY") {g_nasNYEntries++;g_nasNYFillBar=fillBar;}
        else if(session=="ASIA") {g_nasAsiaEntries++;g_nasAsiaFillBar=fillBar;}


        long posId=0; double filledEntry=entry;
        ulong filledDeal=trade.ResultDeal();
        if(filledDeal>0 && HistoryDealSelect(filledDeal)) {
            posId=HistoryDealGetInteger(filledDeal,DEAL_POSITION_ID);
            filledEntry=HistoryDealGetDouble(filledDeal,DEAL_PRICE);
        }
        if(posId > 0) RegisterPosition(posId, session, filledEntry, MathAbs(filledEntry - sl_final), dir);
        if(qualityApplied) Print("NAS_QUALITY_ACCEPT position=",posId," session=",session," side=",dir,
          " high=",DoubleToString(qualityHigh,8)," low=",DoubleToString(qualityLow,8),
          " quoteEntry=",DoubleToString(entry,8)," extension=",DoubleToString(qualityExtension,12),
          " max=",DoubleToString(InpNasMaxBreakoutExtension,12),
          " closed=",DoubleToString(qualityClose,8)," candleOpen=",DoubleToString(qualityOpen,8),
          " directionRequired=",InpNasQualityCandleDirection);
        if(qualityApplied) Print("NAS_BODY_ACCEPT position=",posId,
          " candleHigh=",DoubleToString(qualityCandleHigh,8)," candleLow=",DoubleToString(qualityCandleLow,8),
          " pct=",DoubleToString(qualityBodyPct,12)," minPct=",DoubleToString(InpNasQualityMinBodyPct,12));



        if(session == "LONDON") g_totalLon++;
        else if(session == "NY") g_totalNY++;
        else if(session == "ASIA") g_totalAsia++;
        else if(session == "NY_FADE") g_totalNYFade++;
        else if(session == "ASIA_FADE") g_totalAsiaFade++;
        else if(session == "NY_REVERSAL") g_totalNYRev++;
        else if(session == "PYRAMID") g_totalPyramid++;

        Print("TRADE OPENED: ", session, " | PosID: ", posId, " | Risk: ", DoubleToString(riskAmount,2), " | TP: ", DoubleToString(tp_final, digits));
        Print("NAS_FULL_VOLUME_RISK position=",posId," session=",session,
          " budget=",DoubleToString(riskAmount,12)," nativeRisk=",DoubleToString(-nasFullPnl,12),
          " volume=",DoubleToString(lots,12)," quote=",DoubleToString(entry,12)," stop=",DoubleToString(sl_final,12));

        Print("NAS_CLOCK_RISK_ACCEPT position=",posId," session=",session," dir=",dir,
          " hour=",now.hour," factor=",DoubleToString(nasClockFactor,12),
          " unweighted=",DoubleToString(nasUnweightedRisk,12)," budget=",DoubleToString(riskAmount,12));

    }
}


 
 
 
void OnTradeTransaction(const MqlTradeTransaction &trans, const MqlTradeRequest &request, const MqlTradeResult &result) {
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
    Print("NAS_EXACT_EQUITY_DD_PCT=",DoubleToString(TesterStatistics(STAT_EQUITY_DDREL_PERCENT),12));
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
    string filename=(MQLInfoString(MQL_PROGRAM_NAME)+"_"+m_symbol)+"_trades.csv";
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
NASEngine(){InpLon_StartHour=NAS_InpLon_StartHour;InpLon_StartMin=NAS_InpLon_StartMin;InpLon_DurationMin=NAS_InpLon_DurationMin;InpLon_UseDailyFilter=NAS_InpLon_UseDailyFilter;InpLon_IgnoreBias=NAS_InpLon_IgnoreBias;InpNY_StartHour=NAS_InpNY_StartHour;InpNY_StartMin=NAS_InpNY_StartMin;InpNY_DurationMin=NAS_InpNY_DurationMin;InpNY_UseDailyFilter=NAS_InpNY_UseDailyFilter;InpUseAsia=NAS_InpUseAsia;InpAsia_StartHour=NAS_InpAsia_StartHour;InpAsia_StartMin=NAS_InpAsia_StartMin;InpAsia_DurationMin=NAS_InpAsia_DurationMin;InpAsia_UseDailyFilter=NAS_InpAsia_UseDailyFilter;InpEntryTF=NAS_InpEntryTF;InpMaxEntryDelayMin=NAS_InpMaxEntryDelayMin;InpBodyFilterPct=NAS_InpBodyFilterPct;InpFadeEnabled=NAS_InpFadeEnabled;InpFadeRiskPct=NAS_InpFadeRiskPct;InpFadeSLBufferPoints=NAS_InpFadeSLBufferPoints;InpATRMinRangeMultAsia=NAS_InpATRMinRangeMultAsia;InpATRMinRangeMultNY=NAS_InpATRMinRangeMultNY;InpUseATRFilter=NAS_InpUseATRFilter;InpATRPeriod=NAS_InpATRPeriod;InpATRMinRangeMult=NAS_InpATRMinRangeMult;InpUseBreakEven=NAS_InpUseBreakEven;InpBreakEvenTriggerR=NAS_InpBreakEvenTriggerR;InpBreakEvenLockR=NAS_InpBreakEvenLockR;InpUsePartialClose=NAS_InpUsePartialClose;InpPartialClosePct=NAS_InpPartialClosePct;InpPartialCloseAtR=NAS_InpPartialCloseAtR;InpFixedRiskAmount=NAS_InpFixedRiskAmount;InpUseEquityRisk=NAS_InpUseEquityRisk;InpRiskPercent=NAS_InpRiskPercent;InpRR_Target=NAS_InpRR_Target;InpMaxLossesDay=NAS_InpMaxLossesDay;InpMaxTradesDay=NAS_InpMaxTradesDay;InpMagicNumber=NAS_InpMagicNumber;InpShowComments=NAS_InpShowComments;InpUseEquityDailyLock=NAS_InpUseEquityDailyLock;InpMaxDailyLossPct=NAS_InpMaxDailyLossPct;InpPyramidAtR=NAS_InpPyramidAtR;InpPyramidRiskPercent=NAS_InpPyramidRiskPercent;InpPyramidExtraTargetR=NAS_InpPyramidExtraTargetR;InpTotalOpenStopRiskPct=NAS_InpTotalOpenStopRiskPct;g_profitPyramid=0;g_winsPyramid=0;g_totalPyramid=0;InpAsiaRiskMultiplier=NAS_InpAsiaRiskMultiplier;InpNYRiskMultiplier=NAS_InpNYRiskMultiplier;InpLondonRiskMultiplier=NAS_InpLondonRiskMultiplier;InpTradeWeekend=NAS_InpTradeWeekend;InpMaxSpreadToStopPct=NAS_InpMaxSpreadToStopPct;InpRequireWeekendSessions=NAS_InpRequireWeekendSessions;g_lonLoaded=false;g_nyLoaded=false;g_asiaLoaded=false;InpNasOnlyNY=NAS_InpNasOnlyNY;InpNasCloseEndOfDay=NAS_InpNasCloseEndOfDay;InpNasCloseHour=NAS_InpNasCloseHour;InpNasCloseMinute=NAS_InpNasCloseMinute;InpNasMaxHoldMinutes=NAS_InpNasMaxHoldMinutes;InpNasAllowLong=NAS_InpNasAllowLong;InpNasAllowShort=NAS_InpNasAllowShort;InpNasTrendFilter=NAS_InpNasTrendFilter;InpNasTrendFast=NAS_InpNasTrendFast;InpNasTrendSlow=NAS_InpNasTrendSlow;InpNasStopFraction=NAS_InpNasStopFraction;g_nasTrendFast=INVALID_HANDLE;g_nasTrendSlow=INVALID_HANDLE;InpNasReentry=NAS_InpNasReentry;InpNasMaxSessionEntries=NAS_InpNasMaxSessionEntries;g_nasLonEntries=0;g_nasNYEntries=0;g_nasAsiaEntries=0;g_nasEntryDay=0;g_nasRearmBar=0;g_nasLonFillBar=0;g_nasNYFillBar=0;g_nasAsiaFillBar=0;InpNasAdaptiveRisk=NAS_InpNasAdaptiveRisk;InpNasAdaptiveSoftDDPct=NAS_InpNasAdaptiveSoftDDPct;InpNasAdaptiveHardDDPct=NAS_InpNasAdaptiveHardDDPct;InpNasAdaptiveCurve=NAS_InpNasAdaptiveCurve;InpNasAdaptiveMinFraction=NAS_InpNasAdaptiveMinFraction;InpNasAdaptiveRoomFraction=NAS_InpNasAdaptiveRoomFraction;g_nasAdaptivePeak=0;g_nasAdaptiveHalted=false;InpNasMaxBreakoutExtension=NAS_InpNasMaxBreakoutExtension;InpNasQualityStrongOnly=NAS_InpNasQualityStrongOnly;InpNasQualityCandleDirection=NAS_InpNasQualityCandleDirection;InpNasQualityMinBodyPct=NAS_InpNasQualityMinBodyPct;InpNasAsiaLongRisk=NAS_InpNasAsiaLongRisk;InpNasAsiaShortRisk=NAS_InpNasAsiaShortRisk;InpNasLondonLongRisk=NAS_InpNasLondonLongRisk;InpNasLondonShortRisk=NAS_InpNasLondonShortRisk;InpNasNYLongRisk=NAS_InpNasNYLongRisk;InpNasNYShortRisk=NAS_InpNasNYShortRisk;InpNasTrailStartR=NAS_InpNasTrailStartR;InpNasTrailGapR=NAS_InpNasTrailGapR;InpNasTrailStepR=NAS_InpNasTrailStepR;InpNasClockWeakRisk=NAS_InpNasClockWeakRisk;InpNasAsiaLongSoftHoursMask=NAS_InpNasAsiaLongSoftHoursMask;InpNasLondonShortSoftHoursMask=NAS_InpNasLondonShortSoftHoursMask;InpNasNYLongSoftHoursMask=NAS_InpNasNYLongSoftHoursMask;g_atrHandle=INVALID_HANDLE;g_lonHigh=0.0;g_lonLow=999999.0;g_lonRangeDefined=false;g_lonExecutedToday=false;g_nyHigh=0.0;g_nyLow=999999.0;g_nyRangeDefined=false;g_nyExecutedToday=false;g_asiaHigh=0.0;g_asiaLow=999999.0;g_asiaRangeDefined=false;g_asiaExecutedToday=false;g_dailyBias=0;InpSwingLen_Daily=5;g_currentDay=0;g_dayStartEquity=0.0;g_tradesToday=0;g_lossesToday=0;g_dayLocked=false;g_profitLon=0;g_profitNY=0;g_profitAsia=0;g_winsLon=0;g_winsNY=0;g_winsAsia=0;g_totalLon=0;g_totalNY=0;g_totalAsia=0;g_profitNYFade=0;g_profitAsiaFade=0;g_winsNYFade=0;g_winsAsiaFade=0;g_totalNYFade=0;g_totalAsiaFade=0;g_profitNYRev=0;g_winsNYRev=0;g_totalNYRev=0;g_nyFadeDir=0;g_nyFadeSL=0;g_nyReArmActive=false;g_nyReArmDeadline=0;}
};

class XAUEngine{
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
XAUEngine(){InpLon_StartHour=XAU_InpLon_StartHour;InpLon_StartMin=XAU_InpLon_StartMin;InpLon_DurationMin=XAU_InpLon_DurationMin;InpLon_UseDailyFilter=XAU_InpLon_UseDailyFilter;InpLon_IgnoreBias=XAU_InpLon_IgnoreBias;InpNY_StartHour=XAU_InpNY_StartHour;InpNY_StartMin=XAU_InpNY_StartMin;InpNY_DurationMin=XAU_InpNY_DurationMin;InpNY_UseDailyFilter=XAU_InpNY_UseDailyFilter;InpUseAsia=XAU_InpUseAsia;InpAsia_StartHour=XAU_InpAsia_StartHour;InpAsia_StartMin=XAU_InpAsia_StartMin;InpAsia_DurationMin=XAU_InpAsia_DurationMin;InpAsia_UseDailyFilter=XAU_InpAsia_UseDailyFilter;InpEntryTF=XAU_InpEntryTF;InpMaxEntryDelayMin=XAU_InpMaxEntryDelayMin;InpBodyFilterPct=XAU_InpBodyFilterPct;InpFadeEnabled=XAU_InpFadeEnabled;InpFadeRiskPct=XAU_InpFadeRiskPct;InpFadeSLBufferPoints=XAU_InpFadeSLBufferPoints;InpATRMinRangeMultAsia=XAU_InpATRMinRangeMultAsia;InpATRMinRangeMultNY=XAU_InpATRMinRangeMultNY;InpUseATRFilter=XAU_InpUseATRFilter;InpATRPeriod=XAU_InpATRPeriod;InpATRMinRangeMult=XAU_InpATRMinRangeMult;InpUseBreakEven=XAU_InpUseBreakEven;InpBreakEvenTriggerR=XAU_InpBreakEvenTriggerR;InpBreakEvenLockR=XAU_InpBreakEvenLockR;InpUsePartialClose=XAU_InpUsePartialClose;InpPartialClosePct=XAU_InpPartialClosePct;InpPartialCloseAtR=XAU_InpPartialCloseAtR;InpFixedRiskAmount=XAU_InpFixedRiskAmount;InpUseEquityRisk=XAU_InpUseEquityRisk;InpRiskPercent=XAU_InpRiskPercent;InpRR_Target=XAU_InpRR_Target;InpMaxLossesDay=XAU_InpMaxLossesDay;InpMaxTradesDay=XAU_InpMaxTradesDay;InpMagicNumber=XAU_InpMagicNumber;InpShowComments=XAU_InpShowComments;InpUseEquityDailyLock=XAU_InpUseEquityDailyLock;InpMaxDailyLossPct=XAU_InpMaxDailyLossPct;InpPyramidAtR=XAU_InpPyramidAtR;InpPyramidRiskPercent=XAU_InpPyramidRiskPercent;InpPyramidExtraTargetR=XAU_InpPyramidExtraTargetR;InpTotalOpenStopRiskPct=XAU_InpTotalOpenStopRiskPct;g_profitPyramid=0;g_winsPyramid=0;g_totalPyramid=0;InpAsiaRiskMultiplier=XAU_InpAsiaRiskMultiplier;InpNYRiskMultiplier=XAU_InpNYRiskMultiplier;InpLondonRiskMultiplier=XAU_InpLondonRiskMultiplier;InpFTMOInitialCapital=XAU_InpFTMOInitialCapital;InpFTMOTargetPercent=XAU_InpFTMOTargetPercent;InpFTMOStopAtTarget=XAU_InpFTMOStopAtTarget;InpFTMOMinTradingDays=XAU_InpFTMOMinTradingDays;InpFTMODailyLossEUR=XAU_InpFTMODailyLossEUR;InpFTMOTotalLossEUR=XAU_InpFTMOTotalLossEUR;InpFTMOBufferEUR=XAU_InpFTMOBufferEUR;InpFTMOReservePerTradeEUR=XAU_InpFTMOReservePerTradeEUR;InpFTMOQualificationRiskPct=XAU_InpFTMOQualificationRiskPct;InpFTMOServerUTCWinter=XAU_InpFTMOServerUTCWinter;InpFTMOServerUTCSummer=XAU_InpFTMOServerUTCSummer;InpFTMOServerDST=XAU_InpFTMOServerDST;InpFTMOBrokerClockConfirmed=XAU_InpFTMOBrokerClockConfirmed;InpResearchMonth=XAU_InpResearchMonth;InpResearchStopServer=XAU_InpResearchStopServer;g_ftmoReady=false;g_ftmoFailed=false;g_ftmoPassed=false;g_ftmoTotalStopped=false;g_ftmoDay=0;g_ftmoLockedDay=0;g_ftmoMonth=0;g_ftmoMidnight=10000;g_ftmoMinEquity=10000;g_ftmoMinBalance=10000;g_ftmoWorstDaily=0;g_ftmoStartCET=0;g_ftmoEndCET=0;g_ftmoPassedServer=0;g_ftmoFirstEntry=0;g_ftmoFailureReason="";InpMaximumPositions=XAU_InpMaximumPositions;InpSessionMaxEntries=XAU_InpSessionMaxEntries;InpReentryCooldownMin=XAU_InpReentryCooldownMin;g_lonEntries=0;g_nyEntries=0;g_asiaEntries=0;g_lonLastEntry=0;g_nyLastEntry=0;g_asiaLastEntry=0;g_auditIndex=-1;InpAllowLong=XAU_InpAllowLong;InpAllowShort=XAU_InpAllowShort;InpCloseFriday=XAU_InpCloseFriday;InpRolloverCheckServerHour=XAU_InpRolloverCheckServerHour;InpRolloverCheckServerMinute=XAU_InpRolloverCheckServerMinute;InpDailyDirectionPeriod=XAU_InpDailyDirectionPeriod;InpDailyFilterLong=XAU_InpDailyFilterLong;InpDailyFilterShort=XAU_InpDailyFilterShort;g_dailyDirectionMA=INVALID_HANDLE;InpBelowInitialRiskPercent=XAU_InpBelowInitialRiskPercent;InpOvernightOnlyProtected=XAU_InpOvernightOnlyProtected;InpDailyFlat=XAU_InpDailyFlat;g_ftmoElapsedStartCET=0;g_lonLoaded=false;g_nyLoaded=false;g_asiaLoaded=false;g_atrHandle=INVALID_HANDLE;g_lonHigh=0.0;g_lonLow=999999.0;g_lonRangeDefined=false;g_lonExecutedToday=false;g_nyHigh=0.0;g_nyLow=999999.0;g_nyRangeDefined=false;g_nyExecutedToday=false;g_asiaHigh=0.0;g_asiaLow=999999.0;g_asiaRangeDefined=false;g_asiaExecutedToday=false;g_dailyBias=0;InpSwingLen_Daily=5;g_currentDay=0;g_dayStartEquity=0.0;g_tradesToday=0;g_lossesToday=0;g_dayLocked=false;g_profitLon=0;g_profitNY=0;g_profitAsia=0;g_winsLon=0;g_winsNY=0;g_winsAsia=0;g_totalLon=0;g_totalNY=0;g_totalAsia=0;g_profitNYFade=0;g_profitAsiaFade=0;g_winsNYFade=0;g_winsAsiaFade=0;g_totalNYFade=0;g_totalAsiaFade=0;g_profitNYRev=0;g_winsNYRev=0;g_totalNYRev=0;g_nyFadeDir=0;g_nyFadeSL=0;g_nyReArmActive=false;g_nyReArmDeadline=0;}
};

NASEngine nas;
XAUEngine xau;
bool g_nasReady=false,g_xauReady=false;
input bool InpCombinedEnableNAS=true;
input bool InpCombinedEnableXAU=false;
input string InpCombinedNASsymbol="NAS100";
input string InpCombinedXAUsymbol="XAUUSD";
bool CombinedOwnDeal(ulong d){
 string s=HistoryDealGetString(d,DEAL_SYMBOL);long magic=HistoryDealGetInteger(d,DEAL_MAGIC);
 return (s==InpCombinedNASsymbol && magic==(long)NAS_InpMagicNumber) ||
        (s==InpCombinedXAUsymbol && magic==(long)XAU_InpMagicNumber);
}
int OnInit(){
 if(!MQLInfoInteger(MQL_TESTER)){Print("COMBINED_TESTER_ONLY: nessuna attivazione e nessun ordine sul conto.");return INIT_FAILED;}
 if(!InpCombinedEnableNAS && !InpCombinedEnableXAU)return INIT_PARAMETERS_INCORRECT;
 if(InpCombinedNASsymbol!="NAS100" || InpCombinedXAUsymbol!="XAUUSD" || NAS_InpMagicNumber==XAU_InpMagicNumber)return INIT_PARAMETERS_INCORRECT;
 if(AccountInfoInteger(ACCOUNT_MARGIN_MODE)!=ACCOUNT_MARGIN_MODE_RETAIL_HEDGING)return INIT_PARAMETERS_INCORRECT;
 if(InpCombinedEnableNAS){
  if(!SymbolSelect(InpCombinedNASsymbol,true))return INIT_FAILED;
  nas.m_symbol=InpCombinedNASsymbol;g_nasReady=true;
  int result=nas.OnInit();if(result!=INIT_SUCCEEDED)return result;
 }
 if(InpCombinedEnableXAU){
  if(!SymbolSelect(InpCombinedXAUsymbol,true))return INIT_FAILED;
  xau.m_symbol=InpCombinedXAUsymbol;g_xauReady=true;
  int result=xau.OnInit();if(result!=INIT_SUCCEEDED)return result;
 }
 if((g_nasReady && InpCombinedNASsymbol!=_Symbol) || (g_xauReady && InpCombinedXAUsymbol!=_Symbol))EventSetTimer(5);
 Print("COMBINED_INIT: NAS1211 + XAU183; parametri e controlli originali, conto condiviso.");
 return INIT_SUCCEEDED;
}
MqlTick g_lastNasTick,g_lastXauTick;
bool CombinedNewQuote(string symbol,MqlTick &previous){
 MqlTick q;if(!SymbolInfoTick(symbol,q) || q.bid<=0 || q.ask<q.bid)return false;
 bool changed=q.time_msc!=previous.time_msc || q.bid!=previous.bid || q.ask!=previous.ask;
 if(changed)previous=q;return changed;
}
void CombinedRun(bool fromTimer){
 if(!MQLInfoInteger(MQL_TESTER))return;
 if(g_nasReady && (InpCombinedNASsymbol==_Symbol ? !fromTimer : CombinedNewQuote(InpCombinedNASsymbol,g_lastNasTick)))nas.OnTick();
 if(g_xauReady && (InpCombinedXAUsymbol==_Symbol ? !fromTimer : CombinedNewQuote(InpCombinedXAUsymbol,g_lastXauTick)))xau.OnTick();
}
void OnTick(){CombinedRun(false);}
void OnTimer(){CombinedRun(true);}
void OnTradeTransaction(const MqlTradeTransaction &t,const MqlTradeRequest &r,const MqlTradeResult &s){
 if(g_nasReady)nas.OnTradeTransaction(t,r,s);
 if(g_xauReady)xau.OnTradeTransaction(t,r,s);
}
void OnDeinit(const int reason){
 EventKillTimer();
 if(g_nasReady)nas.OnDeinit(reason);
 if(g_xauReady)xau.OnDeinit(reason);
}
void CombinedExportDeals(){
 if(!MQLInfoInteger(MQL_TESTER) || !HistorySelect(0,TimeCurrent()))return;
 int h=FileOpen(MQLInfoString(MQL_PROGRAM_NAME)+"_combined_deals.csv",FILE_COMMON|FILE_WRITE|FILE_CSV|FILE_ANSI,';',CP_UTF8);
 if(h==INVALID_HANDLE)return;
 FileWrite(h,"deal_id","position_id","time","symbol","magic","type","entry","volume","price","commission","swap","fee","profit");
 for(int i=0;i<HistoryDealsTotal();i++){
  ulong d=HistoryDealGetTicket(i);long t=HistoryDealGetInteger(d,DEAL_TYPE);
  if(t!=DEAL_TYPE_BUY && t!=DEAL_TYPE_SELL)continue;
  FileWrite(h,d,HistoryDealGetInteger(d,DEAL_POSITION_ID),TimeToString((datetime)HistoryDealGetInteger(d,DEAL_TIME),TIME_DATE|TIME_SECONDS),HistoryDealGetString(d,DEAL_SYMBOL),HistoryDealGetInteger(d,DEAL_MAGIC),t,HistoryDealGetInteger(d,DEAL_ENTRY),
   DoubleToString(HistoryDealGetDouble(d,DEAL_VOLUME),4),DoubleToString(HistoryDealGetDouble(d,DEAL_PRICE),5),DoubleToString(HistoryDealGetDouble(d,DEAL_COMMISSION),2),DoubleToString(HistoryDealGetDouble(d,DEAL_SWAP),2),DoubleToString(HistoryDealGetDouble(d,DEAL_FEE),2),DoubleToString(HistoryDealGetDouble(d,DEAL_PROFIT),2));
 }
 FileClose(h);
}
double OnTester() {
 if(g_nasReady)nas.OnTester(); if(g_xauReady)xau.OnTester(); CombinedExportDeals();
    Print("NAS_EXACT_EQUITY_DD_PCT=",DoubleToString(TesterStatistics(STAT_EQUITY_DDREL_PERCENT),12));
    if(!HistorySelect(0,TimeCurrent())) return TesterStatistics(STAT_PROFIT);
    long ids[];
    for(int i=0;i<HistoryDealsTotal();i++) {
        ulong deal=HistoryDealGetTicket(i);
        if(!CombinedOwnDeal(deal)) continue;
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
