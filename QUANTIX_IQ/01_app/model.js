/* QUANTIX IQ: modello della demo locale. Nessun collegamento a MT5 o a un broker. */
(function (global) {
  'use strict';
  const round = n => Math.round((n + Number.EPSILON) * 100) / 100;
  const DEFAULTS = { name: 'Marco', capital: 10000, dailyLimit: 450, buffer: 75, riskPercent: 2.5, notifications: true };
  const SCENARIOS = [
    { id: 'wait', title: 'Attesa del breakout', text: 'Range valido. Manca una chiusura M5 oltre il livello.', icon: 'clock-3' },
    { id: 'long', title: 'Ingresso valido', text: 'Si apre una posizione interamente simulata, con SL e TP.', icon: 'trending-up' },
    { id: 'atr', title: 'Filtro ATR in attesa', text: 'Range stretto. Il bot continua a valutare il breakout successivo.', icon: 'activity' },
    { id: 'short', title: 'Vendita filtrata', text: 'Breakout ribassista, ma il filtro giornaliero blocca lo short.', icon: 'filter' },
    { id: 'risk', title: 'Protezione giornaliera', text: 'La perdita simulata raggiunge il margine di protezione.', icon: 'shield-alert' },
    { id: 'offline', title: 'Connessione assente', text: 'Mostra la differenza tra attesa di un segnale e dati non aggiornati.', icon: 'wifi-off' }
  ];
  const SEED = [68,-35,22,-65,44,-34,55,-27,18,-41,73,-52,31,-49,62,-38,29,-56,47,-25,14,-63,52,-43,37,-21];
  SEED.push(round(-SEED.reduce((a,b)=>a+b,0)));
  function create() {
    const s = { version: 1, view: 'home', theme: 'dark', scenario: 'wait', minute: 15*60+10,
      settings: { ...DEFAULTS }, connected: true, paused: false, showLevels: true, timeframe: 'M5', period: '30d', filter: 'all', search: '',
      high: 2336, low: 2333, price: 2334.50, previous: 2334.20, atr: 16, todayClosed: 84.50, position: null,
      events: [], messages: [], serial: 0, closedTrades: [] };
    addEvent(s, 'system', 'Sessione dimostrativa caricata', 'Dati locali di esempio. Nessun conto collegato.', '14:30');
    addEvent(s, 'decision', 'Range New York definito', 'Massimo 2.336,00; minimo 2.333,00. Orario italiano dello scenario.', '15:00');
    addEvent(s, 'decision', 'Filtro di volatilità superato', 'Ampiezza del range 3,00; soglia ATR 2,40. Dati dimostrativi.', '15:05');
    addEvent(s, 'trade', 'Operazione precedente chiusa', 'Risultato dimostrativo realizzato oggi: +84,50 EUR.', '15:05');
    s.messages.push({ role: 'bot', text: 'Sto monitorando XAUUSD nella demo della prova298. Il range è valido: aspetto una chiusura M5 sopra 2.336,00 o sotto 2.333,00, con gli altri filtri soddisfatti.', time: time(s) });
    return s;
  }
  function time(s) { return `${String(Math.floor(s.minute/60)%24).padStart(2,'0')}:${String(s.minute%60).padStart(2,'0')}`; }
  function addEvent(s, type, title, detail, at) {
    s.events.unshift({ id: ++s.serial, type, title, detail, time: at || time(s), snapshot: { price: s.price, high: s.high, low: s.low, scenario: s.scenario } });
    s.events = s.events.slice(0,60);
  }
  function history(s) {
    const days = SEED.map((p,i) => ({ date: `2026-09-${String(19+i).padStart(2,'0')}`, pnl: p }));
    // Date reali e ordinabili, fino al giorno fittizio dello scenario.
    return days.map((d,i)=>({date:new Date(Date.UTC(2026,8,19+i)).toISOString().slice(0,10),pnl:d.pnl}))
      .concat([{date:'2026-10-16',pnl:s.todayClosed}]);
  }
  function account(s) {
    const p = s.position;
    const floating = p ? round((s.price-p.entry)*p.volume*100*p.dir) : 0;
    const balance = round(s.settings.capital + s.todayClosed);
    const equity = round(balance + floating);
    const dayPnl = round(s.todayClosed + floating);
    const dailyFloor = s.settings.capital-s.settings.dailyLimit;
    const totalFloor = s.settings.capital*0.9;
    const protection = Math.max(dailyFloor,totalFloor)+s.settings.buffer;
    const remaining = round(Math.max(0,equity-protection));
    const stopRisk = p ? round(Math.max(0,(p.entry-p.sl)*p.volume*100*p.dir)) : 0;
    const currentToStop = p ? round(Math.max(0,(s.price-p.sl)*p.volume*100*p.dir)) : 0;
    return { balance,equity,floating,dayPnl,dailyFloor,totalFloor,protection,remaining,stopRisk,currentToStop,capacity:round(Math.max(0,remaining-currentToStop-5)) };
  }
  function decision(s) {
    const a=account(s), width=round(s.high-s.low), required=round(s.atr*0.15);
    const long=s.price>s.high, short=s.price<s.low, outside=long||short;
    const extension=long?s.price-s.high:short?s.low-s.price:0;
    const movement=long?s.price-s.low:short?s.high-s.price:width;
    const deferred=width<required && outside && movement>=required && extension<=width;
    const atrOk=width>=required || deferred;
    const dailyOk=!(s.scenario==='short' && short);
    const bodyOk=Math.abs(s.price-s.previous)>0.05;
    const end=s.minute>=16*60;
    const checks=[
      {label:'Range della sessione',ok:true,value:`${width.toFixed(2)} punti`,why:`Massimo ${s.high.toFixed(2)}, minimo ${s.low.toFixed(2)}.`},
      {label:'Filtro di volatilità',ok:atrOk,value:atrOk?(deferred?'Raggiunto dopo il breakout':'Superato'):'In attesa',why:`Soglia: 15% di ATR D1 = ${required.toFixed(2)}. Movimento osservato: ${movement.toFixed(2)}.`},
      {label:'Chiusura M5 oltre il range',ok:outside,value:outside?(long?'Sopra il massimo':'Sotto il minimo'):'In attesa',why:'Il segnale usa una candela chiusa. Il semplice tocco del livello non basta.'},
      {label:'Corpo della candela',ok:bodyOk,value:bodyOk?'Direzionale, oltre il 25%':'Non valido',why:'Nella demo viene rappresentata una candela coerente con il movimento.'},
      {label:'Filtro giornaliero',ok:dailyOk,value:dailyOk?'Compatibile':'Short bloccato',why:dailyOk?'La direzione è ammessa nello scenario.':'La chiusura D1 non è sotto la EMA20: lo short viene scartato.'},
      {label:'Budget di rischio',ok:a.remaining>5,value:a.remaining>5?'Disponibile':'Protezione attiva',why:`Equity ${a.equity.toFixed(2)}; protezione ${a.protection.toFixed(2)}.`}
    ];
    let kind='waiting',title='In attesa di un segnale',reason=`Aspetto una chiusura M5 sopra ${s.high.toFixed(2)} o sotto ${s.low.toFixed(2)}, con i filtri soddisfatti.`;
    if(!s.connected){kind='offline';title='Dati non aggiornati';reason='Il collegamento simulato è interrotto. Questi sono gli ultimi dati disponibili.';}
    else if(s.paused){kind='paused';title='Simulazione in pausa';reason='Hai messo in pausa la demo. I valori rimangono fermi fino alla ripresa.';}
    else if(a.equity<=a.protection){kind='blocked';title='Protezione giornaliera attiva';reason=`Equity ${a.equity.toFixed(2)} sotto la protezione ${a.protection.toFixed(2)}. Nessun nuovo ingresso nella demo.`;}
    else if(end){kind='blocked';title='Sessione del venerdì conclusa';reason='Nello scenario sono le16:00 italiane. La prova298 chiude le proprie posizioni e termina gli ingressi del venerdì.';}
    else if(s.position){kind='position';title='Gestione della posizione';reason=s.position.be?'Lo stop è stato protetto. Seguo la posizione con target e trailing.':'Posizione simulata aperta. Controllo lo stop e il raggiungimento di +1R.';}
    else if(!atrOk){reason=`Il range iniziale è stretto: ${width.toFixed(2)} contro una soglia ${required.toFixed(2)}. Continuo ad analizzare il breakout successivo, mantenendo la stessa soglia ATR.`;}
    else if(!dailyOk){reason='Il breakout ribassista è presente, ma il filtro giornaliero scarta la vendita. Il prezzo da solo non autorizza l’ingresso.';}
    else if(outside&&bodyOk){kind='ready';title='Condizioni di ingresso soddisfatte';reason=`Il breakout ${long?'rialzista':'ribassista'} soddisfa le condizioni dello scenario. Il rischio rimane soggetto al budget disponibile.`;}
    return {kind,title,reason,checks,atrOk,deferred,long,short,canEnter:kind==='ready'};
  }
  function openPosition(s, deferred) {
    if(s.position)return;
    const d=decision(s);if(!d.canEnter)return;
    const dir=d.long?1:-1,entry=round(s.price+dir*0.02),sl=dir===1?s.low:s.high,dist=Math.abs(entry-sl);
    const requested=account(s).equity*s.settings.riskPercent/100*0.8*(deferred?0.1:1);
    const capacity=account(s).capacity;
    const volume=Math.floor(Math.min(requested,capacity)/(dist*100)*100)/100;
    if(volume<0.01){addEvent(s,'protection','Volume insufficiente','Il lotto minimo simulato supererebbe il budget disponibile.');return;}
    s.position={id:++s.serial,dir,entry,sl:round(sl),initialSL:round(sl),r:round(dist),tp:round(entry+dir*dist*6),volume,risk:round(dist*volume*100),be:false,trail:false,opened:time(s),deferred:!!deferred};
    addEvent(s,'trade',`Ingresso ${dir===1?'BUY':'SELL'} simulato`,`${volume.toFixed(2)} lotti demo. Stop ${sl.toFixed(2)}, target ${s.position.tp.toFixed(2)}. Rischio ${s.position.risk.toFixed(2)} EUR.`);
    s.messages.push({role:'bot',text:`Ho aperto una posizione ${dir===1?'BUY':'SELL'} nella simulazione: candela fuori dal range, filtri soddisfatti e budget disponibile. Rischio iniziale ${s.position.risk.toFixed(2)} EUR. Nessun ordine è stato inviato a MT5.`,time:time(s)});
  }
  function closePosition(s,reason,price) {
    const p=s.position;if(!p)return;
    const exit=price==null?s.price:price;
    const net=round((exit-p.entry)*p.volume*100*p.dir);
    s.todayClosed=round(s.todayClosed+net);
    s.closedTrades.unshift({...p,exit,net,reason,closed:time(s)});
    s.position=null;s.price=round(exit);
    addEvent(s,'trade','Posizione simulata chiusa',`${reason}. Risultato ${net>=0?'+':''}${net.toFixed(2)} EUR.`);
    s.messages.push({role:'bot',text:`Posizione chiusa nella demo: ${reason}. Risultato realizzato ${net.toFixed(2)} EUR.`,time:time(s)});
  }
  function manage(s) {
    const p=s.position;if(!p)return;
    if((p.dir===1&&s.price<=p.sl)||(p.dir===-1&&s.price>=p.sl)){closePosition(s,'Stop loss',p.sl);return;}
    if((p.dir===1&&s.price>=p.tp)||(p.dir===-1&&s.price<=p.tp)){closePosition(s,'Take profit',p.tp);return;}
    const achieved=(s.price-p.entry)*p.dir/p.r;
    if(achieved+1e-8>=1&&!p.be){p.sl=round(p.entry+p.dir*p.r*0.1);p.be=true;addEvent(s,'protection','Stop protetto a +0,1R','La posizione ha raggiunto +1R. Lo stop viene spostato oltre il prezzo di entrata.');}
    if(achieved+1e-8>=2.5){const next=round(s.price-p.dir*p.r*1.25);if(p.dir*(next-p.sl)>0){p.sl=next;p.trail=true;addEvent(s,'protection','Trailing aggiornato',`Distanza del trailing 1,25R. Nuovo stop ${p.sl.toFixed(2)}.`);}}
  }
  function scenario(s,id) {
    if(!SCENARIOS.some(x=>x.id===id))return s;
    s.scenario=id;s.minute=15*60+10;s.position=null;s.closedTrades=[];s.todayClosed=id==='risk'?-400:84.50;s.price=2334.50;s.previous=2334.20;s.high=2336;s.low=2333;s.connected=id!=='offline';s.paused=false;
    if(id==='atr'){s.high=2335;s.low=2333.50;s.price=2335.20;s.previous=2334.95;}
    if(id==='short'){s.price=2332.70;s.previous=2333.10;}
    if(id==='long'){s.price=2336.40;s.previous=2335.50;openPosition(s,false);s.price=2337.12;}
    const d=decision(s);
    addEvent(s,id==='risk'?'protection':'decision',SCENARIOS.find(x=>x.id===id).title,d.reason);
    s.messages.push({role:'bot',text:`Scenario “${SCENARIOS.find(x=>x.id===id).title}”. ${d.reason}`,time:time(s)});
    return s;
  }
  function advance(s) {
    if(!s.connected||s.paused)return false;
    if(account(s).equity<=account(s).protection||s.minute>=16*60)return false;
    s.minute+=5;s.previous=s.price;s.price=round(s.price+(s.position?s.position.r*0.4:0.25));
    if(s.minute>=16*60){if(s.position)closePosition(s,'Chiusura del venerdì alle16:00 italiane');addEvent(s,'protection','Fine operatività del venerdì','Nella prova298 la chiusura resta quella originale.');return true;}
    manage(s);
    const d=decision(s);
    if(d.canEnter)openPosition(s,d.deferred);
    else if(!s.position)addEvent(s,'decision','Nuova candela M5 valutata',d.reason);
    return true;
  }
  function simulatePosition(s,action) {
    const p=s.position;if(!p)return;
    s.previous=s.price;
    if(action==='one')s.price=round(p.entry+p.dir*p.r);
    if(action==='trail')s.price=round(p.entry+p.dir*p.r*2.5);
    if(action==='tp')s.price=p.tp;
    if(action==='sl')s.price=p.sl;
    if(action==='close'){closePosition(s,'Chiusura manuale nella demo');return;}
    manage(s);
  }
  function stats(s,period) {
    let daily=history(s);
    if(period==='today')daily=daily.slice(-1);
    if(period==='7d')daily=daily.slice(-7);
    const trades=daily.map((d,i)=>({id:i+1,date:d.date,net:d.pnl})).filter(t=>t.net!==0);
    let balance=s.settings.capital,peak=balance,maxDD=0;
    const points=[{date:daily[0]?.date||'2026-10-16',balance}];
    for(const d of daily){balance=round(balance+d.pnl);peak=Math.max(peak,balance);maxDD=Math.max(maxDD,(peak-balance)/peak*100);points.push({date:d.date,balance});}
    const positive=trades.filter(t=>t.net>0),negative=trades.filter(t=>t.net<0);
    const gross=positive.reduce((a,t)=>a+t.net,0),loss=-negative.reduce((a,t)=>a+t.net,0);
    const net=round(daily.reduce((a,d)=>a+d.pnl,0));
    return {daily,trades,points,net,count:trades.length,winRate:trades.length?positive.length/trades.length*100:0,pf:loss?gross/loss:null,payoff:trades.length?net/trades.length:0,dd:maxDD};
  }
  function answer(s,question) {
    const q=question.toLowerCase().normalize('NFD').replace(/[\u0300-\u036f]/g,'');const a=account(s),d=decision(s),p=s.position;
    if(/(^|\W)(ciao|buongiorno|salve)(\W|$)/.test(q))return `Ciao ${s.settings.name}! Posso spiegarti segnali, filtri, rischio, stop e target dello scenario. ${d.title}.`;
    if(/risch|perdit|prote|equity|saldo|drawdown|budget|margine/.test(q))return `Saldo ${a.balance.toFixed(2)} EUR; equity ${a.equity.toFixed(2)} EUR. Il risultato di oggi, incluse le posizioni aperte, è ${a.dayPnl.toFixed(2)} EUR. La protezione giornaliera interviene a ${a.protection.toFixed(2)} EUR: margine residuo ${a.remaining.toFixed(2)} EUR. Il rischio iniziale ancora esposto allo stop è ${a.stopRisk.toFixed(2)} EUR. DD dal picco e calo dal capitale iniziale sono misure differenti: qui il calo sotto il capitale iniziale è ${Math.max(0,s.settings.capital-a.equity).toFixed(2)} EUR.`;
    if(/stop|target|take|\bsl\b|\btp\b|break.even|trailing/.test(q))return p?`La posizione demo entra a ${p.entry.toFixed(2)}, ha stop attuale ${p.sl.toFixed(2)} e target ${p.tp.toFixed(2)}. Il target iniziale è6R. A +1R sposto lo stop a +0,1R; da +2,5R applico il trailing con distanza1,25R. ${p.be?'Lo stop è già protetto.':'Lo stop non ha ancora raggiunto il break-even.'}`:`Non ci sono posizioni aperte. Per un BUY lo stop iniziale è il minimo del range (${s.low.toFixed(2)}), per un SELL il massimo (${s.high.toFixed(2)}). Il target è6 volte la distanza tra entrata effettiva e stop. Puoi scegliere “Ingresso valido” per provare questa gestione.`;
    if(/atr|volatil|filtro/.test(q))return `Il filtro usa il15% dell’ATR D1 a14 periodi: nello scenario la soglia è ${(s.atr*.15).toFixed(2)} punti. Il range misura ${(s.high-s.low).toFixed(2)}. ${d.atrOk?'La condizione ATR è soddisfatta.':'Continuo a valutare se il movimento successivo raggiunge la stessa soglia, entro i limiti di estensione.'} ${s.scenario==='short'?'Il filtro giornaliero, però, blocca la vendita.':''}`;
    if(/venerd|orari|session|quando.*chiud|weekend/.test(q))return 'La sessione New York della demo forma il range dalle14:30 alle15:00 italiane. Il segnale usa le chiusure M5. La prova298 conserva la chiusura del venerdì alle16:00 italiane: in quel momento chiude anche le proprie posizioni. Questa anteprima non scarica prezzi né consulta un mercato reale.';
    if(/strateg|funzion|regol|version|prova298/.test(q))return 'Il riferimento è la prova298: breakout dei range di sessione, soglia ATR, candela M5 direzionale con corpo oltre25%, filtri giornalieri e limiti di rischio. Target6R, protezione a+1R e trailing da+2,5R. La ORB extra richiede saldo ed equity almeno20.000 nella valuta del conto. Qui stai provando un modello illustrativo di una sessione, non l’esecuzione completa dell’EA.';
    if(/entr|segnal|aspett|perche|decision|motiv|compr|vend/.test(q))return `${d.reason} ${d.checks.filter(c=>!c.ok).map(c=>`${c.label}: ${c.value}. ${c.why}`).join(' ')} ${d.kind==='position'?'La posizione aperta è soltanto simulata.':''}`;
    if(/reale|demo|colleg|mt5|broker|internet|intelligen|\bai\b/.test(q))return 'Questa è una demo interattiva locale. Prezzi, connessione e operazioni sono simulati. Le risposte seguono regole locali e leggono lo scenario: non c’è ancora un modello AI collegato, né un ponte verso MT5. Nessuna operazione reale o demo viene inviata al broker.';
    return 'In questa anteprima rispondo alle domande su ingressi, filtri ATR, rischio, stop, target e orari dello scenario. Non ho dati esterni per rispondere a questa domanda. Prova “Perché non entri?” oppure “Quanto rischio oggi?”.';
  }
  function validateSettings(x) {
    if(!String(x.name||'').trim())return 'Inserisci il nome da mostrare.';
    if(!Number.isFinite(+x.capital)||+x.capital<100)return 'Il capitale dimostrativo deve essere almeno100 EUR.';
    if(!Number.isFinite(+x.dailyLimit)||+x.dailyLimit<=0||+x.dailyLimit>+x.capital*.1)return 'Il limite giornaliero deve essere positivo e non superare il10% del capitale.';
    if(!Number.isFinite(+x.buffer)||+x.buffer<0||+x.buffer>=+x.dailyLimit)return 'Il cuscinetto deve essere positivo o zero e inferiore al limite giornaliero.';
    if(!Number.isFinite(+x.riskPercent)||+x.riskPercent<=0||+x.riskPercent>5)return 'Il rischio base deve essere maggiore di zero e non superiore al5%.';
    return null;
  }
  global.QuantixModel={create,account,decision,scenario,advance,simulatePosition,stats,answer,addEvent,time,validateSettings,history,round,SCENARIOS,DEFAULTS};
})(typeof window!=='undefined'?window:globalThis);
