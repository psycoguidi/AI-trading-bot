/* Calendario, diario e profilo personali, conservati sul PC e protetti dall'accesso privato. */
(function(){
  'use strict';
  const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const num=v=>v==null?'—':new Intl.NumberFormat('it-IT',{maximumFractionDigits:2}).format(v);
  const today=()=>new Intl.DateTimeFormat('sv-SE',{timeZone:'Europe/Rome',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());
  const monthName=m=>new Date(m+'-15T12:00:00').toLocaleDateString('it-IT',{month:'long',year:'numeric'});
  const date=d=>d?new Date(d.length===10?d+'T12:00:00':d).toLocaleString('it-IT',{timeZone:'Europe/Rome',day:'2-digit',month:'short',year:'numeric',...(d.length>10?{hour:'2-digit',minute:'2-digit'}:{})}):'—';
  const icon=n=>`<i data-lucide="${n}" class="qi-icon" aria-hidden="true"></i>`;
  const s={month:today().slice(0,7),origin:'all',tab:'journal',selectedDay:null,cache:new Map(),loading:new Set(),errors:{},retryAt:{},profileDraft:null,noteDraft:{},search:'',activeTrade:null,tradeData:null,chart:null,push:null,pushLocal:null,message:'',lastView:null,notify:null};
  const get=k=>s.cache.get(k)?.value;
  const currency=()=>get('/api/profile')?.account?.currency||get(calKey())?.currency||'USD';
  const money=(v,c=currency())=>v==null?'—':new Intl.NumberFormat('it-IT',{style:'currency',currency:c}).format(v);
  const sign=v=>v>0?'qi-positive':v<0?'qi-negative':'';
  const action=(label,id,extra='',cls='')=>`<button type="button" class="qi-button ${cls}" data-personal-action="${id}" ${extra}>${label}</button>`;
  const calKey=()=>'/api/calendar?month='+s.month+'&origin='+s.origin;
  const dayKey=()=>'/api/day?day='+s.selectedDay+'&origin='+s.origin;
  const journalKey=()=>'/api/journal?origin='+s.origin;
  const filter=()=>`<div class="qi-segment" role="group" aria-label="Origine delle operazioni">${[['all','Tutto il conto'],['bot','Solo bot'],['other','Altre operazioni']].map(([id,label])=>`<button type="button" data-personal-action="origin" data-origin="${id}" aria-pressed="${s.origin===id}">${label}</button>`).join('')}</div>`;
  const empty=(text)=>`<div class="qi-empty"><strong>${text}</strong></div>`;
  const loading=k=>s.errors[k]?`<p class="qi-personal-error" role="alert">${e(s.errors[k])} ${action('Riprova','retry')}</p>`:`<p class="qi-inline-note" role="status">Caricamento dei dati salvati…</p>`;
  function redraw(){s.notify?.();}
  async function api(url,body){const r=await fetch(url,{method:body===undefined?'GET':'POST',headers:body===undefined?{}:{'Content-Type':'application/json'},body:body===undefined?undefined:JSON.stringify(body),credentials:'same-origin',cache:'no-store',signal:AbortSignal.timeout(15000)});const data=await r.json();if(!r.ok)throw new Error(data.error||'Il servizio non risponde.');return data;}
  async function ensure(key,force=false){if(s.loading.has(key))return;if(!force&&Date.now()<(s.retryAt[key]||0))return;if(!force&&Date.now()-(s.cache.get(key)?.at||0)<20000)return;s.loading.add(key);try{s.cache.set(key,{at:Date.now(),value:await api(key)});delete s.errors[key];delete s.retryAt[key];}catch(err){s.errors[key]=err.message;s.retryAt[key]=Date.now()+20000;}finally{s.loading.delete(key);if(!interacting())redraw();}}
  function kpi(label,value,extra=''){return `<div class="qi-calendar-kpi"><span>${label}</span><strong class="qi-number ${extra}">${value}</strong></div>`;}
  function calendarMarkup(){
    const key=calKey(),data=get(key);
    if(!data)return `<article class="qi-card"><h2>Calendario dei risultati</h2>${loading(key)}</article>`;
    const rows=new Map(data.daily.map(d=>[d.date,d])),year=+s.month.slice(0,4),month=+s.month.slice(5),start=new Date(Date.UTC(year,month-1,1)),offset=(start.getUTCDay()+6)%7,count=new Date(Date.UTC(year,month,0)).getUTCDate(),weeks=Math.ceil((offset+count)/7),cells=[],weekly=[];
    for(let w=0;w<weeks;w++){
      let net=0;const ids=new Set();let executions=0,days=[];
      for(let d=0;d<7;d++){
        const day=w*7+d-offset+1;
        if(day<1||day>count){cells.push('<div class="qi-calendar-day outside" aria-hidden="true"></div>');continue;}
        const id=s.month+'-'+String(day).padStart(2,'0'),row=rows.get(id),amount=row?.net||0;
        net+=amount;executions+=row?.executions||0;days.push(day);
        const compact=new Intl.NumberFormat('it-IT',{notation:Math.abs(amount)>=1000?'compact':'standard',maximumFractionDigits:0}).format(amount);
        const label=`${date(id)}: ${row?money(amount,data.currency)+', '+row.trades+' operazioni':'nessuna esecuzione nello storico disponibile'}`;
        cells.push(`<button type="button" class="qi-calendar-day ${row?(amount>0?'positive':amount<0?'negative':'neutral'):''} ${id===today()?'today':''} ${id===s.selectedDay?'selected':''}" data-personal-action="day" data-day="${id}" aria-pressed="${id===s.selectedDay}" aria-label="${e(label)}" title="${e(label)}"><span class="qi-calendar-date">${day}</span>${row?`<strong class="${sign(amount)}">${amount>0?'+':''}${e(compact)}</strong><small>${row.trades}<span class="qi-calendar-count-label"> ${row.trades===1?'operazione':'operazioni'}</span></small>`:'<small class="qi-calendar-no-trades">—</small>'}</button>`);
      }
      weekly.push(`<div class="qi-calendar-week ${sign(net)}"><div><strong>Settimana ${w+1}</strong><small>${days[0]}–${days.at(-1)} ${monthName(s.month).split(' ')[0]}</small></div><strong class="qi-number">${money(net,data.currency)}</strong><small>${executions} esecuzioni</small></div>`);
    }
    return `<div class="qi-calendar-toolbar">${filter()}${action('Questo mese','this-month','',s.month===today().slice(0,7)?'primary':'')}</div><div class="qi-calendar-kpis">${kpi('P&L netto del mese',money(data.totals.net,data.currency),sign(data.totals.net))}${kpi('Operazioni interessate',num(data.totals.trades))}${kpi('Win rate · trade chiusi',data.totals.win_rate==null?'—':num(data.totals.win_rate)+'%')}${kpi('Giorni attivi',num(data.totals.active_days))}</div><div class="qi-calendar-layout"><article class="qi-card qi-calendar-card"><div class="qi-calendar-month">${action('‹','month','data-step="-1" aria-label="Mese precedente"','ghost')}<h2>${e(monthName(s.month))}</h2>${action('›','month','data-step="1" aria-label="Mese successivo"','ghost')}</div><label class="qi-calendar-picker">Vai al mese <input type="month" data-personal-month value="${s.month}" min="2025-01" max="2100-12"></label><div class="qi-calendar-grid"><div class="qi-calendar-weekday">Lun</div><div class="qi-calendar-weekday">Mar</div><div class="qi-calendar-weekday">Mer</div><div class="qi-calendar-weekday">Gio</div><div class="qi-calendar-weekday">Ven</div><div class="qi-calendar-weekday">Sab</div><div class="qi-calendar-weekday">Dom</div>${cells.join('')}</div><p class="qi-inline-note">Celle con importi arrotondati. Tocca un giorno per importi esatti, operazioni e statistiche. Valuta ${e(data.currency)}.</p></article><aside class="qi-calendar-weeks" aria-label="Riepilogo settimanale">${weekly.join('')}</aside></div>${s.selectedDay?dayMarkup():''}<p class="qi-inline-note">${e(data.note)} Storico richiesto dal ${e(data.coverage.history_from||'primo collegamento')}; la disponibilità effettiva dipende dal broker.</p>`;
  }
  function tradeRow(t,day=false){return `<button type="button" class="qi-diary-row" data-personal-action="trade" data-id="${e(t.id)}"><div><strong>${e(t.side)} ${e(t.symbol)}</strong><small>${date(t.opened)} · ${t.own_bot?'Bot':'Altra operazione'}${t.session?' · '+e(t.session):''}</small><small>${num(t.volume)} lotti · ${({closed:'Chiusa',open:'Aperta',partial:'Parzialmente chiusa',unknown:'Storico incompleto'})[t.state]}${t.has_note?' · nota salvata':''}</small></div><div><strong class="qi-number ${sign(day?t.day_net:t.net)}">${money(day?t.day_net:t.net)}</strong><small>${day?'Contributo al giorno':'Realizzato'}${t.floating!=null?' · aperto '+money(t.floating):''}</small></div>${icon('chevron-right')}</button>`;}
  function dayMarkup(){const key=dayKey(),d=get(key);return `<article class="qi-card qi-day-detail" data-day-detail><div class="qi-section-head"><h2>${date(s.selectedDay)}</h2>${action('Chiudi','close-day','','ghost')}</div>${!d?loading(key):`<div class="qi-calendar-kpis day">${kpi('P&L netto del giorno',money(d.totals.net,d.currency),sign(d.totals.net))}${kpi('Operazioni interessate',num(d.totals.trades))}${kpi('Trade chiusi',num(d.totals.closed))}${kpi('Win rate · trade chiusi',d.totals.win_rate==null?'—':num(d.totals.win_rate)+'%')}</div><div class="qi-day-secondary"><span>Costi registrati: ${money(d.totals.costs,d.currency)}</span><span>Profit factor: ${d.totals.profit_factor==null?'—':num(d.totals.profit_factor)}</span><span>Media trade chiusi: ${money(d.totals.average_closed,d.currency)}</span></div>${d.trades.length?d.trades.map(t=>tradeRow(t,true)).join(''):empty('Nessuna esecuzione nello storico disponibile per questa giornata.')}<details class="qi-personal-details"><summary>Singole esecuzioni (${d.deals.length})</summary>${executions(d.deals,d.currency)}</details>`}</article>`;}
  function executions(deals,c=currency()){return deals.length?`<div class="qi-execution-list">${deals.map(d=>`<div><strong>${date(d.time_italy)} · ${d.entry===0?'Entrata':d.entry===1||d.entry===2?'Uscita':'Esecuzione'} ${e(d.side)} ${e(d.symbol)}</strong><span>${num(d.volume)} lotti a ${num(d.price)}</span><span class="${sign(d.net)}">${money(d.net,c)} <small>costi inclusi</small></span></div>`).join('')}</div>`:empty('Nessuna esecuzione disponibile.');}
  function journalMarkup(legacy){
    const key=journalKey(),data=get(key);const tabs=`<div class="qi-calendar-toolbar"><div class="qi-segment" role="group" aria-label="Vista attività">${[['journal','Diario'],['executions','Esecuzioni recenti']].map(([id,label])=>`<button type="button" data-personal-action="tab" data-tab="${id}" aria-pressed="${s.tab===id}">${label}</button>`).join('')}</div>${filter()}</div>`;
    if(s.tab==='executions')return tabs+legacy;
    const items=(data?.trades||[]).filter(t=>(t.symbol+' '+t.side+' '+(t.session||'')).toLowerCase().includes(s.search.toLowerCase()));
    return `${tabs}<article class="qi-card"><div class="qi-section-head"><div><span class="qi-overline">Registrazione automatica da MT5</span><h2>Diario delle operazioni</h2></div>${action('Esporta','export-journal','','ghost')}</div><label class="qi-search qi-journal-search"><span class="qi-sr">Cerca un’operazione</span><input class="qi-input" data-personal-search placeholder="Cerca asset, direzione o sessione…" value="${e(s.search)}"></label>${!data?loading(key):items.length?items.map(t=>tradeRow(t)).join(''):empty('Nessuna operazione corrispondente.')}<p class="qi-inline-note">${data?'Mostrate '+Math.min(data.trades.length,250)+' di '+data.total+' operazioni salvate. ':''}Il diario conserva le esecuzioni anche quando MT5 è chiuso. SL/TP e contesto vengono registrati dalle letture successive all’attivazione del diario; le motivazioni interne dell’EA non sono esposte.</p></article>`;
  }
  function accountMarkup(){
    const data=get('/api/profile');if(!data)return `<article class="qi-card"><h2>Il tuo account</h2>${loading('/api/profile')}</article>`;
    const p=s.profileDraft||data.profile,status=s.push,local=s.pushLocal;
    const fields=[['name','Nome','text'],['email','Email (facoltativa)','email'],['account_alias','Nome del conto','text']];
    return `<div class="qi-stack qi-personal-account"><article class="qi-card"><span class="qi-overline">Profilo personale · accesso privato</span><h2>Il tuo account</h2><form id="qi-personal-profile"><div class="qi-form-grid">${fields.map(([k,label,type])=>`<label class="qi-field">${label}<input class="qi-input" name="${k}" type="${type}" maxlength="${k==='email'?160:k==='name'?60:80}" value="${e(p[k])}" ${k!=='email'?'required':''}></label>`).join('')}<div class="qi-field"><span>Conto collegato</span><strong>${e(data.account.broker||'MT5')} · ${e(data.account.currency||'—')}</strong><small>${e(data.account.label||'Lettura in attesa')}</small></div></div><details class="qi-personal-details"><summary>Preferenze delle notifiche</summary>${[['notify_open','Aperture delle operazioni'],['notify_close','Chiusure e uscite parziali'],['notify_stops','Variazioni di SL e TP'],['notify_risk','Avviso sul margine alla protezione'],['notify_connection','Perdita e ripristino del collegamento MT5'],['notify_amounts','Mostra gli importi nelle notifiche']].map(([k,label])=>`<label class="qi-toggle-row"><span>${label}</span><input type="checkbox" name="${k}" ${p[k]?'checked':''}></label>`).join('')}<label class="qi-field">Avvisa con margine alla protezione sotto (${e(data.account.currency||'valuta conto')})<input class="qi-input" type="number" min="1" max="10000" step="0.01" name="risk_warning" value="${p.risk_warning}" required></label></details><div class="qi-actions"><button type="submit" class="qi-button primary">Salva profilo e preferenze</button></div></form><p class="qi-inline-note">${e(data.scope)} L’email serve soltanto al profilo; gli avvisi arrivano come notifiche push.</p>${s.message?`<p class="qi-personal-feedback" role="status">${e(s.message)}</p>`:''}</article><article class="qi-card"><h2>Notifiche su questo dispositivo</h2><p class="qi-inline-note" data-notification-state>${!('Notification' in window)?'Questo browser non supporta le notifiche.':local?.subscribed?'Dispositivo iscritto alle notifiche.':Notification.permission==='denied'?'Notifiche bloccate nelle impostazioni del browser.':'Notifiche non ancora attivate su questo dispositivo.'}</p><div class="qi-actions">${action('Attiva notifiche','enable-push','','primary')}${action('Invia una prova','test-push',local?.subscribed?'':'disabled')}${action('Disattiva qui','disable-push',local?.subscribed?'':'disabled','ghost')}</div><p class="qi-inline-note">${e(status?.note||'Serve il tuo consenso nel browser. Il PC e il servizio QUANTIX IQ devono restare accesi e connessi.')} ${status?.last_status?.length?'Ultimo invio: '+e(status.last_status.at(-1))+'.':''}</p><p class="qi-inline-note">Il servizio push del browser (per esempio Google su Android) trasporta gli avvisi cifrati al dispositivo. Non occorre rendere pubblico QUANTIX IQ.</p></article><article class="qi-card"><h2>I tuoi dati salvati</h2><div class="qi-grid2"><div><span>Esecuzioni</span>${num(data.storage.executions)}</div><div><span>Note personali</span>${num(data.storage.notes)}</div></div><div class="qi-actions">${action('Scarica una copia dei miei dati','export-all')}</div><p class="qi-inline-note">Archivio conservato su questo PC e condiviso tra telefono, tablet e computer autorizzati. Una copia scaricata contiene dati del conto e note personali.</p></article></div>`;
  }
  function drawTradeChart(bars,trade,context){
    if(!bars?.length)return '<p class="qi-inline-note">Grafico non disponibile per questa operazione.</p>';
    const w=600,h=230,left=8,right=65,top=14,bottom=27,entry=trade.entry_price,sl=context?.position?.sl,tp=context?.position?.tp;
    let min=Math.min(...bars.map(b=>b.low)),max=Math.max(...bars.map(b=>b.high)),pad=Math.max(.1,(max-min)*.08);min-=pad;max+=pad;
    const x=i=>left+(i+.5)*(w-left-right)/bars.length,y=p=>top+(max-p)/(max-min)*(h-top-bottom),bw=Math.max(2,(w-left-right)/bars.length*.55);
    let shapes='';for(let i=0;i<4;i++){const v=min+(max-min)*i/3;shapes+=`<line x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}" stroke="var(--qi-border)"/><text x="${w-right+6}" y="${y(v)+4}">${num(v)}</text>`;}
    bars.forEach((b,i)=>{const color=b.close>=b.open?'var(--qi-green)':'var(--qi-red)';shapes+=`<line x1="${x(i)}" x2="${x(i)}" y1="${y(b.high)}" y2="${y(b.low)}" stroke="${color}"/><rect x="${x(i)-bw/2}" width="${bw}" y="${Math.min(y(b.open),y(b.close))}" height="${Math.max(1,Math.abs(y(b.open)-y(b.close)))}" fill="${color}"/>`;});
    for(const [p,label,color] of [[entry,'Entrata','var(--qi-accent)'],[sl,'SL osservato','var(--qi-red)'],[tp,'TP osservato','var(--qi-green)']])if(p&&p>=min&&p<=max)shapes+=`<line x1="${left}" x2="${w-right}" y1="${y(p)}" y2="${y(p)}" stroke="${color}" stroke-dasharray="5 4"/><text x="${left+5}" y="${y(p)-5}">${label} ${num(p)}</text>`;
    return `<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Candele M5 reali relative all’operazione">${shapes}</svg>`;
  }
  function modal(){return document.querySelector('#quantix-app dialog');}
  function drawTrade(){
    const d=s.tradeData,dialog=modal();if(!dialog)return;
    dialog.dataset.personalModal='true';dialog.onclose=()=>{s.activeTrade=null;s.tradeData=null;delete dialog.dataset.personalModal;};
    if(!d){dialog.innerHTML=`${action('Chiudi','close-modal','','ghost')}<h2>Operazione</h2><p class="qi-inline-note">Caricamento del diario…</p>`;if(!dialog.open)dialog.showModal();return;}
    const t=d.trade,context=d.context,position=context?.position;
    dialog.innerHTML=`<button type="button" class="qi-icon-button qi-modal-close" data-personal-action="close-modal" aria-label="Chiudi">×</button><span class="qi-overline">Diario · ${t.own_bot?'Bot':'Altra operazione'}${t.session?' · '+e(t.session):''}</span><h2>${e(t.side)} ${e(t.symbol)}</h2><div class="qi-trade-result ${sign(t.net)}">${money(t.net,d.currency)} <small>realizzato</small></div><div class="qi-grid2"><div><span>Apertura</span>${date(t.opened)}</div><div><span>Chiusura</span>${date(t.closed)}</div><div><span>Entrata</span>${num(t.entry_price)}</div><div><span>Ultima uscita</span>${num(t.exit_price)}</div><div><span>Volume iniziale</span>${num(t.volume)} lotti</div><div><span>Costi complessivi</span>${money(t.costs,d.currency)}</div></div><div class="qi-diary-chart">${drawTradeChart(s.chart||context?.candles,t,context)}</div><p class="qi-inline-note">${context?'Prima rilevazione del diario: '+date(context.captured_at)+'. SL/TP indicati sono quelli osservati in quel momento.':'Operazione precedente all’attivazione del diario: SL/TP e filtri iniziali non sono stati registrati.'}</p>${position?`<div class="qi-grid2"><div><span>SL alla prima rilevazione</span>${position.sl?num(position.sl):'Assente'}</div><div><span>TP alla prima rilevazione</span>${position.tp?num(position.tp):'Assente'}</div></div>`:''}<details class="qi-personal-details"><summary>Esecuzioni e costi (${t.deals.length})</summary>${executions(t.deals,d.currency)}</details>${context?`<details class="qi-personal-details"><summary>Contesto alla prima rilevazione</summary><p class="qi-inline-note">${e(context.analysis.title)}. ${e(context.analysis.reason)}</p><p class="qi-inline-note">${e(context.source)} Non è il ragionamento interno dell’EA.</p></details>`:''}<details class="qi-personal-details"><summary>Variazioni di SL/TP osservate (${d.events.length})</summary>${d.events.length?d.events.map(v=>`<p class="qi-inline-note">${date(v.stamp)} · SL ${num(v.data.before.sl)} → ${num(v.data.after.sl)} · TP ${num(v.data.before.tp)} → ${num(v.data.after.tp)}</p>`).join(''):'<p class="qi-inline-note">Nessuna variazione registrata dal diario. Le letture periodiche possono non rilevare modifiche intermedie.</p>'}</details><form id="qi-personal-note" data-id="${e(t.id)}"><label class="qi-field">La tua nota<textarea class="qi-input" name="note" maxlength="4000" rows="4" placeholder="Osservazioni su questa operazione…">${e(s.noteDraft[t.id]??d.note)}</textarea></label><div class="qi-actions"><button class="qi-button primary" type="submit">Salva nota</button></div></form><p class="qi-personal-feedback" data-note-feedback role="status"></p>`;
    if(!dialog.open)dialog.showModal();window.lucide?.createIcons();
    dialog.onclose=()=>{s.activeTrade=null;s.tradeData=null;delete dialog.dataset.personalModal;};
  }
  async function openTrade(id){
    s.activeTrade=String(id);s.tradeData=null;s.chart=null;drawTrade();
    try{
      const data=await api('/api/journal/trade?id='+encodeURIComponent(id));
      if(s.activeTrade!==String(id))return;
      s.tradeData=data;s.chart=data.context?.candles;drawTrade();
      const raw=data.trade.deals.find(d=>d.entry===0)?.time_raw;
      if(!s.chart?.length&&raw&&data.trade.symbol===data.chart_symbol){
        try{const chart=await api('/api/chart?tf=M5&count=48&before='+(Number(raw)+3600));if(s.activeTrade===String(id)){s.chart=chart.candles||chart.bars;drawTrade();}}
        catch(_){/* Il diario salvato resta consultabile anche senza lo storico grafico di MT5. */}
      }
    }catch(err){if(s.activeTrade===String(id)){const dialog=modal();if(dialog)dialog.innerHTML=`${action('Chiudi','close-modal')}<p class="qi-personal-error">${e(err.message)}</p>`;}}
  }
  function interacting(){return Boolean(modal()?.open&&modal()?.dataset.personalModal);}
  async function pushLocal(){if(!('serviceWorker' in navigator)||!('Notification' in window)||!('PushManager' in window))return {supported:false};const reg=await navigator.serviceWorker.getRegistration('/01_app/');const sub=await reg?.pushManager.getSubscription();return {supported:true,subscribed:Boolean(sub),endpoint:sub?.endpoint};}
  async function refreshPush(){try{s.push=await api('/api/push/status');s.pushLocal=await pushLocal();}catch(err){s.message=err.message;}}
  function decodeKey(text){const pad='='.repeat((4-text.length%4)%4);return Uint8Array.from(atob((text+pad).replace(/-/g,'+').replace(/_/g,'/')),c=>c.charCodeAt(0));}
  async function pushAction(which){
    if(!('Notification' in window)||!('PushManager' in window)||!navigator.serviceWorker||!isSecureContext)throw new Error('Apri QUANTIX IQ in Chrome tramite il collegamento HTTPS privato.');
    if(which==='enable-push'){
      const permission=await Notification.requestPermission();if(permission!=='granted')throw new Error('Consenso non concesso. Puoi abilitarlo nelle impostazioni delle notifiche di Chrome.');
      const status=await api('/api/push/status');if(!status.available)throw new Error('Il servizio notifiche non è ancora pronto sul PC.');
      const reg=await navigator.serviceWorker.ready;let sub=await reg.pushManager.getSubscription();if(!sub)sub=await reg.pushManager.subscribe({userVisibleOnly:true,applicationServerKey:decodeKey(status.public_key)});
      await api('/api/push/subscribe',sub.toJSON());s.message='Notifiche attivate su questo dispositivo. Usa «Invia una prova» per verificarle.';
    }else{
      const reg=await navigator.serviceWorker.ready,sub=await reg.pushManager.getSubscription();if(!sub)throw new Error('Attiva prima le notifiche su questo dispositivo.');
      if(which==='disable-push'){await api('/api/push/unsubscribe',{endpoint:sub.endpoint});await sub.unsubscribe();s.message='Notifiche disattivate su questo dispositivo.';}
      else{s.message=(await api('/api/push/test',{endpoint:sub.endpoint})).message;}
    }
    await refreshPush();
  }
  function download(value,name,type='application/json'){const url=URL.createObjectURL(new Blob([typeof value==='string'?value:JSON.stringify(value,null,2)],{type}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
  document.addEventListener('click',async event=>{
    const b=event.target.closest('[data-personal-action]');if(!b)return;const id=b.dataset.personalAction;
    try{
      if(id==='origin'){s.origin=b.dataset.origin;redraw();return;}
      if(id==='month'){const d=new Date(s.month+'-15T12:00:00');d.setMonth(d.getMonth()+Number(b.dataset.step));const m=d.getFullYear()+'-'+String(d.getMonth()+1).padStart(2,'0');if(m<'2025-01'||m>'2100-12')return;s.month=m;s.selectedDay=null;redraw();return;}
      if(id==='this-month'){s.month=today().slice(0,7);s.selectedDay=null;redraw();return;}
      if(id==='day'){s.selectedDay=b.dataset.day;redraw();await ensure(dayKey(),true);document.querySelector('[data-day-detail]')?.scrollIntoView({behavior:'smooth',block:'start'});return;}
      if(id==='close-day'){s.selectedDay=null;redraw();return;}
      if(id==='tab'){s.tab=b.dataset.tab;redraw();return;}
      if(id==='trade'){await openTrade(b.dataset.id);return;}
      if(id==='close-modal'){modal()?.close();s.activeTrade=null;return;}
      if(id==='retry'){s.cache.clear();s.errors={};s.retryAt={};redraw();return;}
      if(id==='export-all'){download(await api('/api/archive/export'),'QUANTIX_IQ_dati_personali.json');return;}
      if(id==='export-journal'){const data=await api(journalKey());const rows=[['id','asset','direzione','apertura','chiusura','volume','realizzato','origine'],...data.trades.map(t=>[t.id,t.symbol,t.side,t.opened,t.closed,t.volume,t.net,t.own_bot?'bot':'altra'])];download('\ufeff'+rows.map(r=>r.map(v=>'"'+String(v??'').replace(/"/g,'""')+'"').join(';')).join('\r\n'),'QUANTIX_IQ_diario.csv','text/csv;charset=utf-8');return;}
      if(['enable-push','disable-push','test-push'].includes(id)){b.disabled=true;await pushAction(id);redraw();return;}
    }catch(err){s.message=err.message;redraw();}
  });
  document.addEventListener('input',event=>{
    const form=event.target.closest('#qi-personal-profile');if(form){if(!s.profileDraft)s.profileDraft={...get('/api/profile')?.profile};const field=event.target;s.profileDraft[field.name]=field.type==='checkbox'?field.checked:field.name==='risk_warning'?Number(field.value):field.value;}
    if(event.target.closest('#qi-personal-note'))s.noteDraft[event.target.closest('form').dataset.id]=event.target.value;
    if(event.target.matches('[data-personal-search]')){s.search=event.target.value;const start=event.target.selectionStart;redraw();const input=document.querySelector('[data-personal-search]');input?.focus();input?.setSelectionRange(start,start);}
  });
  document.addEventListener('change',event=>{if(event.target.matches('[data-personal-month]')&&/^20\d\d-(0[1-9]|1[0-2])$/.test(event.target.value)){s.month=event.target.value;s.selectedDay=null;redraw();}});
  document.addEventListener('submit',async event=>{
    const form=event.target;if(!['qi-personal-profile','qi-personal-note'].includes(form.id))return;event.preventDefault();
    try{
      if(form.id==='qi-personal-profile'){const p={...(s.profileDraft||get('/api/profile').profile)},fd=new FormData(form);for(const key of ['name','email','account_alias'])p[key]=fd.get(key);for(const key of Object.keys(p).filter(k=>k.startsWith('notify_')))p[key]=fd.has(key);p.risk_warning=Number(fd.get('risk_warning'));await api('/api/profile',p);s.profileDraft=null;s.message='Profilo e preferenze salvati sul PC e disponibili sugli altri tuoi dispositivi.';await ensure('/api/profile',true);redraw();}
      else{const id=form.dataset.id,text=new FormData(form).get('note');await api('/api/journal/note',{id,note:text});s.noteDraft[id]=text;if(s.tradeData)s.tradeData.note=text;const feedback=document.querySelector('[data-note-feedback]');if(feedback)feedback.textContent='Nota salvata.';s.cache.delete(journalKey());}
    }catch(err){if(form.id==='qi-personal-note'){document.querySelector('[data-note-feedback]').textContent=err.message;}else{s.message=err.message;redraw();}}
  });
  function mount(view,snapshot,notify){
    s.notify=notify;ensure('/api/profile');
    if(['settings','stats','activity'].includes(view)){
      ensure('/api/profile');
      if(view==='stats'){ensure(calKey());if(s.selectedDay)ensure(dayKey());}
      if(view==='activity')ensure(journalKey());
      if(view==='settings'&&s.lastView!=='settings')refreshPush().then(redraw);
    }
    s.lastView=view;
    if(view==='activity'){
      const query=new URLSearchParams(location.search),id=query.get('trade');if(id&&/^[0-9]{1,24}$/.test(id)&&!s.activeTrade){history.replaceState(null,'',location.pathname);openTrade(id);}
    }
  }
  window.QuantixPersonal={origin:()=>s.origin,calendarMarkup,journalMarkup,accountMarkup,mount,interacting,openTrade,profileName:fallback=>get('/api/profile')?.profile?.name||fallback};
})();
