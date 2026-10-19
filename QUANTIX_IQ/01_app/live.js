/* Dati reali in sola lettura. Le spiegazioni ricostruite sono sempre etichettate. */
(function(){
  'use strict';
  let mode='demo',snapshot=null,networkError='',privateUrl=null,busy=false,lastSuccess=0;
  try{mode=localStorage.getItem('quantix-data-source')||((location.protocol==='http:'||location.protocol==='https:')?'live':'demo');}catch(_){}
  let context=null;
  const messages=[];
  const openPositionDetails=new Set();
  let aiPending=false,aiInfo=null,chatDraft='',aiStarted=0,aiPartial='';
  const escape=value=>String(value??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icon=name=>`<i data-lucide="${name}" class="qi-icon" aria-hidden="true"></i>`;
  const currency=()=>snapshot?.account?.currency||'USD';
  const money=value=>value==null?'—':new Intl.NumberFormat('it-IT',{style:'currency',currency:currency()}).format(value);
  const number=value=>value==null?'—':new Intl.NumberFormat('it-IT',{minimumFractionDigits:2,maximumFractionDigits:2}).format(value);
  const clock=value=>value?String(value).slice(11,16):'—';
  const date=value=>value?new Date(value).toLocaleString('it-IT',{timeZone:'Europe/Rome',day:'2-digit',month:'2-digit',hour:'2-digit',minute:'2-digit'}):'—';
  const navs={home:['Home','house'],analysis:['Analisi','scan-line'],chat:['Chat','message-circle'],activity:['Attività','list'],stats:['Statistiche','chart-no-axes-column'],settings:['Account','user-round']};
  const titles={home:'Panoramica',analysis:'Analisi del bot',chat:'Chat del bot',activity:'Attività',stats:'Statistiche',settings:'Il tuo account'};
  const liveButton=(text,action,extra='')=>`<button type="button" class="qi-button" data-live-action="${action}" ${extra}>${text}</button>`;
  const modeMarkup=()=>`<div class="qi-segment" role="group" aria-label="Origine dei dati"><button data-action="source" data-source="live" aria-pressed="${mode==='live'}">MT5 · sola lettura</button><button data-action="source" data-source="demo" aria-pressed="${mode==='demo'}">Demo</button></div>`;
  function stale(){return Boolean(networkError||!snapshot?.ok||!navigator.onLine||Date.now()-lastSuccess>20000);}
  function announce(){window.dispatchEvent(new Event('quantix:live-update'));}
  function metric(label,value,detail='',positive=false){return `<article class="qi-card qi-kpi"><div class="qi-kpi-head">${label}</div><div class="qi-kpi-value qi-number ${positive?'qi-positive':''}">${value}</div><p class="qi-kpi-foot">${detail}</p></article>`;}
  function entryAvailability(){
    const r=snapshot?.bot?.entry_rules;if(!r)return '';
    const observer=snapshot.analysis?.bot_observer;
    return `<article class="qi-card qi-entry-availability"><div class="qi-section-head"><h2>Nuovi ingressi tra sessioni</h2><span class="qi-pill">${r.own_positions}/${r.maximum_positions} posizioni</span></div><p><strong>${r.free_slots>0?r.free_slots+(r.free_slots===1?' posto teoricamente disponibile':' posti teoricamente disponibili'):'Limite di posizioni raggiunto'}</strong></p><p class="qi-inline-note">Un ingresso ordinario della nuova sessione può convivere con posizioni Asia. Deve comunque superare i filtri M5 e rientrare nel budget di rischio complessivo del ${number(r.total_open_stop_risk_percent)}%.</p><details><summary>Le condizioni da rispettare</summary><ul><li>Massimo ${r.maximum_positions} posizioni proprie, ${r.session_max_entries} ingressi per sessione e ${r.max_trades_day} al giorno.</li><li>Gli ingressi recuperati dopo lo scarto ATR ${r.deferred_only_when_flat?'richiedono zero posizioni proprie aperte':'possono convivere con altre posizioni, entro i limiti'}.</li>${r.primary_entry_priority?'<li>Un ingresso ordinario ha priorità: il bot tenta prima di chiudere eventuali posizioni differite DATR.</li>':''}<li>Il cambio sessione non chiude da solo le posizioni Asia ordinarie.</li><li>I posti liberi non confermano un segnale o la disponibilità del budget di rischio.</li></ul><p class="qi-inline-note">Profilo salvato e codice prova298. ${observer?.available?'Contatore esposto dal bot: '+observer.trades_today_from_comment+' ingressi oggi.':'Il contatore interno degli ingressi non è disponibile.'}</p></details></article>`;
  }
  function positionCards(){
    const positions=snapshot?.positions||[],total=snapshot?.account?.floating;
    return `<article class="qi-card qi-open-positions" aria-label="Operazioni aperte e profitto attuale"><div class="qi-open-header"><h2>Operazioni aperte <span class="qi-count">${positions.length}</span></h2><div class="qi-open-total"><span>Profitto / perdita aperta</span><strong data-open-profit class="qi-number ${total>0?'qi-positive':total<0?'qi-negative':''}">${money(total)}</strong></div></div>${positions.length?positions.map(p=>`<div class="qi-position qi-open-row"><div class="qi-position-top"><div><strong>${escape(p.side)} ${escape(p.symbol)}</strong><p class="qi-small qi-muted">${number(p.volume)} lotti · ${p.own_bot?'prova298':'Altra posizione del conto'}${p.own_bot&&p.session_tag?' · '+escape(p.session_tag):''}</p></div><strong class="qi-number ${p.profit>=0?'qi-positive':'qi-negative'}" data-position-profit>${money(p.profit)}</strong></div><details class="qi-position-details" data-position-details="${escape(p.position_id)}" ${openPositionDetails.has(String(p.position_id))?'open':''}><summary>Prezzi, stop loss e take profit</summary><div class="qi-grid2"><div><span>Entrata</span>${number(p.entry_price)}</div><div><span>Prezzo attuale</span>${number(p.price)}</div><div><span>Stop loss</span>${p.sl?number(p.sl):'Assente'}</div><div><span>Take profit</span>${p.tp?number(p.tp):'Assente'}</div></div>${p.swap?`<p class="qi-small qi-muted">Swap: ${money(p.swap)}</p>`:''}</details></div>`).join(''):`<p class="qi-open-empty">Nessuna posizione aperta · ${snapshot?.pending_orders??0} ordini pendenti.</p>`}<p class="qi-open-foot">${stale()?'Ultimi dati disponibili · non aggiornati':'Dati MT5 · conto completo'} · ${escape(currency())} · risultato non realizzato.</p></article>`;
  }
  function sessionMarkup(){
    return (snapshot?.analysis?.sessions||[]).map(session=>{
      const label={disabled:'Disabilitata',waiting:'In attesa',forming:'Range in formazione',searching:'Finestra temporale aperta',ended:'Finestra terminata'}[session.phase];
      return `<div class="qi-live-session"><div class="qi-section-head"><strong>${session.name}</strong><span class="qi-pill">${label}</span></div><p class="qi-small qi-muted">Range ${clock(session.start_italy)}–${clock(session.end_italy)} · ricerca fino alle ${clock(session.deadline_italy)} italiane</p><div class="qi-grid2" style="margin-top:10px"><div><span>Massimo ${session.level_source==='bot_comment'?'del bot':'ricostruito'}</span>${number(session.level_source==='bot_comment'?session.bot_high:session.high)}</div><div><span>Minimo ${session.level_source==='bot_comment'?'del bot':'ricostruito'}</span>${number(session.level_source==='bot_comment'?session.bot_low:session.low)}</div></div><p class="qi-inline-note">${session.bars_available}/${session.expected_bars} candele M1 osservate. ${session.phase==='forming'?'Livelli provvisori. ':''}${session.level_source==='bot_comment'?'Valori letti dal commento del bot in MT5.':'Il range interno del bot potrebbe differire.'}</p></div>`;
    }).join('');
  }
  function chart(id){if(window.QuantixMarketChart)return window.QuantixMarketChart.markup(id);return `<div class="qi-chart" id="${id}" data-live-chart aria-label="Candele M5 reali lette da MT5"></div><p class="qi-small qi-muted">XAUUSD · M5 · orario italiano · prezzi del broker</p>`;}
  function accountSummary(){
    const a=snapshot.account;
    return `<article class="qi-card qi-account-summary" aria-label="Riepilogo del conto"><div><span>Saldo</span><strong class="qi-number">${money(a.balance)}</strong></div><div><span>Equity</span><strong class="qi-number">${money(a.equity)}</strong></div><div><span>Realizzato oggi</span><strong class="qi-number ${a.day_closed_net>=0?'qi-positive':'qi-negative'}">${money(a.day_closed_net)}</strong></div><div><span>Margine alla protezione <small>stimato</small></span><strong class="qi-number ${a.protection_distance_estimate<=0?'qi-negative':''}">${money(a.protection_distance_estimate)}</strong></div></article>`;
  }
  function home(){
    const s=snapshot,connected=!stale()&&s.connection.connected;
    return `${positionCards()}${accountSummary()}<article class="qi-card qi-bot-summary"><span class="qi-dot ${connected&&s.connection.algo_enabled?'':'offline'}" aria-hidden="true"></span><div><strong>${escape(s.bot.name)} · ${escape(s.bot.symbol)}</strong><p>${connected?'MT5 connesso':'Collegamento non aggiornato'} · Algo Trading ${s.connection.algo_enabled?'abilitato':'disabilitato'}</p></div><button type="button" class="qi-button ghost" data-action="nav" data-view="analysis">Analisi ${icon('chevron-right')}</button></article><article class="qi-card qi-market-card qi-home-chart"><div class="qi-section-head"><h2>Mercato</h2><button type="button" class="qi-button ghost" data-action="nav" data-view="analysis">Approfondisci ${icon('arrow-up-right')}</button></div>${chart('live-home-chart')}</article>`;
  }
  function analysis(){
    const s=snapshot,a=s.account,decision=s.analysis;
    return `<div class="qi-stack qi-analysis-page"><article class="qi-card"><span class="qi-overline">${escape(s.bot.name)} · ${escape(s.bot.symbol)} · M5</span><h2>${stale()?'Dati non aggiornati':escape(decision.title)}</h2><p class="qi-inline-note">${escape(decision.reason)}</p><p class="qi-inline-note">${escape(s.bot.readiness||'')}</p></article>${window.QuantixEntryCriteria?.markup(snapshot,stale())||''}${entryAvailability()}<article class="qi-card"><h2>Range e orari delle sessioni</h2>${sessionMarkup()}</article><article class="qi-card"><h2>Protezione del conto</h2><div class="qi-grid2 qi-risk-summary"><div><span>Equity attuale</span>${money(a.equity)}</div><div><span>Soglia stimata</span>${money(a.protection_estimate)}</div><div><span>Margine stimato</span>${money(a.protection_distance_estimate)}</div><div><span>Budget stimato nuovi ingressi</span>${money(s.analysis.entry_criteria?.risk_budget_estimate)}</div></div><p class="qi-inline-note">${escape(a.risk_note)}</p></article><article class="qi-card"><h2>Ultima candela chiusa</h2><p class="qi-inline-note">${s.analysis.last_closed_m5?`${date(s.analysis.last_closed_m5.time_italy)} · chiusura ${number(s.analysis.last_closed_m5.close)} · corpo ${number(s.analysis.last_closed_m5.body_pct)}%`:'Dato non disponibile.'}</p></article><article class="qi-card qi-market-card"><h2>Grafico e strumenti</h2>${chart('live-analysis-chart')}</article></div>`;
  }
  function chat(){
    return `${snapshot?.ok?`<article class="qi-card qi-chat-chart"><h2>Il mercato della conversazione</h2>${chart('live-chat-chart')}</article>`:''}<article class="qi-card qi-ai-card"><div class="qi-section-head"><h2>Parla con QUANTIX IQ</h2><span class="qi-pill">AI LOCALE · VELOCE</span></div><p class="qi-small qi-muted">${escape(aiInfo?.model||'llama3.1:8b')} sul tuo PC · puoi fare domande libere. Quando chiedi del bot, il modello riceve i dati osservati da MT5.</p><p class="qi-inline-note">Le risposte sono spiegazioni AI: possono contenere errori e non sono il ragionamento interno dell’EA. Nessun accesso agli ordini o al web. La conversazione resta in questa pagina fino alla chiusura o al ricaricamento.</p>${aiInfo?.available===false?`<p class="qi-ai-error" role="status">${escape(aiInfo.message)}</p>`:''}<div class="qi-chat-log" aria-live="polite">${[{role:'bot',text:'Ciao Marco. Puoi chiedermi del bot, del conto o di altri argomenti. '+(stale()?'Al momento non ho una lettura MT5 aggiornata.':'Userò i dati del conto disponibili al momento della tua domanda.')},...messages].map(m=>`<div class="qi-message ${m.role==='user'?'user':''} ${m.error?'qi-ai-error':''}"><div class="qi-message-head">${m.role==='user'?'Tu':m.model?'QUANTIX IQ · '+escape(m.model):'QUANTIX IQ'}${m.observed?'<span>MT5 letto alle '+date(m.observed)+'</span>':''}</div><div class="qi-message-text">${escape(m.text)}</div></div>`).join('')}${aiPending?`<div class="qi-message qi-ai-pending" role="status"><div class="qi-message-head">QUANTIX IQ · ${escape(aiInfo?.model||'llama3.1:8b')}</div><div class="qi-message-text" data-ai-stream>${aiPartial?escape(aiPartial):'<span class="qi-ai-dots" aria-hidden="true">● ● ●</span> Sto preparando la risposta…'}</div><span class="qi-small qi-muted" data-ai-writing>${aiPartial?'In scrittura…':'Il modello locale si sta preparando. Puoi continuare a usare l’app.'}</span></div>`:''}</div><div class="qi-chips">${['Puoi entrare con le posizioni Asia aperte?','Come sta il conto?','Spiegami il rapporto rischio/rendimento'].map(q=>`<button type="button" data-live-question="${escape(q)}" ${aiPending?'disabled':''}>${q}</button>`).join('')}</div><form id="qi-live-chat-form" class="qi-compose"><textarea id="qi-live-question" name="question" placeholder="Scrivi una domanda, anche non suggerita…" aria-label="Domanda libera all’AI" required maxlength="4000">${escape(chatDraft)}</textarea><button class="qi-button primary" type="submit" aria-label="Invia domanda" ${aiPending?'disabled':''}>${icon('send')}</button></form></article>`;
  }
  function activity(){
    const source=window.QuantixPersonal?.origin()||'all',isOwn=t=>t.magic===snapshot?.bot?.magic&&t.symbol===snapshot?.bot?.symbol;
    const trades=(snapshot?.deals||[]).filter(t=>source==='all'||(source==='bot'?isOwn(t):!isOwn(t))).slice().reverse().slice(0,100);
    const legacy=`<article class="qi-card"><div class="qi-section-head"><h2>Attività del conto</h2>${liveButton('Esporta CSV','export')}</div><p class="qi-inline-note">Esecuzioni realmente registrate da MT5, incluse entrate e uscite parziali. Non sono i motivi di scarto dei segnali.</p><div class="qi-activity-list">${trades.length?trades.map(t=>`<div class="qi-event"><time>${clock(t.time_italy)}</time><span class="qi-event-symbol">${icon(t.side==='BUY'?'trending-up':'trending-down')}</span><div><strong>${t.entry===0?'Entrata':t.entry===1?'Uscita':'Esecuzione'} ${t.side} ${escape(t.symbol)}</strong><p>${date(t.time_italy)} · ${number(t.volume)} lotti a ${number(t.price)} · ${isOwn(t)?'prova298':'altra operazione'}</p><p>Risultato dell’esecuzione, costi inclusi: ${money(t.net)}</p></div></div>`).join(''):'<div class="qi-empty">Nessuna esecuzione nei dati disponibili.</div>'}</div></article>`;
    return window.QuantixPersonal?.journalMarkup(legacy)||legacy;
  }
  function stats(){
    if(window.QuantixPersonal)return window.QuantixPersonal.calendarMarkup();
    const rows=snapshot?.daily||[],net=rows.reduce((sum,r)=>sum+r.net,0),winners=rows.filter(r=>r.net>0).length;
    return `<div class="qi-kpis">${metric('Netto del periodo',money(net),'Cronologia disponibile: fino a 35 giorni',net>=0)}${metric('Giorni positivi',winners,'Giorni con risultato realizzato > 0')}${metric('Esecuzioni lette',snapshot.deals.length,'Entrate e uscite, non trade completi')}${metric('Drawdown storico','Non disponibile','Richiede una serie di equity registrata')}</div><article class="qi-card"><div class="qi-section-head"><h2>Risultati realizzati del conto</h2>${liveButton('Esporta CSV','export')}</div><p class="qi-inline-note">Valuta ${escape(currency())}. Costi di entrata e uscita inclusi nel giorno di registrazione. Dati reali recenti, distinti dai risultati del backtest.</p><div class="qi-table-scroll"><table><thead><tr><th>Giorno italiano</th><th>Risultato netto</th></tr></thead><tbody>${rows.slice().reverse().map(r=>`<tr><td>${r.date}</td><td class="${r.net>=0?'qi-positive':'qi-negative'}">${money(r.net)}</td></tr>`).join('')}</tbody></table></div></article>`;
  }
  function settings(){
    const b=snapshot?.bot,c=snapshot?.connection;
    return `${window.QuantixPersonal?.accountMarkup()||''}<div class="qi-stack"><article class="qi-card"><h2>Installazione sul telefono</h2><p class="qi-inline-note">Android: apri l’indirizzo privato in Chrome, poi scegli Installa app dal menu ⋮. Mantieni Tailscale connesso con lo stesso account del PC.</p>${window.QuantixPWA?.markup()||''}<p class="qi-inline-note">${privateUrl?`Indirizzo privato: <a class="qi-live-url" href="${escape(privateUrl)}">${escape(privateUrl)}</a>`:'Collegamento privato in preparazione.'}</p></article><article class="qi-card"><h2>Collegamento MT5</h2><div class="qi-check-row"><span>Terminale</span><span>${c?.connected?'Connesso':'Non disponibile'}</span></div><div class="qi-check-row"><span>Algo Trading</span><span>${c?.algo_enabled?'Abilitato':'Non disponibile'}</span></div><div class="qi-check-row"><span>Profilo letto</span><span>${escape(b?.name||'—')}</span></div><p class="qi-inline-note">${escape(b?.readiness||'Lettura in corso.')}</p><p class="qi-inline-note">${escape(snapshot?.account?.risk_note||'')}</p></article><article class="qi-card"><h2>Comandi e dati</h2><p class="qi-inline-note">Questa versione non apre, chiude o modifica operazioni. Per cambiare le impostazioni del bot si usa MT5. La Chat e i risultati di simulazione rimangono separati dai dati reali.</p>${liveButton('Aggiorna i dati','refresh')}<div class="qi-actions"><button type="button" class="qi-button" data-action="source" data-source="demo">Apri la demo interattiva</button></div></article></div>`;
  }
  function drawCharts(){
    document.querySelectorAll('#quantix-app [data-live-chart]').forEach(el=>{
      if(window.QuantixMarketChart){window.QuantixMarketChart.draw(el,snapshot);return;}
      const bars=snapshot?.candles?.slice(-48)||[];
      if(!bars.length){el.innerHTML='<p class="qi-inline-note">Grafico non disponibile.</p>';return;}
      const w=Math.max(210,el.clientWidth),h=225,right=65,left=3,top=14,bottom=28,pw=w-left-right,ph=h-top-bottom;
      let min=Math.min(...bars.map(b=>b.low)),max=Math.max(...bars.map(b=>b.high));
      const pad=Math.max(.2,(max-min)*.08);min-=pad;max+=pad;
      const y=p=>top+(max-p)/(max-min)*ph,x=i=>left+(i+.5)*pw/bars.length,bw=Math.max(2,pw/bars.length*.58);
      let marks='';
      for(let i=0;i<4;i++){const v=min+(max-min)*i/3;marks+=`<line class="grid" x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}"/><text x="${w-right+6}" y="${y(v)+4}">${number(v)}</text>`;}
      bars.forEach((b,i)=>{const c=b.close>=b.open?'up':'down';marks+=`<line class="${c}" x1="${x(i)}" x2="${x(i)}" y1="${y(b.high)}" y2="${y(b.low)}"/><rect class="${c}" x="${x(i)-bw/2}" y="${Math.min(y(b.open),y(b.close))}" width="${bw}" height="${Math.max(1.4,Math.abs(y(b.open)-y(b.close)))}"/>`;});
      for(const i of [0,Math.floor(bars.length/2),bars.length-1])marks+=`<text x="${x(i)}" y="${h-4}" text-anchor="${i===0?'start':i===bars.length-1?'end':'middle'}">${clock(bars[i].time_italy)}</text>`;
      el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Ultime 48 candele M5 di XAUUSD lette da MT5, orario italiano">${marks}</svg>`;
    });
  }
  function render(ctx){
    const draft=document.getElementById('qi-live-question');if(draft)chatDraft=draft.value;
    context=ctx;const {root,theme,name,view}=ctx;
    root.dataset.theme=theme;root.dataset.source='live';root.dataset.view=view;
    const dataAvailable=snapshot?.ok;const displayName=window.QuantixPersonal?.profileName(name)||name;
    const brand=`<button class="qi-brand" data-action="nav" data-view="home" aria-label="QUANTIX IQ, vai alla home"><img src="${escape(root.dataset.logo||'../02_immagini/logo/logo-app.png')}" alt="Logo QUANTIX IQ"><span>QUANTIX <em>IQ</em></span></button>`;
    const themeMarkup=`<div class="qi-theme" role="group" aria-label="Tema dell’applicazione">${['light','dark'].map((t,i)=>`<button data-action="theme" data-theme="${t}" aria-label="Tema ${i?'scuro':'chiaro'}" aria-pressed="${theme===t}">${icon(i?'moon':'sun')}</button>`).join('')}</div>`;
    root.innerHTML=`<div class="qi-shell"><div class="qi-main"><div class="qi-mobile-brand">${brand}</div><header class="qi-top"><div><span class="qi-overline">${view==='home'?'Ciao '+escape(displayName):'QUANTIX IQ'}</span><h1>${titles[view]}</h1></div><div class="qi-top-actions">${window.QuantixPWA?.markup()||''}${themeMarkup}</div></header><div class="qi-demo-line"><span class="qi-pill">MT5 · SOLA LETTURA${dataAvailable?' · '+escape(currency()):''}</span>${modeMarkup()}</div><div class="qi-live-refresh"><span>${stale()?'Collegamento non aggiornato':'Ultima lettura: '+date(snapshot.observed_italy)}${dataAvailable?' · prezzo '+date(snapshot.quote.time_italy):''}</span>${liveButton('Aggiorna','refresh')}</div>${stale()?`<div class="qi-card qi-live-warning" role="status"><strong>${dataAvailable?'Ultimi dati disponibili, non attuali':'Dati MT5 non disponibili'}</strong><p>${escape(networkError||snapshot?.error||'Attendo il collegamento al PC.')}</p></div>`:''}<section aria-label="${titles[view]}">${!dataAvailable&&!['settings','stats','activity'].includes(view)?`<article class="qi-card"><h2>Collegamento a MT5</h2><p class="qi-inline-note">I dati reali compariranno quando il PC e il terminale risponderanno. Puoi intanto provare le funzioni nella Demo, chiaramente separata.</p><div class="qi-actions">${liveButton('Riprova','refresh')}<button class="qi-button" data-action="source" data-source="demo">Apri Demo</button></div></article>`:view==='home'?home():view==='analysis'?analysis():view==='chat'?chat():view==='activity'?activity():view==='stats'?stats():settings()}</section><footer class="qi-footer-note">${icon('shield-check')} Dati letti da MT5. Analisi ricostruita, senza accesso allo stato interno dell’EA. Nessun comando di trading dall’app.</footer></div><nav class="qi-nav" aria-label="Navigazione principale">${Object.entries(navs).map(([v,[label,i]])=>`<button data-action="nav" data-view="${v}" ${view===v?'aria-current="page"':''}>${icon(i)}<span>${label}</span></button>`).join('')}</nav></div><dialog class="qi-modal" aria-label="Dettaglio QUANTIX IQ"></dialog><div class="qi-toast" role="status" hidden></div>`;
    if(!dataAvailable&&view==='chat'){
      const section=root.querySelector('section');if(section)section.innerHTML=chat();
    }
    window.lucide?.createIcons({attrs:{width:20,height:20,'stroke-width':1.7}});drawCharts();window.QuantixPersonal?.mount(view,snapshot,announce);
  }
  async function refresh(force=false){
    if(busy||location.protocol==='file:')return;
    if(document.hidden&&!force)return;
    busy=true;
    try{
      const response=await fetch('/api/mt5',{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(12000)});
      if(!response.ok)throw new Error(response.status===403?'Accesso privato non autorizzato. Verifica Tailscale e l’account.':'Il servizio sul PC non risponde correttamente.');
      const value=await response.json();
      if(value.mode!=='mt5_read_only')throw new Error('Il servizio MT5 non è disponibile.');
      snapshot=value;networkError='';lastSuccess=Date.now();
      if(!privateUrl){const health=await fetch('/api/health',{cache:'no-store',signal:AbortSignal.timeout(4000)});if(health.ok)privateUrl=(await health.json()).private_url;}
      if(!aiInfo)fetch('/api/ai',{cache:'no-store',signal:AbortSignal.timeout(5000)}).then(r=>r.ok?r.json():null).then(info=>{if(info)aiInfo=info;}).catch(()=>{});
    }catch(error){networkError=error.name==='TimeoutError'?'Il PC non risponde. Verifica connessione e Tailscale.':error.message||'Collegamento non disponibile.';}
    finally{busy=false;const active=document.activeElement;if(!window.QuantixMarketChart?.interacting()&&!window.QuantixPersonal?.interacting()&&(force||!active?.matches('textarea,input,select')))announce();}
  }
  async function ask(question){
    const text=String(question??'').trim().slice(0,4000);if(!text||aiPending)return;
    const history=messages.filter(m=>!m.error).slice(-8).map(m=>({role:m.role==='user'?'user':'assistant',content:m.text}));
    messages.push({role:'user',text});if(messages.length>20)messages.splice(0,messages.length-20);
    chatDraft='';const field=document.getElementById('qi-live-question');if(field)field.value='';aiPending=true;aiPartial='';aiStarted=Date.now();announce();
    document.querySelector('.qi-ai-pending')?.scrollIntoView({behavior:'smooth',block:'nearest'});
    let pollFailed=0;
    try{
      const response=await fetch('/api/chat',{method:'POST',headers:{'Content-Type':'application/json'},credentials:'same-origin',cache:'no-store',body:JSON.stringify({question:text,history}),signal:AbortSignal.timeout(12000)});
      const job=await response.json();if(!response.ok)throw new Error(job.error||'Impossibile avviare la risposta AI.');
      aiInfo={available:true,model:job.model,local:true};
      while(Date.now()-aiStarted<140000){
        await new Promise(resolve=>setTimeout(resolve,450));
        let data;
        try{const r=await fetch('/api/chat/'+encodeURIComponent(job.job_id),{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(10000)});data=await r.json();if(!r.ok)throw new Error(data.error||'Risposta non disponibile.');pollFailed=0;}
        catch(error){if(++pollFailed<3)continue;throw error;}
        if(data.partial_text&&data.partial_text!==aiPartial){
          aiPartial=data.partial_text;
          const live=document.querySelector('.qi-ai-pending [data-ai-stream]');if(live)live.textContent=aiPartial;
          const note=document.querySelector('.qi-ai-pending [data-ai-writing]');if(note)note.textContent='In scrittura…';
        }
        if(data.status==='error')throw new Error(data.error);
        if(data.status==='done'){messages.push({role:'bot',text:data.result.text,model:data.result.model,observed:data.result.observed_utc,timing:data.result.timing,truncated:data.result.truncated});return;}
      }
      throw new Error('Il modello sta impiegando troppo tempo. Riprova con una domanda più breve.');
    }catch(error){messages.push({role:'bot',text:error.name==='TimeoutError'?'Il PC non risponde alla Chat. Controlla Tailscale e riprova.':error.message||'Risposta AI non disponibile.',error:true});}
    finally{aiPending=false;aiPartial='';announce();if(context?.view==='chat')document.querySelector('.qi-chat-log .qi-message:last-child')?.scrollIntoView({behavior:'smooth',block:'nearest'});}
  }
  document.addEventListener('toggle',event=>{const el=event.target;if(!el.matches?.('[data-position-details]'))return;const key=el.dataset.positionDetails;if(el.open)openPositionDetails.add(key);else openPositionDetails.delete(key);},true);
  document.addEventListener('input',event=>{if(event.target.id==='qi-live-question')chatDraft=event.target.value;});
  document.addEventListener('click',event=>{
    const question=event.target.closest('[data-live-question]');if(question){ask(question.dataset.liveQuestion);return;}
    const button=event.target.closest('[data-live-action]');if(!button)return;
    if(button.dataset.liveAction==='refresh')refresh(true);
    if(button.dataset.liveAction==='export'){
      if(!snapshot?.ok||!context)return;
      const rows=context.view==='stats'?[['giorno_italiano','netto_'+currency()],...snapshot.daily.map(r=>[r.date,r.net])]:[['orario_italiano','asset','direzione','volume','prezzo','netto_'+currency()],...snapshot.deals.map(r=>[r.time_italy,r.symbol,r.side,r.volume,r.price,r.net])];
      const csv='\ufeff'+rows.map(row=>row.map(v=>'"'+String(v).replace(/"/g,'""')+'"').join(';')).join('\r\n');
      const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download='QUANTIX_IQ_MT5_'+context.view+'.csv';a.click();setTimeout(()=>URL.revokeObjectURL(url),2000);
    }
  });
  document.addEventListener('submit',event=>{if(event.target.id==='qi-live-chat-form'){event.preventDefault();ask(new FormData(event.target).get('question'));}});
  addEventListener('online',()=>refresh(true));addEventListener('offline',()=>{networkError='Telefono o computer senza collegamento. I valori rimasti a video non sono attuali.';announce();});
  addEventListener('visibilitychange',()=>{if(!document.hidden)refresh(true);});
  let resizeTimer;addEventListener('resize',()=>{clearTimeout(resizeTimer);resizeTimer=setTimeout(drawCharts,150);});
  window.QuantixLive={enabled:()=>mode==='live',render,modeMarkup,setMode(value){mode=value==='live'?'live':'demo';try{localStorage.setItem('quantix-data-source',mode);}catch(_){}if(mode==='live')refresh(true);},refresh};
  setInterval(()=>{if(mode==='live')refresh();},5000);
  if(mode==='live')setTimeout(()=>refresh(true),0);
})();
