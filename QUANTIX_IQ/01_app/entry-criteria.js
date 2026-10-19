/* Diagnostica della Home. Nessun semaforo equivale a un ordine o a uno stato interno dell'EA. */
(function(){
  'use strict';
  const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const number=v=>new Intl.NumberFormat('it-IT',{maximumFractionDigits:2,minimumFractionDigits:2}).format(v);
  const labels={pass:'Soddisfatto',fail:'Non soddisfatto',pending:'In attesa',unknown:'Da verificare',off:'Non attivo',alternative:'Recupero ATR'};
  const symbols={pass:'✓',fail:'!',pending:'◷',unknown:'?',off:'—',alternative:'↗'};
  const sources={observed:'Dati MT5',profile:'Profilo salvato',bot_comment:'Esposto dal bot',reconstructed:'Ricostruito',internal:'Stato interno non esposto'};
  let chosenSession='auto',chosenSide='auto',lastSnapshot=null,lastStale=false;
  const opened=new Set();
  function effective(c,stale){return stale&&!['off','unknown'].includes(c.state)?{...c,state:'unknown',missing:'Attendere una lettura MT5 e quotazioni aggiornate prima di confermare questo criterio.'}:c;}
  function gapText(c){
    if(c.state==='pass'||c.state==='off')return '';
    if(c.gap==null||!Number.isFinite(c.gap))return '';
    if(c.gap<=0)return c.state==='pending'?'Resta da confermare alla chiusura della candela.':'';
    if(c.unit==='secondi'){const m=Math.ceil(c.gap/60);return 'Mancano '+m+' minuti.';}
    return 'Mancano '+number(c.gap)+' '+esc(c.unit||'')+'.';
  }
  function row(original,stale){
    const c=effective(original,stale),gap=gapText(c);
    return `<div class="qi-criterion" data-criterion="${esc(c.key)}" data-state="${esc(c.state)}"><div class="qi-criterion-symbol ${c.state}" aria-hidden="true">${symbols[c.state]}</div><div class="qi-criterion-content"><div class="qi-criterion-top"><h4>${esc(c.title)}</h4><span class="qi-criterion-status ${c.state}">${labels[c.state]}</span></div><p>${esc(c.value)}</p>${gap?`<strong class="qi-criterion-gap">${gap}</strong>`:''}${c.missing&&c.state!=='pass'&&c.state!=='off'?`<p class="qi-criterion-missing">${esc(c.missing)}</p>`:''}<span class="qi-criterion-source">${sources[c.state==='off'||c.key==='side'?'profile':c.source]||'Dati osservati'}</span></div></div>`;
  }
  function details(key,title,body){return `<details class="qi-criteria-details" data-criteria-details="${key}" ${opened.has(key)?'open':''}><summary>${title}</summary>${body}</details>`;}
  function markup(snapshot,stale=false){
    lastSnapshot=snapshot;lastStale=stale;
    const data=snapshot?.analysis?.entry_criteria;
    if(!data)return `<article class="qi-card qi-entry-criteria" aria-label="Criteri di entrata"><h2>Criteri di entrata</h2><p class="qi-inline-note">In attesa della prossima lettura dei criteri da MT5.</p></article>`;
    const sessions=data.sessions||[],session=sessions.find(s=>s.name===chosenSession)||sessions.find(s=>s.phase==='forming')||sessions.find(s=>s.phase==='searching')||sessions.find(s=>s.phase==='waiting')||sessions.at(-1);
    if(!session)return '';
    const old=stale||!data.fresh_quotes;
    const side=session.directions.find(d=>d.name===chosenSide)||session.directions.find(d=>d.checks.find(c=>c.key==='breakout')?.state==='pass')||session.directions[0];
    const checks=side.checks.map(c=>effective(c,old)),common=session.common.map(c=>effective(c,old)),all=[...common,...checks];
    const missing=all.filter(c=>c.state==='fail'||c.state==='pending');
    const unknown=all.filter(c=>c.state==='unknown'),passed=all.filter(c=>c.state==='pass');
    const priority=['time','capital','slots','risk','breakout','body','direction','deferred','daily','bias','stop','lot','margin'];
    missing.sort((a,b)=>priority.indexOf(a.key)-priority.indexOf(b.key));
    const title=old?'Lettura non aggiornata':missing.length?'Cosa manca per '+side.name:'Condizioni osservate favorevoli · conferma EA necessaria';
    const summary=old?'I criteri non vengono confermati usando prezzi o letture non aggiornati.':missing.length?missing.slice(0,3).map(c=>c.title+(c.gap>0&&c.unit&&c.unit!=='secondi'?' · '+number(c.gap)+' '+c.unit+' mancanti':'')).join(' • '):'I controlli disponibili passano. Restano da verificare i blocchi interni prima di un eventuale ordine.';
    const coreKeys=['breakout','body','direction','deferred','daily','bias'];
    const commonKeys=['time','atr','slots','risk'];
    const clock=t=>new Date(t).toLocaleTimeString('it-IT',{timeZone:'Europe/Rome',hour:'2-digit',minute:'2-digit'});
    const candle=data.candle_time?clock(data.candle_time)+'–'+clock(new Date(Date.parse(data.candle_time)+300000).toISOString()):'non disponibile';
    return `<article class="qi-card qi-entry-criteria" aria-label="Criteri di entrata" data-entry-session="${esc(session.name)}" data-entry-side="${side.name}"><div class="qi-section-head"><div><span class="qi-overline">Il prossimo ingresso</span><h2>Criteri di entrata</h2></div><span class="qi-pill">BOT · M5</span></div><div class="qi-criteria-tabs" role="group" aria-label="Sessione dei criteri">${['auto',...sessions.map(s=>s.name)].map(name=>`<button type="button" data-criteria-session="${name}" aria-pressed="${chosenSession===name}">${name==='auto'?'Attuale':name}</button>`).join('')}</div><p class="qi-criteria-context"><strong>${esc(session.name)}</strong> · Ultima M5 chiusa ${candle} IT · prossima chiusura fra ${Math.floor(data.seconds_to_next_close/60)}m ${data.seconds_to_next_close%60}s</p><div class="qi-criteria-summary ${old?'stale':''}" role="status"><strong>${esc(title)}</strong><p>${esc(summary)}</p><div class="qi-criteria-counts"><span>${passed.length} soddisfatti</span><span>${missing.length} da soddisfare</span><span>${unknown.length} da verificare</span></div></div><p class="qi-criteria-note">Un criterio verde è soddisfatto sui dati indicati. Non conferma un ordine: i contatori e i blocchi interni dell’EA non sono tutti esposti. Gli importi per un nuovo ingresso sono stime sul prezzo attuale.</p><div class="qi-criteria-columns"><div><h3>Condizioni della sessione</h3>${session.common.filter(c=>commonKeys.includes(c.key)).map(c=>row(c,old)).join('')}${details('common','Altri controlli della sessione',session.common.filter(c=>!commonKeys.includes(c.key)).map(c=>row(c,old)).join(''))}</div><div><div class="qi-criteria-direction-head"><h3>Segnale ${side.name}</h3><div class="qi-segment" role="group" aria-label="Direzione dei criteri">${session.directions.map(d=>`<button type="button" data-criteria-side="${d.name}" aria-pressed="${side.name===d.name}">${d.name}</button>`).join('')}</div></div>${side.checks.filter(c=>coreKeys.includes(c.key)).map(c=>row(c,old)).join('')}${details('broker','Lotto minimo, stop e margine',side.checks.filter(c=>!coreKeys.includes(c.key)).map(c=>row(c,old)).join(''))}</div></div>${details('pyramid','Ingressi aggiuntivi: piramidazione',`<p class="qi-inline-note">${esc(data.pyramid.missing)}</p><p class="qi-inline-note">Soglia configurata: +${number(data.pyramid.trigger_r)}R. ${esc(data.pyramid.note)}</p><p class="qi-inline-note">Il rischio iniziale e i flag della piramide non sono esposti: la distanza esatta non viene dedotta dallo stop attuale, che può essere già stato spostato.</p>`)}<p class="qi-criteria-foot">Rilevazione ${esc(snapshot.observed_italy?.slice(11,19))} IT · profilo MT5 salvato · ${session.atr_route==='deferred'?'percorso ATR differito':session.atr_route==='ordinary'?'percorso ordinario':'percorso ATR non disponibile'} · prezzi del broker. Le soglie di prezzo si riferiscono a XAUUSD, gli importi monetari a ${esc(snapshot.account.currency)}.</p></article>`;
  }
  document.addEventListener('click',event=>{
    const session=event.target.closest('[data-criteria-session]'),side=event.target.closest('[data-criteria-side]');if(!session&&!side)return;
    if(session)chosenSession=session.dataset.criteriaSession;if(side)chosenSide=side.dataset.criteriaSide;
    const card=document.querySelector('.qi-entry-criteria');if(card&&lastSnapshot)card.outerHTML=markup(lastSnapshot,lastStale);
  });
  document.addEventListener('toggle',event=>{const d=event.target;if(!d.matches?.('[data-criteria-details]'))return;if(d.open)opened.add(d.dataset.criteriaDetails);else opened.delete(d.dataset.criteriaDetails);},true);
  window.QuantixEntryCriteria={markup};
})();
