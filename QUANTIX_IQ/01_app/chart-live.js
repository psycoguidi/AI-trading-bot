
/* Candele MT5 reali, storico paginato e navigazione touch. Il bot resta su M5. */
(function(){
  'use strict';
  const n=v=>new Intl.NumberFormat('it-IT',{minimumFractionDigits:2,maximumFractionDigits:2}).format(v);
  const e=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const frames={M1:60,M5:300,M15:900,M30:1800,H1:3600,H4:14400,D1:86400},cache={};
  let selectedSession='auto',tf='M5',windowSize=48,fitPositions=false,lastData=null,anchor=null,anchorRaw=null,drag=null,suppressClick=0;
  let requestQueue=Promise.resolve();
  const state=()=>cache[tf]||(cache[tf]={bars:[],loading:false,error:'',hasMore:true,latestAt:0});
  const duration=size=>{const m=size*frames[tf]/60;return m<60?m+' min':m<1440?n(m/60).replace(',00','')+' ore':n(m/1440).replace(',00','')+' giorni';};
  const stamp=b=>b.time_italy;
  function merge(s,bars){const map=new Map(s.bars.map(b=>[stamp(b),b]));bars.forEach(b=>map.set(stamp(b),{...map.get(stamp(b)),...b}));s.bars=[...map.values()].sort((a,b)=>Date.parse(stamp(a))-Date.parse(stamp(b)));}
  async function load(which='latest'){
    const target=tf,s=state();if(s.loading)return false;
    s.loading=true;s.error='';
    const query=new URLSearchParams({tf:target,count:'300'});
    if(which==='older'&&s.bars[0]?.time_raw)query.set('before',s.bars[0].time_raw);
    if(which==='anchor'&&anchorRaw)query.set('before',anchorRaw+1);
    redraw(false);
    try{
      const task=requestQueue.then(async()=>{
        for(let attempt=0;attempt<3;attempt++){
          const response=await fetch('/api/chart?'+query,{cache:'no-store',credentials:'same-origin',signal:AbortSignal.timeout(28000)});
          const data=await response.json();
          if(response.status===429&&attempt<2){await new Promise(resolve=>setTimeout(resolve,1000));continue;}
          return {response,data};
        }
      });
      requestQueue=task.catch(()=>{});
      const {response,data}=await task;if(!response.ok||!data.ok)throw new Error(data.error||'Storico non disponibile.');
      if(data.timeframe!==target)throw new Error('Timeframe non corrispondente.');
      if(which!=='older'&&data.candles.length&&s.bars.length){
        const keys=new Set(data.candles.map(stamp));
        // Due intervalli senza sovrapposizione non vanno uniti: mancherebbero le candele intermedie.
        if(!s.bars.some(b=>keys.has(stamp(b)))){s.bars=[];s.hasMore=data.has_more;}
      }
      merge(s,data.candles);if(which!=='latest'||!s.latestAt)s.hasMore=data.has_more;
      if(which==='latest'||!s.latestAt)s.latestAt=Date.now();
      if(!data.candles.length&&which!=='latest')s.hasMore=false;
      return true;
    }catch(error){s.error=error.name==='TimeoutError'?'MT5 sta caricando lo storico. Premi Riprova.':error.message;return false;}
    finally{s.loading=false;if(tf===target)redraw(false);}
  }
  function activeSession(data){const ss=data?.analysis?.sessions||[];return ss.find(s=>s.name===selectedSession)||ss.find(s=>s.phase==='forming')||ss.find(s=>s.phase==='searching')||ss.filter(s=>s.phase==='ended').at(-1)||ss[0];}
  function markup(id){return `<div class="qi-chart-controls qi-market-controls"><strong class="qi-small" data-market-title>XAUUSD · ${tf} · MT5</strong><span class="qi-small qi-muted">Bot operativo su M5</span></div><div class="qi-market-timeframes" role="group" aria-label="Timeframe del grafico">${Object.keys(frames).map(t=>`<button type="button" data-market-tf="${t}" aria-pressed="${tf===t}">${t}</button>`).join('')}</div><div class="qi-chart-controls"><div class="qi-segment" role="group" aria-label="Ampiezza del grafico">${[24,48,96].map(size=>`<button type="button" data-market-window="${size}" aria-pressed="${windowSize===size}">${duration(size)}</button>`).join('')}</div></div><div class="qi-market-session-controls" role="group" aria-label="Sessione ORB mostrata">${['auto','Asia','Londra','New York'].map(name=>`<button type="button" data-market-session="${name}" aria-pressed="${selectedSession===name}">${name==='auto'?'Attuale':name}</button>`).join('')}</div><div class="qi-market-prices" data-market-prices></div><div class="qi-chart qi-market-chart" id="${id}" data-live-chart tabindex="0" role="group" aria-label="Grafico interattivo: trascina per lo storico, frecce per scorrere"></div><div class="qi-market-navigation"><button type="button" class="qi-button ghost" data-market-pan="back" aria-label="Candele precedenti">← Indietro</button><button type="button" class="qi-button ghost" data-market-pan="forward" aria-label="Candele successive">Avanti →</button><button type="button" class="qi-button" data-market-latest>Torna in diretta</button></div><div class="qi-market-history-status qi-small" data-market-history-status role="status"></div><div class="qi-market-caption" data-market-caption></div><div class="qi-chart-controls"><button class="qi-button ghost" type="button" data-market-fit aria-pressed="${fitPositions}">Mostra tutti gli SL/TP</button><span class="qi-small qi-muted">Trascina a destra per tornare indietro · tocca una candela</span></div><div class="qi-market-detail qi-small" data-market-detail aria-live="polite"></div><div class="qi-market-positions" data-market-positions></div>`;}
  function viewBars(s){let end=s.bars.length-1;if(anchor)end=s.bars.findLastIndex(b=>Date.parse(stamp(b))<=Date.parse(anchor));return {end,bars:s.bars.slice(Math.max(0,end-windowSize+1),end+1)};}
  function draw(el,data,allowLoad=true){
    lastData=data;if(!data?.ok)return;
    const s=state();
    if(tf==='M5'){
      const incoming=data.candles||[],keys=new Set(incoming.map(stamp)),overlap=s.bars.some(b=>keys.has(stamp(b)));
      if(!s.bars.length||overlap)merge(s,incoming);
      else if(!anchor){s.bars=[];s.latestAt=0;s.hasMore=true;merge(s,incoming);}
    }
    const wrap=el.parentElement,{bars}=viewBars(s),session=activeSession(data),quote=data.quote||{},positions=(data.positions||[]).filter(p=>p.symbol===data.bot.symbol);
    wrap.querySelector('[data-market-title]').textContent=`${data.bot.symbol} · ${tf} · MT5`;
    wrap.querySelectorAll('[data-market-tf]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.marketTf===tf)));
    wrap.querySelectorAll('[data-market-window]').forEach(b=>{b.setAttribute('aria-pressed',String(+b.dataset.marketWindow===windowSize));b.textContent=duration(+b.dataset.marketWindow);});
    wrap.querySelectorAll('[data-market-session]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.marketSession===selectedSession)));
    wrap.querySelector('[data-market-latest]').setAttribute('aria-pressed',String(!anchor));
    const status=wrap.querySelector('[data-market-history-status]');
    status.innerHTML=`<span class="${anchor?'':'qi-positive'}">${anchor?'Storico · aggiornamento del prezzo separato':'● Segui il mercato'}</span> · ${s.loading?'Caricamento candele MT5…':s.error?e(s.error):!s.hasMore?'Inizio dello storico disponibile nel terminale':s.bars.length+' candele caricate'}${s.error?' <button type="button" class="qi-button ghost" data-market-retry>Riprova</button>':''}`;
    wrap.querySelector('[data-market-prices]').innerHTML=`<span>Bid ora <strong>${n(quote.bid)}</strong></span><span>Ask ora <strong>${n(quote.ask)}</strong></span><span class="qi-muted">Spread ${n(quote.ask-quote.bid)}</span><span class="qi-muted">${e(quote.time_italy?.slice(11,19))} IT</span>`;
    if(allowLoad&&!s.loading&&!s.error&&(!s.latestAt||!anchor&&tf!=='M5'&&Date.now()-s.latestAt>15000))load(anchor?'anchor':'latest');
    if(!bars.length){el.innerHTML='<p class="qi-inline-note">'+(s.loading?'Caricamento dello storico…':s.error?e(s.error):'Candele non disponibili per questo intervallo.')+'</p>';return;}
    const exact=session?.level_source==='bot_comment'&&Number.isFinite(session.bot_high)&&Number.isFinite(session.bot_low);
    const high=exact?session.bot_high:session?.high,low=exact?session.bot_low:session?.low;
    const today=data.observed_italy?.slice(0,10),currentLayers=!anchor||bars.at(-1).time_italy.slice(0,10)>=today;
    const levels=currentLayers&&high>low&&low>0&&!['waiting','disabled'].includes(session?.phase);
    const w=Math.max(220,el.clientWidth),h=w<500?330:410,right=w<500?65:78,left=5,top=48,bottom=32,pw=w-left-right,ph=h-top-bottom;
    const values=bars.flatMap(b=>[b.high,b.low]).filter(Number.isFinite);
    if(currentLayers){values.push(quote.bid,quote.ask);if(levels)values.push(high,low);}
    if(fitPositions&&currentLayers)positions.forEach(p=>{values.push(p.entry_price);if(p.sl>0)values.push(p.sl);if(p.tp>0)values.push(p.tp);});
    let min=Math.min(...values),max=Math.max(...values);const pad=Math.max(.3,(max-min)*.12);min-=pad;max+=pad;
    const y=p=>top+(max-p)/(max-min)*ph,x=i=>left+(i+.5)*pw/bars.length,bw=Math.max(1.4,pw/bars.length*.58),visible=p=>p>=min&&p<=max;
    let marks='';
    for(let i=0;i<5;i++){const v=min+(max-min)*i/4;marks+=`<line class="grid" x1="${left}" x2="${w-right}" y1="${y(v)}" y2="${y(v)}"/>`;if(!currentLayers||Math.abs(y(v)-y(quote.bid))>19)marks+=`<text x="${w-right+7}" y="${y(v)+4}">${n(v)}</text>`;}
    if(levels){marks+=`<rect class="qi-orb-band" x="${left}" y="${y(high)}" width="${pw}" height="${Math.max(1,y(low)-y(high))}"/>`;for(const [label,p] of [['MAX',high],['MIN',low]])marks+=`<line class="qi-orb-line ${exact?'exact':'estimated'}" data-orb-level="${label}" data-level-source="${exact?'bot_comment':'reconstructed'}" data-price="${p}" x1="${left}" x2="${w-right}" y1="${y(p)}" y2="${y(p)}"/>`;marks+=`<text class="qi-chart-range-label" x="${left}" y="17">${e(session.name)} · ${exact?'livelli del bot':'livelli ricostruiti'} · oggi</text><text class="qi-chart-range-label" x="${left}" y="34">MAX ${n(high)} · MIN ${n(low)}${session.phase==='forming'?' · provvisori':''}</text>`;}
    else marks+=`<text x="${left}" y="22">${currentLayers?e(session?.name||'ORB')+' · range non disponibile':'Storico · livelli e posizioni odierni nascosti'}</text>`;
    bars.forEach((b,i)=>{const cls=b.close>=b.open?'up':'down';marks+=`<line class="${cls}" x1="${x(i)}" x2="${x(i)}" y1="${y(b.high)}" y2="${y(b.low)}"/><rect class="${cls}" x="${x(i)-bw/2}" y="${Math.min(y(b.open),y(b.close))}" width="${bw}" height="${Math.max(1.3,Math.abs(y(b.open)-y(b.close)))}" stroke-width="0"/>`;});
    const tradeLines=[];
    if(currentLayers)positions.forEach((p,index)=>{
      for(const [type,price] of [['entry',p.entry_price],['sl',p.sl],['tp',p.tp]]){if(!(price>0))continue;const duplicate=tradeLines.find(l=>l.type===type&&Math.abs(l.price-price)<.03);if(duplicate)duplicate.labels.push(index+1);else tradeLines.push({type,price,labels:[index+1]});}
      const when=Date.parse(p.time_italy),barIndex=bars.findIndex(b=>Date.parse(b.time_italy)+frames[tf]*1000>when);
      if(barIndex>=0&&when>=Date.parse(bars[0].time_italy)&&visible(p.entry_price)){const yy=y(p.entry_price),xx=x(barIndex),buy=p.side==='BUY';marks+=`<path class="qi-entry-marker ${buy?'buy':'sell'}" d="M ${xx-5} ${yy+(buy?8:-8)} L ${xx} ${yy} L ${xx+5} ${yy+(buy?8:-8)} Z"/>`;}
    });
    tradeLines.forEach(line=>{if(visible(line.price))marks+=`<line class="qi-position-line ${line.type}" data-position-level="${line.type}" data-price="${line.price}" x1="${left}" x2="${w-right}" y1="${y(line.price)}" y2="${y(line.price)}"/>`;});
    if(currentLayers)marks+=`<line class="qi-current-price" data-current-bid="${quote.bid}" x1="${left}" x2="${w-right}" y1="${y(quote.bid)}" y2="${y(quote.bid)}"/><rect class="qi-price-tag" x="${w-right+2}" y="${y(quote.bid)-10}" width="${right-3}" height="20" rx="4"/><text class="qi-price-tag-text" x="${w-right+7}" y="${y(quote.bid)+4}">${n(quote.bid)}</text>`;
    for(const i of [0,Math.floor(bars.length/2),bars.length-1]){const b=bars[i],label=tf==='D1'?b.time_italy.slice(5,10).split('-').reverse().join('/'):b.time_italy.slice(11,16);marks+=`<text x="${x(i)}" y="${h-5}" text-anchor="${i===0?'start':i===bars.length-1?'end':'middle'}">${e(label)}</text>`;}
    bars.forEach((b,i)=>marks+=`<rect class="qi-hit" data-market-candle="${i}" x="${x(i)-pw/bars.length/2}" y="${top}" width="${pw/bars.length}" height="${ph}"/>`);
    el.innerHTML=`<svg viewBox="0 0 ${w} ${h}" role="img" aria-label="Candele ${tf} MT5 di XAUUSD, ${anchor?'storico':'ultimi prezzi'}, trascinabili">${marks}</svg>`;
    el.querySelector('svg').style.height=h+'px';el._marketBars=bars;el._plotWidth=pw;el.dataset.timeframe=tf;el.dataset.viewEnd=bars.at(-1).time_italy;el.dataset.followLive=String(!anchor);
    const caption=wrap.querySelector('[data-market-caption]');caption.innerHTML=`<span class="qi-market-key orb">${exact?'Range del bot MT5':'Range ricostruito'}</span><span class="qi-market-key entry">Entrata</span><span class="qi-market-key sl">Stop loss</span><span class="qi-market-key tp">Take profit</span><p>${currentLayers?(exact?'Massimo e minimo letti dal commento del bot sul grafico MT5.':'Range ricostruito dalle candele M1; il bot non espone valori validi per questa sessione.'):'Lo storico mostra le candele effettive del broker. I livelli ORB e le posizioni attuali non sono attribuiti ai giorni precedenti.'} Prezzi aggiornati periodicamente, non tick per tick.</p>`;
    const format=t=>new Date(t).toLocaleString('it-IT',{timeZone:'Europe/Rome',day:'2-digit',month:'2-digit',year:'numeric',hour:'2-digit',minute:'2-digit'});
    wrap.querySelector('[data-market-detail]').textContent=`${format(bars[0].time_italy)} → ${format(bars.at(-1).time_italy)} IT · ${bars.length} candele ${tf}`;
    wrap.querySelector('[data-market-positions]').innerHTML=positions.length?positions.map((p,i)=>`<div class="qi-chart-position"><strong>${p.side} ${n(p.volume)} <span class="qi-muted">#${i+1} · aperta ora</span></strong><span class="${p.profit>=0?'qi-positive':'qi-negative'}">${new Intl.NumberFormat('it-IT',{style:'currency',currency:data.account.currency}).format(p.profit)}</span><div><span class="qi-trade-entry">Entrata ${n(p.entry_price)}</span><span class="qi-trade-sl">SL ${p.sl?n(p.sl):'assente'}${p.sl&&(!currentLayers||!visible(p.sl))?' · fuori vista':''}</span><span class="qi-trade-tp">TP ${p.tp?n(p.tp):'assente'}${p.tp&&(!currentLayers||!visible(p.tp))?' · fuori vista':''}</span></div></div>`).join(''):'<p class="qi-inline-note">Nessuna posizione XAUUSD aperta in MT5.</p>';
    const fit=wrap.querySelector('[data-market-fit]');fit.setAttribute('aria-pressed',String(fitPositions));fit.textContent=fitPositions?'Vista vicina al prezzo':'Mostra tutti gli SL/TP';fit.disabled=!currentLayers;
  }
  function redraw(allowLoad=true){if(lastData)document.querySelectorAll('#quantix-app [data-live-chart]').forEach(el=>draw(el,lastData,allowLoad));}
  function setEnd(index){const s=state();index=Math.max(Math.min(windowSize-1,s.bars.length-1),Math.min(s.bars.length-1,index));const b=s.bars[index];if(!b)return;anchor=index===s.bars.length-1?null:b.time_italy;anchorRaw=anchor?b.time_raw:null;redraw(false);}
  async function pan(direction){const s=state(),saved=anchor||s.bars.at(-1)?.time_italy,amount=Math.max(1,Math.floor(windowSize/2));let end=viewBars(s).end;if(direction>0&&end-amount<windowSize&&s.hasMore)await load('older');end=s.bars.findLastIndex(b=>Date.parse(stamp(b))<=Date.parse(saved));setEnd(end-direction*amount);}
  function latest(){anchor=null;anchorRaw=null;redraw(false);if(!state().loading)load('latest');}
  document.addEventListener('click',event=>{
    if(Date.now()<suppressClick&&event.target.closest('[data-live-chart]'))return;
    const t=event.target.closest('[data-market-tf]');if(t){tf=t.dataset.marketTf;redraw(false);const s=state();if(!s.loading)load(anchor?'anchor':'latest');return;}
    const session=event.target.closest('[data-market-session]');if(session){selectedSession=session.dataset.marketSession;redraw();return;}
    const win=event.target.closest('[data-market-window]');if(win){windowSize=+win.dataset.marketWindow;redraw();return;}
    const p=event.target.closest('[data-market-pan]');if(p){pan(p.dataset.marketPan==='back'?1:-1);return;}
    if(event.target.closest('[data-market-latest]')){latest();return;}
    if(event.target.closest('[data-market-retry]')){load(anchor?'anchor':'latest');return;}
    if(event.target.closest('[data-market-fit]')){fitPositions=!fitPositions;redraw();return;}
    const target=event.target.closest('[data-market-candle]');if(target){const el=target.closest('[data-live-chart]'),bar=el?._marketBars?.[+target.dataset.marketCandle];if(bar)el.parentElement.querySelector('[data-market-detail]').textContent=`${bar.time_italy.slice(0,16).replace('T',' ')} IT · O ${n(bar.open)} · H ${n(bar.high)} · L ${n(bar.low)} · C ${n(bar.close)}`;}
  });
  document.addEventListener('pointerdown',event=>{const el=event.target.closest('[data-live-chart]');if(el&&event.button===0){drag={el,id:event.pointerId,x:event.clientX,y:event.clientY,end:state().bars[viewBars(state()).end]?.time_italy,moved:false};}});
  document.addEventListener('pointermove',event=>{
    if(!drag||event.pointerId!==drag.id)return;const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
    if(!drag.moved&&(Math.abs(dx)<7||Math.abs(dx)<Math.abs(dy)*1.2))return;
    if(!drag.moved){drag.moved=true;try{drag.el.setPointerCapture(event.pointerId);}catch(_){}drag.el.classList.add('dragging');}
    if(event.cancelable)event.preventDefault();const s=state(),base=s.bars.findLastIndex(b=>Date.parse(stamp(b))<=Date.parse(drag.end));
    const next=base-Math.round(dx/(drag.el._plotWidth||400)*windowSize);setEnd(next);
    if(next<windowSize&&s.hasMore&&!s.loading&&!s.error)load('older');
  },{passive:false});
  const endDrag=event=>{if(!drag||event.pointerId!==drag.id)return;if(drag.moved)suppressClick=Date.now()+350;drag.el.classList.remove('dragging');drag=null;};
  document.addEventListener('pointerup',endDrag);document.addEventListener('pointercancel',endDrag);
  document.addEventListener('keydown',event=>{if(!event.target.matches('[data-live-chart]'))return;if(['ArrowLeft','ArrowRight','End','+','-'].includes(event.key)){event.preventDefault();if(event.key==='End')latest();else if(event.key==='ArrowLeft'||event.key==='ArrowRight')pan(event.key==='ArrowLeft'?1:-1);else{windowSize=event.key==='+'?Math.max(12,windowSize/2):Math.min(192,windowSize*2);redraw();}}});
  window.QuantixMarketChart={markup,draw,interacting:()=>Boolean(drag)};
  const root=document.getElementById('quantix-app');if(root&&typeof ResizeObserver!=='undefined'){let width=0;new ResizeObserver(entries=>{const next=Math.round(entries[0].contentRect.width);if(next!==width){width=next;redraw();}}).observe(root);}
})();
