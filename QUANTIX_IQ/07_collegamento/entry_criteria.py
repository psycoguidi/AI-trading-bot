"""Diagnostica leggibile degli ingressi: osservazioni e ricostruzioni, mai segnali eseguibili."""
from datetime import datetime, timezone
import math


def flag(p, key, default=False):
    return str(p.get(key, str(default))).lower() in ('true', '1')


def num(p, key, default):
    return float(p.get(key, default))


def fmt(value, decimals=2):
    return f'{value:,.{decimals}f}'.replace(',', '_').replace('.', ',').replace('_', '.')


def item(key, title, state, value, missing='', source='observed', gap=None, unit=None):
    return {'key': key, 'title': title, 'state': state, 'value': value, 'missing': missing,
            'source': source, 'gap': gap, 'unit': unit}


def atr_closed(d1, period):
    """SMA dei True Range: formula dell'esempio MetaQuotes MQL5/Indicators/Examples/ATR.mq5."""
    rows = list(d1)[:-1] if d1 is not None else []
    if len(rows) < period + 1:
        return None
    trs = [max(float(rows[i]['high']), float(rows[i-1]['close'])) - min(float(rows[i]['low']), float(rows[i-1]['close'])) for i in range(1, len(rows))]
    return sum(trs[-period:]) / period


def risk_observations(api, account, positions, orders, params, symbol, own, now_raw):
    """Calcoli monetari di sola lettura. I blocchi persistenti dell'EA non sono accessibili."""
    all_used = own_used = 0.0
    complete = True
    for p in positions:
        tick = api.symbol_info_tick(p.symbol)
        if not tick or p.sl <= 0:
            complete = False
            break
        buy = p.type == api.POSITION_TYPE_BUY
        result = api.order_calc_profit(api.ORDER_TYPE_BUY if buy else api.ORDER_TYPE_SELL, p.symbol, p.volume, tick.bid if buy else tick.ask, p.sl) if hasattr(api, 'order_calc_profit') else None
        if result is None or not math.isfinite(result):
            complete = False
            break
        loss = max(0.0, -result)
        all_used += loss
        if p.symbol == symbol and p.magic == int(params.get('InpMagicNumber', 0)):
            own_used += loss
    own_cap = account.equity * num(params, 'InpTotalOpenStopRiskPct', 4) / 100 - own_used
    info = api.symbol_info(symbol) if hasattr(api, 'symbol_info') else None
    return {'all_stop_risk': all_used if complete else None, 'own_stop_risk': own_used if complete else None,
            'own_capacity': max(0.0, own_cap) if complete else None,
            'pending': len(orders), 'position_count': len(positions),
            'min_lot': float(info.volume_min) if info else None,
            'tick_size': float(info.trade_tick_size) if info else .01,
            'min_stop': float(info.trade_stops_level * info.point) if info else None,
            'margin_free': account.margin_free,
            'hedging': getattr(account, 'margin_mode', None) == getattr(api, 'ACCOUNT_MARGIN_MODE_RETAIL_HEDGING', 2)}


def build_criteria(snapshot, params, raw_now, bars, d1, risk, api=None, extra_session=None):
    p=params; a=snapshot['account']; b=snapshot['bot']; q=snapshot['quote']; analysis=snapshot['analysis']
    closed=[r for r in (bars if bars is not None else []) if int(r['time'])+300<=raw_now]
    candle=closed[-1] if closed else None
    atr=atr_closed(d1,int(num(p,'InpATRPeriod',14)))
    own=b['entry_rules']['own_positions']; limit=b['entry_rules']['maximum_positions']; money=a['currency']
    reserve=num(p,'InpFTMOReservePerTradeEUR',5)*(risk['position_count']+1)
    capacity=None if risk['all_stop_risk'] is None else max(0,a['equity']-a['protection_estimate']-risk['all_stop_risk']-reserve)
    if risk['pending']:
        capacity=0
    budget=min(capacity,risk['own_capacity']) if capacity is not None and risk['own_capacity'] is not None else None
    fresh=q['age_seconds']<=180
    connection=snapshot['connection']
    allowed=all(connection.get(k) for k in ('connected','algo_enabled','account_trade_allowed','account_ea_allowed')) and b.get('ea_permission_from_profile',False)
    base=[item('permissions','Terminale e permessi','pass' if allowed and fresh else 'fail',
               'MT5 connesso e permessi generali abilitati.' if allowed else 'Un permesso o il collegamento non è disponibile.',
               '' if allowed and fresh else 'Servono collegamento al broker, prezzi recenti e permessi di trading abilitati.'),
          item('slots','Posizioni disponibili','pass' if own<limit else 'fail',f'{own}/{limit} posizioni proprie aperte.',
               '' if own<limit else 'Deve liberarsi almeno un posto.',gap=max(0,own-limit+1),unit='posizioni'),
          item('protection','Protezione del conto','pass' if a['equity']>a['protection_estimate'] else 'fail',
               f"Equity {fmt(a['equity'])} {money}; soglia stimata {fmt(a['protection_estimate'])} {money}.",
               '' if a['equity']>a['protection_estimate'] else 'Equity sopra la soglia e rimozione del blocco giornaliero da parte dell’EA; un semplice rimbalzo non basta.',
               'reconstructed',max(0,a['protection_estimate']-a['equity']),money),
          item('risk','Rischio residuo per nuovi ordini','unknown' if budget is None else 'pass' if budget>0 else 'fail',
               'Rischio non calcolabile: manca uno stop o un dato del broker.' if budget is None else f'{fmt(budget)} {money} stimati, dopo rischio agli stop e riserve.',
               'Servono stop validi su tutte le posizioni e dati completi.' if budget is None else 'Serve rischio disponibile; eventuali ordini pendenti bloccano nuovi ingressi.' if budget<=0 else '',
               'reconstructed',None if budget is None else max(0,-budget),money)]
    dt=datetime.fromtimestamp(raw_now,timezone.utc)
    rollover=dt.hour*60+dt.minute>=num(p,'InpRolloverCheckServerHour',22)*60+num(p,'InpRolloverCheckServerMinute',45)
    friday=dt.weekday()==4 and dt.hour>=num(p,'InpFridayFlatHour',17)
    if rollover or friday:
        base.append(item('close_window','Finestra di chiusura','fail','Il codice blocca i nuovi ingressi in questa fascia.', 'Attendere una nuova finestra operativa fuori dal rollover e dalla chiusura del venerdì.'))
    sessions=[]
    raw_day=raw_now//86400*86400
    source_sessions=list(analysis['sessions'])+([extra_session] if extra_session else [])
    for ss in source_sessions:
        name=ss['name']; extra=name=='Extra'; prefix={'Asia':'Asia','Londra':'Lon','New York':'NY','Extra':'Extra'}[name]
        common=[dict(x) for x in base]
        phase=ss['phase']; start=raw_day+int(num(p,'Inp'+prefix+'_StartHour' if not extra else 'InpExtraStartHour',{'Asia':1,'Lon':8,'NY':15,'Extra':16}[prefix]))*3600+int(num(p,'Inp'+prefix+'_StartMin' if not extra else 'InpExtraStartMinute',30 if prefix in ('NY','Extra') else 0))*60
        end=start+int(num(p,'Inp'+prefix+'_DurationMin' if not extra else 'InpExtraDurationMin',30 if prefix in ('NY','Extra') else 60))*60
        deadline=end+int(num(p,'InpExtraEntryWindowMin' if extra else 'InpMaxEntryDelayMin',120 if extra else 360))*60
        time_left=max(0,end-raw_now) if raw_now<end else 0
        common.insert(1,item('time','Finestra della sessione','pass' if phase=='searching' else 'pending' if phase in ('waiting','forming') else 'fail',
            f"Range {ss['start_italy'][11:16]}–{ss['end_italy'][11:16]}; ricerca fino alle {ss['deadline_italy'][11:16]} italiane.",
            f'Mancano {math.ceil(time_left/60)} minuti alla fine del range.' if time_left else 'Finestra terminata: attendere una nuova sessione.' if phase=='ended' else 'Sessione disabilitata.' if phase=='disabled' else '',gap=time_left,unit='secondi'))
        exact=ss.get('level_source')=='bot_comment'
        high=ss.get('bot_high') if exact else ss.get('high'); low=ss.get('bot_low') if exact else ss.get('low')
        valid=high is not None and low is not None and high>low>0
        complete=exact or ss.get('bars_available',0)>=ss.get('expected_bars',1)
        common.append(item('range','Massimo e minimo del range','pending' if phase in ('waiting','forming') else 'pass' if valid and complete else 'unknown',
            f'Massimo {fmt(high)} · minimo {fmt(low)}.' if valid else 'Range non disponibile.',
            'Attendere la fine della formazione del range.' if raw_now<end else 'Mancano candele M1 o livelli validi esposti dal bot.' if not valid or not complete else '',
            'bot_comment' if exact else 'reconstructed',max(0,ss.get('expected_bars',0)-ss.get('bars_available',0)) if not exact else 0,'candele M1'))
        if extra:
            capital=min(a['equity'],a['balance']); threshold=num(p,'InpExtraMinCapital',20000)
            common.append(item('capital','Capitale per la sessione extra','pass' if capital>=threshold else 'fail',f'{fmt(capital)} / {fmt(threshold)} {money}.',
                               '' if capital>=threshold else 'Sia saldo sia equity devono raggiungere la soglia.',gap=max(0,threshold-capital),unit=money))
        threshold_mult=num(p,{'Asia':'InpATRMinRangeMultAsia','Londra':'InpATRMinRangeMult','New York':'InpATRMinRangeMultNY','Extra':'InpExtraMinimumATR'}[name],.15)
        width=high-low if valid else None
        required=atr*threshold_mult if atr is not None else None
        initial=valid and required is not None and width>=required
        use_atr=flag(p,'InpUseATRFilter',True)
        common.append(item('atr','Volatilità del range · ATR D1','off' if not use_atr else 'unknown' if required is None or width is None else 'pass' if initial else 'alternative' if flag(p,'InpATRDeferredBreakout',True) and not extra else 'fail',
            'Filtro disabilitato.' if not use_atr else f'Range {fmt(width)} / minimo {fmt(required)}; ATR {fmt(atr)} × {fmt(threshold_mult)}.' if required is not None and width is not None else 'ATR o range non disponibili.',
            '' if not use_atr or initial else 'Il range iniziale è troppo stretto: vedere il recupero ATR nelle due direzioni.' if flag(p,'InpATRDeferredBreakout',True) and not extra else 'La soglia deve essere rispettata prima di un ingresso.',
            'reconstructed',max(0,required-width) if required is not None and width is not None else None,'di prezzo'))
        session_counter=(analysis.get('bot_observer') or {}).get('trades_today_from_comment')
        common.append(item('internal','Contatori e blocchi interni EA','unknown',
            f"Limiti: {int(num(p,'InpMaxTradesDay',12))} ingressi/giorno, {int(num(p,'InpMaxLossesDay',4))} perdite/giorno, {int(num(p,'InpExtraMaxEntries' if extra else 'InpSessionMaxEntries',1 if extra else 2))} ingressi/sessione. Contatore esposto: {session_counter if session_counter is not None else 'non disponibile'}.",
            f"Conferma interna necessaria: sessione non consumata, nessun blocco persistente e, per un rientro, ritorno nel range e oltre {int(num(p,'InpReentryCooldownMin',20))} minuti dall’ingresso precedente. Il tempo da solo non basta.",'internal'))
        directions=[]
        requested=a['equity']*num(p,'InpRiskPercent',2.5)/100 if flag(p,'InpUseEquityRisk',True) else num(p,'InpFixedRiskAmount',80)
        if flag(p,'InpGrowthRiskEnabled',False):
            cap=min(a['balance'],a['equity'])
            growth_pct=num(p,'InpGrowthLowRiskPct',1) if cap<num(p,'InpGrowthMiddleBalance',15000) else num(p,'InpGrowthMiddleRiskPct',2) if cap<num(p,'InpGrowthHighBalance',25000) else num(p,'InpGrowthHighRiskPct',3)
            requested=a['equity']*growth_pct/100
        requested=min(requested,risk['own_capacity']) if risk['own_capacity'] is not None else requested
        requested*=num(p,{'Asia':'InpAsiaRiskMultiplier','Londra':'InpLondonRiskMultiplier','New York':'InpNYRiskMultiplier','Extra':'InpExtraRiskMultiplier'}[name],1)
        if use_atr and not initial:requested*=num(p,'InpDeferredRiskMultiplier',.1)
        if min(a['balance'],a['equity'])<num(p,'InpFTMOInitialCapital',10000):requested=min(requested,num(p,'InpFTMOInitialCapital',10000)*num(p,'InpBelowInitialRiskPercent',1.25)/100)
        effective_budget=min(requested,capacity) if capacity is not None else None
        for direction,label in [(1,'LONG'),(-1,'SHORT')]:
            checks=[]; close=float(candle['close']) if candle is not None else None; op=float(candle['open']) if candle is not None else None
            border=high if direction==1 else low; tick_size=risk.get('tick_size') or .01
            beyond=valid and close is not None and (close>high if direction==1 else close<low)
            target=border+direction*tick_size if valid else None
            gap=max(0,direction*(target-close)) if target is not None and close is not None else None
            checks.append(item('breakout','Chiusura M5 oltre il range','unknown' if close is None or not valid else 'pass' if beyond else 'pending',
                f'Ultima chiusura {fmt(close)}; serve {"sopra" if direction==1 else "sotto"} {fmt(border)}.' if close is not None and valid else 'Manca una candela chiusa o un range valido.',
                '' if beyond else f'Serve una chiusura M5 {"sopra" if direction==1 else "sotto"} il livello, non basta una rottura durante la candela.',
                'bot_comment' if exact else 'reconstructed',0 if beyond else gap,'di prezzo'))
            span=float(candle['high']-candle['low']) if candle is not None else 0
            body=abs(close-op)/span*100 if span>0 else None; minimum=num(p,'InpBodyFilterPct',25)
            body_ok=body is not None and body>minimum
            checks.append(item('body','Corpo della candela M5','unknown' if body is None else 'pass' if body_ok else 'pending',f'{fmt(body)}% del range candela; richiesto oltre {fmt(minimum)}%.' if body is not None else 'Corpo non calcolabile.',
                '' if body_ok else 'La candela deve chiudere con un corpo superiore alla soglia. È valutata nuovamente alla chiusura M5.',gap=max(0,minimum-body) if body is not None else None,unit='punti percentuali'))
            directional=close is not None and direction*(close-op)>0; enforce=flag(p,'InpRequireDirectionalBody',True)
            checks.append(item('direction','Direzione della candela','off' if not enforce else 'unknown' if close is None else 'pass' if directional else 'pending',
                f'Per {label} serve una candela {"rialzista" if direction==1 else "ribassista"}.' if enforce else 'Controllo disabilitato.',
                '' if directional or not enforce else 'Attendere una candela M5 chiusa nella direzione richiesta.'))
            deferred=use_atr and not initial
            if deferred:
                enabled=not extra and flag(p,'InpATRDeferredBreakout',True) and (int(num(p,'InpDeferredSessions',7)) & {'Asia':1,'Londra':2,'New York':4,'Extra':0}[name])!=0
                deferred_target=(low+required if direction==1 else high-required) if valid and required is not None else None
                if valid and deferred_target is not None: deferred_target=max(high+tick_size,deferred_target) if direction==1 else min(low-tick_size,deferred_target)
                post_range=candle is not None and int(candle['time'])>=end
                reached=beyond and required is not None and ((close-low if direction==1 else high-close)+1e-9>=required)
                flat_ok=not flag(p,'InpDeferredOnlyWhenFlat',True) or own==0
                extension=direction*(close-border) if close is not None and valid else None
                ext_limit=num(p,'InpDeferredMaxExtension',1)*width if valid and num(p,'InpDeferredMaxExtension',1)>0 else None
                extension_ok=extension is not None and (ext_limit is None or extension<=ext_limit)
                deferred_dir=not flag(p,'InpDeferredDirectionalBody',True) or directional
                ready=enabled and post_range and reached and flat_ok and extension_ok and deferred_dir
                reason=[]
                if not enabled:reason.append('Recupero ATR non abilitato per questa sessione.')
                if not post_range:reason.append('Serve una M5 interamente successiva alla formazione del range.')
                if not reached and deferred_target is not None:reason.append(f'Attendere una chiusura {"almeno a" if direction==1 else "non sopra"} {fmt(deferred_target)}.')
                if not flat_ok:reason.append(f'Devono chiudersi tutte le {own} posizioni proprie aperte.')
                if extension is not None and ext_limit is not None and extension>ext_limit:reason.append(f'Breakout troppo esteso: {fmt(extension)} contro massimo {fmt(ext_limit)} di prezzo.')
                if not deferred_dir:reason.append('La candela deve essere direzionale.')
                retest=flag(p,'InpDeferredRetest',False)
                if retest:reason.append('Il successivo retest richiede una conferma interna non esposta.')
                checks.append(item('deferred','Recupero della stessa soglia ATR','unknown' if required is None or retest else 'pass' if ready else 'fail' if not enabled or not flat_ok or extension is not None and ext_limit is not None and extension>ext_limit else 'pending',
                    'Percorso differito: stessa soglia ATR, esposizione ridotta.' if enabled else 'Percorso differito non disponibile.', ' '.join(reason),'reconstructed',
                    max(0,direction*(deferred_target-close)) if deferred_target is not None and close is not None else None,'di prezzo'))
            daily_enabled=flag(p,'InpDailyFilterLong' if direction==1 else 'InpDailyFilterShort',direction==-1)
            ema=analysis.get('ema20_reconstructed'); prev=analysis.get('previous_daily_close')
            daily_ok=prev is not None and ema is not None and direction*(prev-ema)>0
            checks.append(item('daily','Filtro giornaliero EMA','off' if not daily_enabled else 'unknown' if ema is None or prev is None else 'pass' if daily_ok else 'fail',
                'Filtro disabilitato per questa direzione.' if not daily_enabled else f'Chiusura D1 {fmt(prev)} / EMA {fmt(ema)}.' if prev is not None and ema is not None else 'Dati D1 insufficienti.',
                '' if not daily_enabled or daily_ok else f'Serve una chiusura giornaliera {"sopra" if direction==1 else "sotto"} l’EMA. Il solo movimento intraday non modifica questa condizione.',
                'reconstructed',max(0,-direction*(prev-ema)) if ema is not None and prev is not None else None,'di prezzo'))
            bias_enabled=not extra and flag(p,'Inp'+prefix+'_UseDailyFilter',True) and not (name=='Londra' and flag(p,'InpLon_IgnoreBias',True))
            bias=(analysis.get('bot_observer') or {}).get('bias')
            bias_ok=bias==('BULL' if direction==1 else 'BEAR') if name=='Londra' else bias!=('BEAR' if direction==1 else 'BULL')
            checks.append(item('bias','Bias della sessione','off' if not bias_enabled else 'unknown' if bias is None else 'pass' if bias_ok else 'fail',
                'Filtro disabilitato per questa sessione.' if not bias_enabled else 'Bias esposto dal bot: '+str(bias or 'non disponibile')+'.',
                '' if not bias_enabled or bias is not None and bias_ok else 'Serve un bias compatibile esposto dal bot; non è possibile quantificare una distanza di prezzo affidabile.','bot_comment'))
            trend_mode=int(num(p,'InpEntryTrendMode',0))
            if trend_mode:
                checks.append(item('trend','Filtro di tendenza aggiuntivo','unknown' if trend_mode==1 else 'off',
                    'Filtro EMA aggiuntivo attivo.' if trend_mode==1 else 'La tendenza aggiuntiva riduce il rischio, senza bloccare da sola l’ingresso.',
                    'Indicatore interno non esposto; serve conferma dell’EA.' if trend_mode==1 else '', 'internal'))
            checks.append(item('side','Direzione abilitata','pass' if flag(p,'InpAllowLong' if direction==1 else 'InpAllowShort',True) else 'fail',label+' abilitato.' if flag(p,'InpAllowLong' if direction==1 else 'InpAllowShort',True) else label+' disabilitato.',
                '' if flag(p,'InpAllowLong' if direction==1 else 'InpAllowShort',True) else 'La configurazione non consente questa direzione.'))
            if valid and risk.get('min_lot') and api is not None:
                entry=q['ask'] if direction==1 else q['bid']; stop=low if direction==1 else high
                fraction=num(p,'InpStopDistanceFraction',1)
                if 0<fraction<1:stop=entry+(stop-entry)*fraction
                stop=math.floor(stop/tick_size+.5)*tick_size
                op_type=api.ORDER_TYPE_BUY if direction==1 else api.ORDER_TYPE_SELL
                loss=api.order_calc_profit(op_type,b['symbol'],risk['min_lot'],entry,stop)
                margin=api.order_calc_margin(op_type,b['symbol'],risk['min_lot'],entry) if hasattr(api,'order_calc_margin') else None
                stop_ok=risk.get('min_stop') is not None and (stop<q['bid']-risk['min_stop'] if direction==1 else stop>q['ask']+risk['min_stop'])
                minimum_risk=-loss if stop_ok and loss is not None and loss<0 else None
                checks.append(item('stop','Distanza minima dello stop','pass' if stop_ok else 'fail',f'Stop candidato {fmt(stop)}; distanza minima broker {fmt(risk["min_stop"])}.' if risk.get('min_stop') is not None else 'Distanza minima non disponibile.',
                    '' if stop_ok else 'Il prezzo e lo stop del range devono rispettare la distanza minima del broker.','reconstructed'))
                modulated=flag(p,'InpPeakGuard') or flag(p,'InpPeakSoftOnly') or flag(p,'InpStreakAdaptive') or trend_mode==2
                checks.append(item('lot','Rischio del lotto minimo','unknown' if minimum_risk is None or effective_budget is None or modulated else 'pass' if effective_budget>=minimum_risk>0 else 'fail',
                    f'Lotto minimo {fmt(risk["min_lot"])}: rischio {fmt(minimum_risk)} {money}; budget candidato {fmt(effective_budget)} {money}.' if minimum_risk is not None and effective_budget is not None else 'Calcolo in attesa di uno stop valido e di un prezzo di ingresso compatibile.',
                    'Il lotto minimo deve rientrare nel budget candidato. Stima sul prezzo attuale: sarà ricalcolata all’eventuale segnale.' if not modulated else 'Ulteriori riduzioni del rischio dipendono da stati interni non esposti.',
                    'reconstructed',max(0,minimum_risk-effective_budget) if minimum_risk is not None and effective_budget is not None else None,money))
                checks.append(item('margin','Margine per il lotto minimo','unknown' if margin is None else 'pass' if margin<=risk['margin_free']*.8 else 'fail',
                    f'Richiesti {fmt(margin)} {money}; utilizzabili {fmt(risk["margin_free"]*.8)} {money}.' if margin is not None else 'Margine non calcolabile.',
                    '' if margin is not None and margin<=risk['margin_free']*.8 else 'Il lotto minimo deve rientrare nell’80% del margine libero.','reconstructed',
                    max(0,margin-risk['margin_free']*.8) if margin is not None else None,money))
            else:
                checks.append(item('broker','Lotto, stop e margine','unknown','Verifica finale del broker non disponibile per questo range.','Servono range valido e dati su lotto minimo, stop e margine.','internal'))
            directions.append({'name':label,'checks':checks})
        sessions.append({'name':name,'phase':phase,'common':common,'directions':directions,
                         'atr_route':'ordinary' if not use_atr or initial else 'deferred' if not extra and flag(p,'InpATRDeferredBreakout',True) else 'blocked'})
    return {'version':1,'source':'Codice prova298, profilo salvato e dati MT5. Ricostruzione senza accesso ai flag interni dell’EA.',
            'observed_italy':snapshot['observed_italy'],'fresh_quotes':fresh,
            'candle_time':analysis.get('last_closed_m5',{}).get('time_italy') if analysis.get('last_closed_m5') else None,
            'seconds_to_next_close':300-raw_now%300,'atr_daily_reconstructed':atr,'risk_budget_estimate':budget,
            'sessions':sessions,'pyramid':{'own_positions':own,'required_positions':1,'trigger_r':num(p,'InpPyramidAtR',1.2),
                'hedging':risk.get('hedging',False),'state':'fail' if own!=1 else 'unknown',
                'missing':f'La piramidazione richiede esattamente una posizione propria: ora sono {own}.' if own!=1 else 'Servono break-even già eseguito, almeno +1,2R sul rischio iniziale e nessuna piramide già fatta. I relativi flag interni non sono esposti.',
                'note':'Inoltre: stop in profitto, conto hedging, budget disponibile e posizione non fade, DATR o EXTRA quando i relativi blocchi sono attivi.'}}
