/* Cache della sola interfaccia. Dati MT5 e risposte API non vengono conservati. */
'use strict';
const CACHE='quantix-iq-shell-v9';
const SHELL=['/01_app/','/01_app/index.html','/01_app/styles.css','/01_app/personal.css','/01_app/personal.js','/01_app/model.js','/01_app/app.js','/01_app/chart-live.js','/01_app/entry-criteria.js','/01_app/live.js','/01_app/pwa.js','/01_app/vendor/lucide.min.js','/01_app/manifest.webmanifest','/01_app/icons/icon-192.png','/01_app/icons/icon-512.png','/01_app/icons/icon-maskable-512.png','/01_app/icons/apple-touch-icon.png','/02_immagini/logo/logo-app.png'];
self.addEventListener('install',event=>event.waitUntil(caches.open(CACHE).then(cache=>cache.addAll(SHELL)).then(()=>self.skipWaiting())));
self.addEventListener('activate',event=>event.waitUntil((async()=>{
  for(const name of await caches.keys())if(name.startsWith('quantix-iq-shell-')&&name!==CACHE)await caches.delete(name);
  await self.clients.claim();
})()));
self.addEventListener('fetch',event=>{
  const url=new URL(event.request.url);
  if(event.request.method!=='GET'||url.origin!==self.location.origin||url.pathname.startsWith('/api/'))return;
  if(!SHELL.includes(url.pathname))return;
  event.respondWith((async()=>{
    try{
      const response=await fetch(event.request);
      if(response.ok){const cache=await caches.open(CACHE);await cache.put(event.request,response.clone());}
      // Un diniego di accesso non deve trasformarsi in una pagina autenticata dalla cache.
      return response;
    }catch(error){
      const cache=await caches.open(CACHE);
      const cached=await cache.match(event.request);
      if(cached)return cached;
      if(event.request.mode==='navigate')return await cache.match('/01_app/');
      throw error;
    }
  })());
});

/* Web Push: il browser può avviare questo worker anche con l'app chiusa. */
self.addEventListener('push',event=>{
  event.waitUntil((async()=>{
    let data={};try{data=event.data?.json()||{};}catch(_){}
    const kind=String(data.kind||'');
    const view=kind==='risk'?'analysis':data.position_id?'activity':'home';
    const trade=/^[0-9]{1,24}$/.test(String(data.position_id||''))?String(data.position_id):null;
    await self.registration.showNotification('QUANTIX IQ',{body:String(data.body||'Nuovo aggiornamento del conto.').slice(0,500),icon:'/01_app/icons/icon-192.png',badge:'/01_app/icons/icon-192.png',tag:String(data.tag||'quantix-update').slice(0,200),data:{view,trade}});
  })());
});
self.addEventListener('notificationclick',event=>{
  event.notification.close();
  event.waitUntil((async()=>{
    const data=event.notification.data||{},view=['home','analysis','activity','stats','settings'].includes(data.view)?data.view:'home';
    const pages=await self.clients.matchAll({type:'window',includeUncontrolled:true});
    const existing=pages.find(c=>new URL(c.url).origin===self.location.origin&&new URL(c.url).pathname.startsWith('/01_app/'));
    if(existing){await existing.focus();existing.postMessage({type:'quantix:notification',view,trade:data.trade});return;}
    const url=new URL('/01_app/',self.location.origin);url.searchParams.set('view',view);if(data.trade)url.searchParams.set('trade',data.trade);await self.clients.openWindow(url.href);
  })());
});
