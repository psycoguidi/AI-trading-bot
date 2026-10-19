/* Installazione Android e cache della grafica. Nessuna operazione di trading. */
(function(){
  'use strict';
  let installPrompt=null;
  const standalone=()=>matchMedia('(display-mode: standalone)').matches||navigator.standalone===true;
  window.QuantixPWA={
    markup:()=>`<button type="button" class="qi-button ghost" data-pwa-install aria-label="Installa QUANTIX IQ">${standalone()?'App installata':'Installa app'}</button>`
  };
  addEventListener('beforeinstallprompt',event=>{event.preventDefault();installPrompt=event;});
  addEventListener('appinstalled',()=>{installPrompt=null;document.querySelectorAll('[data-pwa-install]').forEach(el=>el.textContent='App installata');});
  document.addEventListener('click',async event=>{
    const button=event.target.closest('[data-pwa-install]');if(!button)return;
    if(standalone()){alert('QUANTIX IQ è già aperta come applicazione.');return;}
    if(installPrompt){await installPrompt.prompt();await installPrompt.userChoice;installPrompt=null;return;}
    const root=document.getElementById('quantix-app'),dialog=root?.querySelector('dialog');
    if(!dialog)return;
    dialog.innerHTML=`<button class="qi-icon-button qi-modal-close" data-action="close-modal" aria-label="Chiudi">×</button><h2>QUANTIX IQ sul tuo Android</h2><div class="qi-modal-body"><p>1. Connetti Tailscale sul telefono con lo stesso account del PC.\n2. Apri il tuo indirizzo privato HTTPS in Chrome.\n3. Apri il menu ⋮ e scegli «Installa app» oppure «Aggiungi alla schermata Home».\n\nIl PC e il servizio QUANTIX IQ devono rimanere accesi per vedere dati aggiornati. La demo rimane consultabile senza collegamento; i dati MT5 non vengono simulati quando manca la connessione.</p></div>`;
    dialog.showModal();
  });
  if('serviceWorker' in navigator&&window.isSecureContext&&location.protocol!=='file:'){
    addEventListener('load',()=>navigator.serviceWorker.register('/01_app/sw.js',{scope:'/01_app/',updateViaCache:'none'}).catch(()=>{}));
  }
})();
