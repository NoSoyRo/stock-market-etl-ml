'use strict';
const $ = id => document.getElementById(id);
const base = document.body.dataset.base.replace(/\/$/, '');
let coverage = {}, generation = 0, evidenceOffset = 0, evidenceDate = '', busy = false;
let conversation = [], agentAvailable = false;
const number = (v, digits=2) => v === null || v === undefined ? '—' : Number(v).toLocaleString('es-MX', {maximumFractionDigits:digits,minimumFractionDigits:digits});
const selection = () => ({entity_type:$('entityType').value,entity_id:$('entityId').value,fecha:$('snapshotDate').value});
const selectedKey = () => JSON.stringify(selection());
const utcDay = () => new Date().toISOString().slice(0,10);
const text = (id,value) => { $(id).textContent = value; };
const node = (tag,cls,value) => {const el=document.createElement(tag); if(cls)el.className=cls; if(value!==undefined)el.textContent=value; return el;};
const safeLink = url => {try {const u=new URL(url);return ['http:','https:'].includes(u.protocol)?u.href:null;}catch{return null;}};
async function api(path, params={}, options={}) {
  const query = new URLSearchParams(Object.entries(params).filter(([,v])=>v!==''&&v!==null&&v!==undefined));
  const res = await fetch(`${base}/api/${path}${query.size?'?'+query:''}`,options);
  let body; try{body=await res.json();}catch{throw new Error('El servidor no devolvió una respuesta válida.');}
  if(!res.ok) throw new Error(body.error || body.warnings?.join(' ') || 'No se pudo completar la consulta.');
  return body;
}
function notice(messages) {const value=messages.filter(Boolean).join(' ');text('notice',value);$('notice').hidden=!value;}
function empty(target, message) {$(target).replaceChildren(node('p','empty',message));}
function entities() {
  const type=$('entityType').value, previous=$('entityId').value;
  const values=coverage.entities?.[type==='ticker'?'tickers':'topics'] || [];
  $('entityId').replaceChildren(...values.map(v=>{const o=node('option','',v);o.value=v;return o;}));
  if(values.includes(previous)) $('entityId').value=previous;
  else if(type==='ticker'&&values.includes('WALMEX'))$('entityId').value='WALMEX';
}
async function loadCoverage(){
  const result=await api('coverage'); coverage=result.data||{};
  const count=(coverage.entities?.tickers||[]).length;
  text('coverageBadge',`${count} empresas · ${(coverage.entities?.topics||[]).length} temas`);
  entities();
  const today=utcDay(), end=coverage.date_range_utc?.last || today;
  const start=coverage.date_range_utc?.first || today;
  const recent=new Date(`${end}T00:00:00Z`);recent.setUTCDate(recent.getUTCDate()-29);
  $('fromDate').value=start>recent.toISOString().slice(0,10)?start:recent.toISOString().slice(0,10);
  $('toDate').value=end;
}
function snapshot(result){
  const data=result.data||{}, m=data.metrics||{};
  text('entityHeading',`${$('entityType').value==='ticker'?'Empresa':'Tema'} / ${$('entityId').value||'Sin selección'}`);
  text('cutoff',data.as_of?`Corte ${new Date(data.as_of).toLocaleString('es-MX',{timeZone:'UTC'})} UTC`:'Sin resultado disponible');
  text('sentiment',number(m.sentimiento_general,4));
  text('articleCount',number(m.cantidad_articulos,0));text('delta',number(m.cambio_sentimiento,4));
  text('interpretation',data.interpretation||'No hay una evaluación disponible para esta consulta.');
  text('windowLabel',data.window_days?`En los últimos ${data.window_days} días.`:'Sin ventana de cálculo disponible.');
  text('sourceCount',`${(data.sources||[]).length} fuentes en la ventana reciente`);
  text('method',data.scoring_method==='keywords'?'Método: palabras clave':data.scoring_method||'Método no disponible');
  const value=m.sentimiento_general;
  $('sentimentMarker').style.left=`${value===null||value===undefined?50:(Math.max(-1,Math.min(1,value))+1)*50}%`;
  $('sentimentMarker').hidden=value===null||value===undefined;
  $('sources').replaceChildren(...(data.sources||[]).map(s=>node('span','source-chip',s)));
  const status={not_found:'No hay resultados guardados para esta selección.',stale:'Los datos no están actualizados para la consulta del día.',no_recent_articles:'No hay artículos en la ventana reciente.',incompatible_metrics:'Las métricas tienen una ventana incompatible.'};
  notice([status[result.status],...(result.warnings||[])]);
  return data.date_utc||'';
}
async function loadSelection(){
  const version=++generation;
  $('refresh').disabled=true; $('filters').querySelector('button').disabled=true;
  evidenceOffset=0;evidenceDate='';$('moreArticles').hidden=true;
  empty('articles','Consultando la evidencia…');text('evidenceCount','—');
  try {
    if(!$('entityId').value){empty('articles','No hay entidades disponibles.');notice(['Todavía no hay cobertura de noticias.']);return;}
    const result=await api('snapshot',selection());
    if(version!==generation)return;
    evidenceDate=snapshot(result);
    if(evidenceDate)await loadEvidence(version,false);
    else empty('articles','Elige una fecha con datos guardados para consultar su evidencia.');
  }catch(error){if(version===generation){snapshot({});notice([error.message]);empty('articles','No se pudo consultar la evidencia.');}}
  finally{if(version===generation){$('refresh').disabled=false;$('filters').querySelector('button').disabled=false;}}
  if(version===generation)await loadHistory(version);
}
async function loadHistory(version=generation){
  const key=selectedKey();
  try {
    const result=await api('history',{...selection(),inicio:$('fromDate').value,fin:$('toDate').value});
    if(version!==generation||key!==selectedKey())return;
    drawChart(result.data?.results||[]);
  }catch(error){if(version===generation)empty('chart',error.message);}
}
function drawChart(rows){
  const valid=rows.filter(r=>r.metrics?.sentimiento_general!==null&&r.metrics?.sentimiento_general!==undefined);
  if(!valid.length){empty('chart','No hay observaciones de sentimiento en este periodo. Prueba con otras fechas.');return;}
  const svgNS='http://www.w3.org/2000/svg';
  const svg=document.createElementNS(svgNS,'svg');svg.setAttribute('viewBox','0 0 680 230');
  const elem=(tag,attrs,value)=>{const e=document.createElementNS(svgNS,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,v);if(value!==undefined)e.textContent=value;svg.append(e);return e;};
  const times=valid.map(r=>Date.parse(r.date_utc)), min=Math.min(...times), max=Math.max(...times);
  const x=t=>max===min?355:48+(t-min)/(max-min)*610, y=v=>108-v*78;
  [-1,-.5,0,.5,1].forEach(v=>{elem('line',{x1:48,x2:658,y1:y(v),y2:y(v),stroke:'#e7ede5','stroke-dasharray':v===0?'3 4':'0'});elem('text',{x:30,y:y(v)+4,fill:'#8a988c','font-size':10,'text-anchor':'end'},v.toFixed(1));});
  let previous=null;
  valid.forEach(r=>{
    const t=Date.parse(r.date_utc), v=r.metrics.sentimiento_general;
    if(previous&&t-previous.t<=86400000&&r.window_days===previous.r.window_days&&r.scoring_method===previous.r.scoring_method)
      elem('line',{x1:x(previous.t),y1:y(previous.r.metrics.sentimiento_general),x2:x(t),y2:y(v),stroke:'#137764','stroke-width':2});
    const c=elem('circle',{cx:x(t),cy:y(v),r:4,fill:'#137764',stroke:'#fff','stroke-width':2});
    const title=document.createElementNS(svgNS,'title');title.textContent=`${r.date_utc}: ${number(v,4)} · ${r.metrics.cantidad_articulos} artículos · ventana ${r.window_days} días`;c.append(title);previous={t,r};
  });
  const ticks=[...new Set([min,(min+max)/2,max])];
  ticks.forEach(t=>elem('text',{x:x(t),y:216,fill:'#7f8e80','font-size':10,'text-anchor':'middle'},new Date(t).toLocaleDateString('es-MX',{day:'2-digit',month:'short',timeZone:'UTC'})));
  $('chart').replaceChildren(svg);
}
async function loadEvidence(version=generation,append=false){
  const key=selectedKey();$('moreArticles').disabled=true;
  try{
    const result=await api('evidence',{...selection(),fecha:evidenceDate,offset:append?evidenceOffset:0});
    if(version!==generation||key!==selectedKey())return;
    const data=result.data||{};if(!append)$('articles').replaceChildren();
    for(const a of data.articles||[]){
      const article=node('article','article');
      const meta=node('div','article-meta');meta.append(node('span','',a.source||'Fuente no indicada'),node('span','',a.published_at?new Date(a.published_at).toLocaleDateString('es-MX',{timeZone:'UTC'}):'Publicación desconocida'));
      article.append(meta);
      const url=safeLink(a.url);const title=node(url?'a':'div','',a.title||'Sin titular');
      if(url){title.href=url;title.target='_blank';title.rel='noopener noreferrer';}article.append(title);
      if(a.summary)article.append(node('p','',a.summary));$('articles').append(article);
    }
    if(!(data.articles||[]).length&&!append)empty('articles','No hay noticias recientes para este corte.');
    if(result.status!=='ok'||result.warnings?.length)$('articles').prepend(node('p','notice',(result.warnings||['La evidencia no está completa.']).join(' ')));
    evidenceOffset=(data.offset||0)+(data.articles||[]).length;
    text('evidenceCount',`${data.total??0} artículos`);$('moreArticles').hidden=!data.has_more;
  }catch(error){if(version===generation&&!append)empty('articles',error.message);}
  finally{$('moreArticles').disabled=false;}
}
function addMessage(role,value){const m=node('div',`message ${role}`,value);$('messages').append(m);$('messages').scrollTop=$('messages').scrollHeight;return m;}
async function sendQuestion(event){
  event.preventDefault();if(busy||!agentAvailable)return;
  const question=$('question').value.trim();if(!question)return;
  busy=true;$('send').disabled=true;$('clearChat').disabled=true;$('question').value='';
  addMessage('user',question);const pending=addMessage('assistant','Consultando datos y fuentes…');
  try{
    const result=await api('chat',{}, {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question,context:selection(),history:conversation.slice(-6)})});
    if(!result.ok)throw new Error(result.error||'No se pudo completar el análisis.');
    pending.textContent=result.answer;
    const citations=node('div','citations');
    for(const c of result.citations||[]){const url=safeLink(c.url);if(!url)continue;const a=node('a','',`↗ ${c.title||c.source||'Consultar fuente'}`);a.href=url;a.target='_blank';a.rel='noopener noreferrer';citations.append(a);}
    pending.append(citations);conversation.push({role:'user',content:question},{role:'assistant',content:String(result.answer).slice(0,5000)});conversation=conversation.slice(-6);
  }catch(error){pending.textContent=error.message;pending.classList.add('error');}
  finally{busy=false;$('send').disabled=!agentAvailable;$('clearChat').disabled=false;$('messages').scrollTop=$('messages').scrollHeight;}
}
$('filters').addEventListener('submit',e=>{e.preventDefault();loadSelection();});
$('entityType').addEventListener('change',()=>{entities();loadSelection();});
$('entityId').addEventListener('change',()=>loadSelection());
$('refresh').addEventListener('click',()=>loadSelection());
$('rangeForm').addEventListener('submit',e=>{e.preventDefault();loadHistory();});
$('moreArticles').addEventListener('click',()=>loadEvidence(generation,true));
$('chatForm').addEventListener('submit',sendQuestion);
$('clearChat').addEventListener('click',()=>{conversation=[];$('messages').replaceChildren();addMessage('assistant','Nueva conversación. ¿Qué noticias quieres analizar?');});
document.querySelectorAll('[data-question]').forEach(button=>button.addEventListener('click',()=>{$('question').value=button.dataset.question;$('chatForm').requestSubmit();}));
document.querySelectorAll('nav a').forEach(a=>a.addEventListener('click',()=>{document.querySelectorAll('nav a').forEach(el=>el.classList.remove('active'));a.classList.add('active');}));
(async()=>{
 try{await loadCoverage();await loadSelection();}catch(error){notice([error.message]);text('coverageBadge','Cobertura no disponible');}
})();
(async()=>{try{const result=await api('agent');agentAvailable=!!result.available;text('agentBadge',agentAvailable?`Disponible · ${result.provider==='gemini'?'Gemini':'Anthropic'}`:'IA pendiente de configuración');text('chatHint',agentAvailable?'Verifica las conclusiones con las fuentes.':'El panel funciona sin activar el chat de IA.');}catch{agentAvailable=false;text('agentBadge','Analista no disponible');}finally{$('send').disabled=!agentAvailable;document.querySelectorAll('[data-question]').forEach(b=>b.disabled=!agentAvailable);}})();
