"""
Dashboard de Fundamentales PIT (Equipo Rojo).

Lee SOLO la capa golden (data/fundamentals/golden/fundamentales_pit.parquet) y
genera data/dashboard_fundamentales.html. Ábrelo con doble clic.

Uso, desde la raíz del repo:
    python build_dashboard.py
    python build_dashboard.py [archivo.parquet|archivo.csv] [salida.html]

El diseño está inspirado en las plantillas libres Tabler (MIT) y TailAdmin (MIT).
Las gráficas usan ApexCharts, que se incrusta en el HTML: el dashboard abre sin
internet. Si no encuentra la librería, la descarga una vez y la guarda en
vendor/; si tampoco hay red, la enlaza desde un CDN (ahí sí hace falta internet).
"""
import json
import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
GOLDEN = RAIZ / "data" / "fundamentals" / "golden" / "fundamentales_pit.parquet"
APEX_VERSION = "8.7.0"
APEX_URL = f"https://cdn.jsdelivr.net/npm/apexcharts@{APEX_VERSION}/dist/apexcharts.min.js"
APEX_LOCAL = [RAIZ / "vendor" / "apexcharts.min.js",
              RAIZ / "data" / "fundamentals" / "vendor" / "apexcharts.min.js"]

RATIOS = {"gm": "fund_gross_margin", "om": "fund_oper_margin", "nm": "fund_net_margin",
          "ry": "fund_revenue_yoy", "ey": "fund_earnings_yoy"}


def apexcharts() -> str:
    """Devuelve el <script> de ApexCharts: incrustado si se puede, si no desde el CDN."""
    for ruta in APEX_LOCAL:
        if ruta.exists():
            return f"<script>{ruta.read_text(encoding='utf-8')}</script>"
    destino = APEX_LOCAL[-1]
    try:
        import urllib.request
        with urllib.request.urlopen(APEX_URL, timeout=30) as r:
            js = r.read().decode("utf-8")
        destino.parent.mkdir(parents=True, exist_ok=True)
        destino.write_text(js, encoding="utf-8")
        print(f"[apexcharts] descargado y guardado en {destino}")
        return f"<script>{js}</script>"
    except Exception as exc:  # sin red: el HTML necesitará internet para abrir
        print(f"[apexcharts] no se pudo descargar ({exc}); se enlaza desde el CDN")
        return f'<script src="{APEX_URL}"></script>'


def cargar(src: Path) -> pd.DataFrame:
    df = pd.read_csv(src) if src.suffix == ".csv" else pd.read_parquet(src)
    faltan = {"ticker", "quarter_end", "available_date", *RATIOS.values()} - set(df.columns)
    if faltan:
        sys.exit(f"A {src} le faltan columnas de la capa golden: {sorted(faltan)}")
    df["quarter_end"] = pd.to_datetime(df["quarter_end"])
    df["available_date"] = pd.to_datetime(df["available_date"])
    return df.sort_values(["ticker", "quarter_end"]).reset_index(drop=True)


def datos(df: pd.DataFrame) -> dict:
    """Lo que necesita la página. Todo sale de golden; nada se calcula con otra fuente."""
    r = lambda x: None if pd.isna(x) else round(float(x), 4)
    f = lambda s: s.dt.strftime("%Y-%m-%d").tolist()
    series = {t: dict(q=f(g["quarter_end"]), a=f(g["available_date"]),
                      **{k: [r(v) for v in g[c]] for k, c in RATIOS.items()})
              for t, g in df.groupby("ticker")}

    lag = (df["available_date"] - df["quarter_end"]).dt.days
    moda = int(lag.mode()[0])
    pit = dict(lag_min=int(lag.min()), lag_max=int(lag.max()), lag_mode=moda,
               pct_lag_ok=round(float((lag == moda).mean() * 100), 1))

    kpi = dict(rows=int(len(df)), tickers=int(df["ticker"].nunique()),
               dups=int(df.duplicated(["ticker", "quarter_end"]).sum()),
               generated=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
    try:  # emisoras del catálogo que la API no devolvió (solo desde la raíz del repo)
        sys.path.insert(0, str(RAIZ))
        from src.data import bmv_catalog as bc
        cat = next(v for k, v in vars(bc).items()
                   if k.endswith("FUNDAMENTALS") and isinstance(v, dict))
        kpi["missing"] = sorted(set(cat) - set(df["ticker"].unique()))
        kpi["catalog"] = len(cat)
    except Exception:
        kpi["missing"], kpi["catalog"] = [], None
    return dict(series=series, pit=pit, kpi=kpi)


def main(argv: list[str]) -> None:
    if argv:
        src = Path(argv[0])
    elif GOLDEN.exists():
        src = GOLDEN
    else:
        sys.exit("No existe la capa golden. Corre primero: "
                 "python -m productos.fundamentales_pit.capas todo")
    salida = Path(argv[1]) if len(argv) > 1 else RAIZ / "data" / "dashboard_fundamentales.html"

    df = cargar(src)
    D = datos(df)
    html = (PLANTILLA
            .replace("__APEX__", apexcharts())
            .replace("__DATA__", json.dumps(D, ensure_ascii=False)))
    salida.parent.mkdir(parents=True, exist_ok=True)
    salida.write_text(html, encoding="utf-8")
    print(f"Dashboard generado: {salida.resolve()}  "
          f"({D['kpi']['rows']} filas, {D['kpi']['tickers']} emisoras)")


PLANTILLA = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Fundamentales PIT · Fundamentales PIT</title>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap">
<style>
*{box-sizing:border-box}
html,body{margin:0}
body{font-family:var(--font);color:var(--ink);background:var(--bg);font-size:14px;line-height:1.5;font-variant-numeric:tabular-nums}
a{color:inherit;text-decoration:none}
.app{display:grid;grid-template-columns:var(--side-w) minmax(0,1fr);min-height:100vh;background:linear-gradient(to right,var(--side-bg) 0 var(--side-w),transparent var(--side-w))}
.side{background:var(--side-bg);color:var(--side-ink);padding:24px 16px;position:sticky;top:0;height:100vh;border-right:var(--side-border)}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;font-size:16px;margin:0 8px 28px}
.brand i{width:30px;height:30px;border-radius:8px;background:var(--brand);display:inline-block;position:relative}
.brand i:after{content:"";position:absolute;inset:8px;border:2px solid #fff;border-radius:50%;border-right-color:transparent}
.brand small{display:block;font-weight:500;font-size:12px;color:var(--side-mute)}
.nav{display:flex;flex-direction:column;gap:4px}
.nav a{padding:9px 12px;border-radius:var(--r-sm);color:var(--side-mute);font-weight:500}
.nav a.on{background:var(--side-on-bg);color:var(--side-on)}
.nav .lbl{font-size:11px;letter-spacing:.08em;text-transform:uppercase;color:var(--side-mute);opacity:.7;margin:18px 12px 6px}
.topbar{display:none}
main{padding:28px 32px 48px;min-width:0}
.head{display:flex;justify-content:space-between;align-items:flex-end;gap:16px;flex-wrap:wrap;margin-bottom:20px}
.head h1{margin:0;font-size:26px;font-weight:700;letter-spacing:-.01em}
.head p{margin:4px 0 0;color:var(--mute)}
.pill{display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;padding:4px 10px;border-radius:999px;background:var(--brand-soft);color:var(--brand-ink)}
.card{background:var(--card);border:var(--card-border);border-radius:var(--r);box-shadow:var(--shadow);padding:20px}
.card h3{margin:0;font-size:15px;font-weight:600}
.card .sub{margin:2px 0 12px;color:var(--mute);font-size:13px}
.hero{display:grid;grid-template-columns:1.1fr 1.4fr;gap:28px;align-items:center;margin-bottom:20px;background:var(--hero-bg);color:var(--hero-ink);border:var(--hero-border)}
.hero .q{font-size:13px;font-weight:600;opacity:.8;text-transform:uppercase;letter-spacing:.06em}
.hero .fecha{font-size:34px;font-weight:700;letter-spacing:-.02em;margin:4px 0}
.hero .exp{opacity:.85;margin:0;max-width:46ch}
.slider{width:100%;accent-color:var(--hero-accent);height:6px}
.ticks{display:flex;justify-content:space-between;font-size:12px;opacity:.75;margin-top:6px}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.chips button{font:inherit;font-size:12px;font-weight:600;border:1px solid var(--hero-chip-b);background:transparent;color:inherit;border-radius:999px;padding:5px 11px;cursor:pointer}
.chips button:hover{background:var(--hero-chip-h)}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px;margin-bottom:20px}
.kpi .l{color:var(--mute);font-size:13px;font-weight:500}
.kpi .v{white-space:nowrap;font-size:28px;font-weight:700;letter-spacing:-.02em;margin-top:4px}
.kpi .d{font-size:12px;margin-top:2px;color:var(--mute)}
.kpi .spark{height:48px;margin:6px -6px -10px}
.ico{width:40px;height:40px;border-radius:var(--r-sm);background:var(--brand-soft);display:var(--ico-display);align-items:center;justify-content:center;margin-bottom:12px;color:var(--brand)}
.ico svg{width:20px;height:20px}
.grid2{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:16px;margin-bottom:20px}
.row{display:flex;justify-content:space-between;align-items:flex-start;gap:12px;flex-wrap:wrap}
.seg{display:inline-flex;background:var(--seg-bg);border-radius:var(--r-sm);padding:3px}
.seg button{font:inherit;font-size:12.5px;font-weight:600;border:0;background:transparent;color:var(--mute);padding:6px 12px;border-radius:calc(var(--r-sm) - 2px);cursor:pointer}
.seg button.on{background:var(--card);color:var(--ink);box-shadow:0 1px 2px rgba(16,24,40,.12)}
select,input[type=search]{font:inherit;font-size:13.5px;color:var(--ink);background:var(--card);border:1px solid var(--line);border-radius:var(--r-sm);padding:8px 12px}
.facts{display:flex;gap:24px;flex-wrap:wrap;margin:12px 0 4px}
.facts div{font-size:12px;color:var(--mute)}.facts b{display:block;font-size:18px;color:var(--ink)}
.tbl{overflow:auto;max-height:460px;margin-top:12px;border-radius:var(--r-sm);border:1px solid var(--line)}
table{width:100%;border-collapse:collapse;font-size:13.5px}
th,td{padding:10px 14px;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2),th:nth-child(3),td:nth-child(3){text-align:left}
th{position:sticky;top:0;background:var(--th-bg);color:var(--mute);font-weight:600;font-size:12px;cursor:pointer;user-select:none}
td:first-child{font-weight:600}
tbody tr:hover{background:var(--row-h)}
.neg{color:var(--red)}.pos{color:var(--green)}.na{color:var(--line)}
.badge{font-size:11px;font-weight:600;padding:2px 7px;border-radius:999px}
.badge.ok{background:var(--ok-bg);color:var(--green)}.badge.no{background:var(--no-bg);color:var(--red)}
.qual{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:20px}
.qual .card{display:flex;gap:12px;align-items:flex-start}
.qual b{display:block;font-size:14px}.qual span{color:var(--mute);font-size:13px}
.dot{width:10px;height:10px;border-radius:50%;margin-top:6px;flex:none}
.grid2b{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1.5fr);gap:16px;margin-bottom:20px}
.tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.tools input{width:200px}
.btn{font:inherit;font-size:13px;font-weight:600;border:1px solid var(--line);background:var(--card);color:var(--ink);border-radius:var(--r-sm);padding:8px 12px;cursor:pointer}
.btn:hover{border-color:var(--brand)}
body{transition:background .2s,color .2s}
footer{color:var(--mute);font-size:12px;margin-top:24px}
/* layout: barra superior (Tabler) */
body.lay-top .app{grid-template-columns:minmax(0,1fr)}
body.lay-top .side{display:none}
body.lay-top .topbar{display:flex;align-items:center;gap:28px;background:var(--top-bg);color:var(--top-ink);padding:0 32px;height:58px}
body.lay-top .topbar .brand{margin:0}
body.lay-top .topbar a{color:var(--top-mute);font-weight:500;font-size:14px}
body.lay-top .topbar a.on{color:var(--top-ink)}
body.lay-top main{max-width:1320px;margin:0 auto;width:100%}
@media(max-width:1000px){.app{grid-template-columns:1fr;--side-w:0px;background:none}.side{display:none}.hero,.grid2,.grid2b{grid-template-columns:1fr}.kpis{grid-template-columns:repeat(2,1fr)}.qual{grid-template-columns:1fr}main{padding:20px 16px}}

:root{--font:Inter,-apple-system,"Segoe UI",Roboto,sans-serif;--bg:#f5f7fa;--card:#fff;--ink:#1d273b;--mute:#667382;--line:#e6eaf0;
--brand:#206bc4;--brand-soft:#e9f0f9;--brand-ink:#1a569d;--r:12px;--r-sm:8px;--card-border:1px solid #e6eaf0;--shadow:0 1px 2px rgba(29,39,59,.05);
--side-w:252px;--side-bg:#fff;--side-ink:#1d273b;--side-mute:#667382;--side-border:1px solid #e6eaf0;--side-on-bg:#e9f0f9;--side-on:#1a569d;
--hero-bg:#1d273b;--hero-ink:#fff;--hero-border:0;--hero-accent:#4299e1;--hero-chip-b:rgba(255,255,255,.3);--hero-chip-h:rgba(66,153,225,.25);
--seg-bg:#f1f3f6;--th-bg:#f8fafc;--row-h:#f8fafc;--green:#2fb344;--red:#d63939;--ok-bg:#e6f6e9;--no-bg:#fbebeb;--ico-display:flex}
.hero{border-left:4px solid #206bc4}
.hero .fecha{color:#fff;letter-spacing:-.02em}
.hero .q{color:#9ec5f8}
.kpi .v{font-size:30px;color:#1d273b}
.ico{border:1px solid #d7e3f2}
.side .brand{padding-bottom:16px;border-bottom:1px solid #e6eaf0;margin-bottom:16px}
body.dark{--bg:#121b27;--card:#18212f;--ink:#dde3ea;--mute:#8a98ab;--line:#26354a;--card-border:1px solid #26354a;--brand-soft:#1c3149;--brand-ink:#9ec5f8;
  --side-bg:#121b27;--side-ink:#dde3ea;--side-mute:#8a98ab;--side-border:1px solid #26354a;--side-on-bg:#1c3149;--side-on:#9ec5f8;
  --hero-bg:#0e1724;--seg-bg:#121b27;--th-bg:#1b2634;--row-h:#1c2838;--ok-bg:#16382a;--no-bg:#3f1d22}
  body.dark .kpi .v{color:#dde3ea}body.dark .ico{border-color:#2b415c}body.dark .side .brand{border-bottom-color:#26354a}
</style></head>
<body class="lay-side">
<div class="topbar"><div class="brand"><i></i><div>Fundamentales PIT<small>Equipo Rojo</small></div></div><a class="on" href="#resumen">Resumen</a><a href="#emisora">Emisora</a><a href="#tabla">Detalle</a><a href="#calidad">Calidad</a></div>
<div class="app">
<aside class="side"><div class="brand"><i></i><div>Fundamentales PIT<small>Equipo Rojo</small></div></div>
<nav class="nav"><span class="lbl">Producto</span><a class="on" href="#resumen">Resumen</a><a href="#emisora">Explorar emisora</a><a href="#tabla">Detalle por emisora</a><a href="#calidad">Calidad de datos</a></nav></aside>
<main>
<div class="head" id="resumen"><div><h1>Fundamentales point-in-time</h1><p>Estados financieros trimestrales de la BMV, tal como se conocían en cada fecha.</p></div><div class="tools"><input type="search" id="buscar" list="tks" placeholder="Buscar emisora…" aria-label="Buscar emisora"><datalist id="tks"></datalist><button class="btn" id="modo" type="button">Modo oscuro</button><span class="pill" id="gen"></span></div></div>

<section class="card hero">
 <div><div class="q">¿Qué se sabía el día…?</div><div class="fecha" id="fecha"></div>
 <p class="exp" id="exp"></p></div>
 <div><input type="range" class="slider" id="sl" aria-label="Fecha de consulta"><div class="ticks" id="ticks"></div>
 <div class="chips" id="chips"></div></div>
</section>

<div class="kpis" id="kpis"></div>

<div class="grid2">
 <div class="card"><div class="row"><div><h3>Márgenes del mercado</h3><p class="sub">Mediana entre emisoras, solo trimestres ya publicados a la fecha</p></div>
  <div class="seg" id="segm"><button data-m="m" class="on">Márgenes</button><button data-m="y">Crecimiento YoY</button></div></div><div id="ch-med"></div></div>
 <div class="card"><h3>Margen neto por emisora</h3><p class="sub" id="rank-sub"></p><div id="ch-rank"></div></div>
</div>

<div class="card" id="emisora" style="margin-bottom:20px"><div class="row"><div><h3>Explorar una emisora</h3><p class="sub">La línea punteada es la fecha de consulta: lo que está a la derecha todavía no se podía usar</p></div>
 <div style="display:flex;gap:10px;flex-wrap:wrap"><select id="sel" aria-label="Emisora"></select><div class="seg" id="sege"><button data-m="m" class="on">Márgenes</button><button data-m="y">YoY</button></div></div></div>
 <div class="facts" id="facts"></div><div id="ch-one"></div></div>

<div class="grid2b">
 <div class="card"><h3>¿Cuántas emisoras ganan dinero?</h3><p class="sub">Margen neto del último trimestre conocido de cada emisora</p><div id="ch-dona"></div></div>
 <div class="card"><h3>Trimestres con ganancia y con pérdida</h3><p class="sub">Por año de cierre, solo trimestres ya publicados a la fecha</p><div id="ch-stack"></div></div>
</div>
<div class="qual" id="calidad"></div>

<div class="card" id="tabla"><div class="row"><div><h3>Detalle por emisora</h3><p class="sub">Último trimestre conocido a la fecha de consulta · clic en un encabezado para ordenar</p></div><input type="search" id="q" placeholder="Buscar emisora"></div>
 <div class="tbl"><table><thead id="th"></thead><tbody id="tb"></tbody></table></div></div>
<footer>Fuente: API DataBursátil · capa golden de Fundamentales PIT</footer>
</main></div>
__APEX__
<script>
const D=__DATA__, P={"series": ["#206bc4", "#4299e1", "#2fb344", "#f76707"], "red": "#d63939", "green": "#2fb344", "amber": "#f59f00", "mute": "#667382", "grid": "#eef1f5", "dash": 0, "ink": "#1d273b", "tip": "light", "areaFrom": 0.22, "barR": 4, "dark": {"mute": "#8a98ab", "grid": "#26354a", "ink": "#dde3ea", "tip": "dark"}}, FONT=getComputedStyle(document.body).fontFamily;
const $=i=>document.getElementById(i), DAY=864e5, LAG=D.pit.lag_mode;
const pc=(v,d=1)=>v==null||!isFinite(v)?'—':(v*100).toFixed(d).replace('-','−')+'%';
const fmt=t=>new Date(t).toLocaleDateString('es-MX',{day:'numeric',month:'short',year:'numeric',timeZone:'UTC'});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const NAME={gm:'Margen bruto',om:'Margen operativo',nm:'Margen neto',ry:'Ingresos YoY',ey:'Utilidad YoY'};
const ICON={a:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 21h18M5 21V10m5 11V4m5 17v-8m5 8V7"/></svg>',
 b:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="4" width="18" height="17" rx="2"/><path d="M3 9h18M8 2v4m8-4v4"/></svg>',
 c:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 17l6-6 4 4 8-8M15 7h6v6"/></svg>',
 d:'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>'};
// filas planas: ticker, trimestre, disponible desde
const ROWS=[];for(const [t,s] of Object.entries(D.series))s.q.forEach((q,i)=>{const qt=Date.parse(q);ROWS.push({t,q,qt,av:Date.parse(s.a[i]),gm:s.gm[i],om:s.om[i],nm:s.nm[i],ry:s.ry[i],ey:s.ey[i]})});
const QS=[...new Set(ROWS.map(r=>r.qt))].sort((a,b)=>a-b);
const HOY=Date.parse(D.kpi.generated.slice(0,10)), MIN=Math.min(...ROWS.map(r=>r.av)), MAX=Math.max(HOY,...ROWS.map(r=>r.av));
let F=HOY, modeM='m', modeE='m', sk='nm', dir=-1;
const med=a=>{a=a.filter(v=>v!=null&&isFinite(v)).sort((x,y)=>x-y);if(!a.length)return null;const m=a.length>>1;return a.length%2?a[m]:(a[m-1]+a[m])/2};
function vis(f){return ROWS.filter(r=>r.av<=f)}
function ultimo(f){const m={};for(const r of vis(f))if(!m[r.t]||r.qt>m[r.t].qt)m[r.t]=r;return Object.values(m)}
function medianas(f,k){const v=vis(f),out=[];for(const q of QS){const g=v.filter(r=>r.qt===q);if(g.length>=5)out.push({x:q,y:+(med(g.map(r=>r[k]))*100).toFixed(2)})}return out}
const base=(o)=>Object.assign({chart:{fontFamily:FONT,toolbar:{show:false},zoom:{enabled:false},animations:{speed:350},foreColor:P.mute},dataLabels:{enabled:false},grid:{borderColor:P.grid,strokeDashArray:P.dash,padding:{left:6,right:6}},legend:{position:'top',horizontalAlign:'left',fontWeight:500,markers:{size:5}},tooltip:{theme:P.tip}},o);
const linea=f=>({xaxis:[{x:f,borderColor:P.ink,strokeDashArray:4,label:{text:'Consulta',orientation:'horizontal',textAnchor:'end',offsetX:-4,style:{background:P.ink,color:getComputedStyle(document.body).getPropertyValue('--card').trim()||'#fff',fontSize:'11px'}}}]});
// --------- gráficas
const chMed=new ApexCharts($('ch-med'),base({chart:{type:'area',height:360,fontFamily:FONT,toolbar:{show:false},zoom:{enabled:false},foreColor:P.mute},series:[],colors:P.series,stroke:{curve:'smooth',width:2.5},fill:{type:'gradient',gradient:{opacityFrom:P.areaFrom,opacityTo:0.02}},xaxis:{type:'datetime',min:QS[0],max:MAX,labels:{datetimeFormatter:{year:'yyyy',month:"MMM 'yy"}}},yaxis:{labels:{formatter:v=>Math.round(v)+'%'}},tooltip:{theme:P.tip,x:{format:'dd MMM yyyy'},y:{formatter:v=>v==null?'—':v.toFixed(1)+'%'}}}));chMed.render();
const chRank=new ApexCharts($('ch-rank'),base({chart:{type:'bar',height:360,fontFamily:FONT,toolbar:{show:false},foreColor:P.mute},series:[{name:'Margen neto',data:[]}],plotOptions:{bar:{horizontal:true,borderRadius:P.barR,barHeight:'70%',distributed:true}},legend:{show:false},xaxis:{labels:{formatter:v=>Math.round(v)+'%'}},tooltip:{theme:P.tip,y:{formatter:v=>v.toFixed(1)+'%'}}}));chRank.render();
const chOne=new ApexCharts($('ch-one'),base({chart:{type:'line',height:320,fontFamily:FONT,toolbar:{show:false},zoom:{enabled:false},foreColor:P.mute},series:[],colors:P.series,stroke:{curve:'smooth',width:2.5},markers:{size:0,hover:{size:5}},xaxis:{type:'datetime'},yaxis:{labels:{formatter:v=>Math.round(v)+'%'}},tooltip:{theme:P.tip,x:{format:'dd MMM yyyy'},y:{formatter:v=>v==null?'—':v.toFixed(1)+'%'}}}));chOne.render();
const sparks=[];
const chDona=new ApexCharts($('ch-dona'),base({chart:{type:'donut',height:280,fontFamily:FONT,foreColor:P.mute},series:[0,0,0,0],labels:['Con pérdida','0 a 10%','10 a 20%','Más de 20%'],
 colors:[P.red,P.series[1],P.series[0],P.green],stroke:{width:2,colors:[getComputedStyle(document.body).getPropertyValue('--card').trim()||'#fff']},legend:{position:'bottom'},
 plotOptions:{pie:{donut:{size:'68%',labels:{show:true,total:{show:true,label:'Emisoras',color:P.mute,fontWeight:500},value:{fontSize:'28px',fontWeight:700,color:P.ink}}}}},tooltip:{theme:P.tip,y:{formatter:v=>v+' emisoras'}}}));chDona.render();
const chStack=new ApexCharts($('ch-stack'),base({chart:{type:'bar',height:280,stacked:true,fontFamily:FONT,toolbar:{show:false},foreColor:P.mute},series:[],colors:[P.green,P.red],
 plotOptions:{bar:{borderRadius:P.barR,columnWidth:'55%'}},xaxis:{categories:[]},yaxis:{labels:{formatter:v=>Math.round(v)}},tooltip:{theme:P.tip,shared:true,intersect:false,y:{formatter:v=>v+' trimestres'}}}));chStack.render();
const CLARO={mute:P.mute,grid:P.grid,ink:P.ink,tip:P.tip};
function aplicarTema(){const o=document.body.classList.contains('dark')?P.dark:CLARO;Object.assign(P,o);
 const card=getComputedStyle(document.body).getPropertyValue('--card').trim();
 for(const c of [chMed,chRank,chOne,chStack])c.updateOptions({chart:{foreColor:P.mute},grid:{borderColor:P.grid},tooltip:{theme:P.tip}},false,false);
 chDona.updateOptions({chart:{foreColor:P.mute},stroke:{colors:[card]},tooltip:{theme:P.tip},plotOptions:{pie:{donut:{labels:{show:true,total:{show:true,label:'Emisoras',color:P.mute},value:{color:P.ink}}}}}},false,false);
 $('modo').textContent=document.body.classList.contains('dark')?'Modo claro':'Modo oscuro';todo()}
$('modo').onclick=()=>{document.body.classList.toggle('dark');aplicarTema()};
// --------- slider
const sl=$('sl');sl.min=0;sl.max=Math.round((MAX-MIN)/DAY);sl.value=Math.round((F-MIN)/DAY);
$('ticks').innerHTML=`<span>${fmt(MIN)}</span><span>Hoy · ${fmt(HOY)}</span><span>${fmt(MAX)}</span>`;
const atajos=[['Hoy',HOY],['1 nov 2026',Date.parse('2026-11-01')],['9 dic 2026',Date.parse('2026-12-09')],['Inicio de 2020',Date.parse('2020-01-01')]];
$('chips').innerHTML=atajos.map(([l,t])=>`<button data-t="${t}">${l}</button>`).join('');
document.querySelectorAll('#chips button').forEach(b=>b.onclick=()=>{sl.value=Math.round((+b.dataset.t-MIN)/DAY);mover()});
sl.oninput=mover;
$('gen').textContent='Generado '+D.kpi.generated;
function mover(){F=MIN+(+sl.value)*DAY;todo()}
function todo(){
 const v=vis(F), ult=ultimo(F), emis=new Set(v.map(r=>r.t)).size;
 const pend=ROWS.filter(r=>r.qt<=F&&r.av>F), ultQ=v.length?Math.max(...v.map(r=>r.qt)):null;
 $('fecha').textContent=fmt(F);
 $('exp').innerHTML=F>HOY?`Fecha futura: así se verá el producto cuando se cumpla el rezago de ${LAG} días.`:
  `Solo se ven los trimestres con <b>available_date ≤ ${fmt(F)}</b>. ${pend.length?`${pend.length} trimestres ya cerraron pero todavía no se pueden usar.`:'No hay trimestres cerrados pendientes.'}`;
 // KPIs con mini tendencia
 const qn=QS.filter(q=>v.some(r=>r.qt===q));
 const serieK=k=>medianas(F,k).slice(-12);
 const nmS=serieK('nm'), ryS=serieK('ry');
 const cnt=qn.slice(-12).map(q=>({x:q,y:v.filter(r=>r.qt===q).length}));
 const k=[['a','Emisoras con datos',emis+' / '+D.kpi.catalog,`${v.length.toLocaleString('es-MX')} trimestres visibles`,cnt],
  ['b','Último trimestre publicado',ultQ?trim(ultQ):'—',ultQ?`Disponible desde ${fmt(Math.min(...v.filter(r=>r.qt===ultQ).map(r=>r.av)))}`:'',null],
  ['c','Margen neto (mediana)',nmS.length?nmS.at(-1).y.toFixed(1)+'%':'—',nmS.length>1?delta(nmS):'',nmS],
  ['d','Ingresos YoY (mediana)',ryS.length?ryS.at(-1).y.toFixed(1)+'%':'—',ryS.length>1?delta(ryS):'',ryS]];
 if(!$('kpis').children.length){$('kpis').innerHTML=k.map((x,i)=>`<div class="card kpi"><div class="ico">${ICON[x[0]]}</div><div class="l">${x[1]}</div><div class="v" id="kv${i}"></div><div class="d" id="kd${i}"></div><div class="spark" id="ks${i}"></div></div>`).join('');
  k.forEach((x,i)=>{if(!x[4]){$('ks'+i).style.display='none';sparks.push(null);return}const c=new ApexCharts($('ks'+i),{chart:{type:x[0]==='a'?'bar':'area',height:48,sparkline:{enabled:true},animations:{enabled:false}},series:[{data:[]}],colors:[P.series[i%P.series.length]],stroke:{curve:'smooth',width:2},fill:{type:x[0]==='a'?'solid':'gradient',opacity:.85,gradient:{opacityFrom:.35,opacityTo:0}},plotOptions:{bar:{borderRadius:2,columnWidth:'60%'}},tooltip:{enabled:false}});c.render();sparks.push(c)})}
 k.forEach((x,i)=>{$('kv'+i).textContent=x[2];$('kd'+i).innerHTML=x[3];if(sparks[i])sparks[i].updateSeries([{data:x[4]}])});
 // mercado
 const ks=modeM==='m'?['gm','om','nm']:['ry','ey'];
 chMed.updateOptions({colors:modeM==='m'?P.series:[P.series[0],P.series[3]],annotations:linea(F)},false,false);
 chMed.updateSeries(ks.map(k=>({name:NAME[k],data:medianas(F,k)})));
 // ranking
 const T=ult.filter(r=>r.nm!=null).sort((a,b)=>b.nm-a.nm), R=T.length>15?[...T.slice(0,10),...T.slice(-5)]:T;
 $('rank-sub').textContent=`Las 10 más altas y las 5 más bajas de ${T.length} emisoras, último trimestre conocido; en rojo, pérdida`;
 chRank.updateOptions({colors:R.map(r=>r.nm<0?P.red:P.series[0]),xaxis:{categories:R.map(r=>r.t)}},false,false);
 chRank.updateSeries([{name:'Margen neto',data:R.map(r=>+(r.nm*100).toFixed(1))}]);
 // dona y barras apiladas
 const b=[0,0,0,0];for(const r of ult){if(r.nm==null)continue;b[r.nm<0?0:r.nm<.1?1:r.nm<.2?2:3]++}chDona.updateSeries(b);
 const yrs=[...new Set(v.map(r=>new Date(r.qt).getUTCFullYear()))].sort();
 chStack.updateOptions({xaxis:{categories:yrs.map(String)}},false,false);
 chStack.updateSeries([{name:'Con ganancia',data:yrs.map(y=>v.filter(r=>new Date(r.qt).getUTCFullYear()===y&&r.nm!=null&&r.nm>=0).length)},{name:'Con pérdida',data:yrs.map(y=>v.filter(r=>new Date(r.qt).getUTCFullYear()===y&&r.nm!=null&&r.nm<0).length)}]);
 calidad(v,pend);showOne();tabla();
}
function trim(t){const d=new Date(t);return (Math.floor(d.getUTCMonth()/3)+1)+'T '+d.getUTCFullYear()}
function delta(s){const a=s.at(-1).y,b=s.at(-5)?s.at(-5).y:s[0].y,d=a-b;return `<span class="${d>=0?'pos':'neg'}">${d>=0?'▲':'▼'} ${Math.abs(d).toFixed(1)} pts</span> contra un año antes`}
function calidad(v,pend){
 const c=[[D.pit.lag_min===D.pit.lag_max?P.green:P.amber,D.pit.lag_min===D.pit.lag_max?`Rezago fijo de ${LAG} días`:`El rezago varía entre ${D.pit.lag_min} y ${D.pit.lag_max} días`,`El ${D.pit.pct_lag_ok}% de las filas usa ${LAG} días`],
  [pend.length?P.amber:P.green,`${pend.length} trimestres cerrados sin publicar`,'A esta fecha, la regla PIT los excluye'],
  [P.red,`${D.kpi.missing.length} emisoras sin datos en la API`,D.kpi.missing.map(esc).join(', ')]];
 $('calidad').innerHTML=c.map(([col,b,s])=>`<div class="card"><span class="dot" style="background:${col}"></span><div><b>${b}</b><span>${s}</span></div></div>`).join('');
}
// explorar
const tk=Object.keys(D.series).sort();$('sel').innerHTML=tk.map(t=>`<option>${esc(t)}</option>`).join('');$('sel').value=tk.includes('WALMEX')?'WALMEX':tk[0];
function showOne(){const t=$('sel').value,s=D.series[t],ks=modeE==='m'?['gm','om','nm']:['ry','ey'];
 chOne.updateOptions({colors:modeE==='m'?P.series:[P.series[0],P.series[3]],annotations:linea(F),xaxis:{type:'datetime',min:Date.parse(s.q[0]),max:Math.max(F,Date.parse(s.a.at(-1)))}},false,false);
 chOne.updateSeries(ks.map(k=>({name:NAME[k],data:s.q.map((q,i)=>({x:Date.parse(s.a[i]),y:s[k][i]==null?null:+(s[k][i]*100).toFixed(2)}))})));
 const u=ultimo(F).find(r=>r.t===t);
 $('facts').innerHTML=u?`<div><b>${u.q}</b>último trimestre conocido</div><div><b>${pc(u.nm)}</b>margen neto</div><div><b>${pc(u.ry)}</b>ingresos YoY</div><div><b>${s.a.filter(a=>Date.parse(a)<=F).length}</b>trimestres visibles</div>`:'<div><b>—</b>sin trimestres publicados a esta fecha</div>'}
$('sel').onchange=showOne;
$('tks').innerHTML=tk.map(t=>`<option value="${esc(t)}">`).join('');
$('buscar').onchange=()=>{const t=$('buscar').value.trim().toUpperCase();if(tk.includes(t)){$('sel').value=t;showOne();$('emisora').scrollIntoView({behavior:'smooth'})}};
function seg(id,fn){document.querySelectorAll('#'+id+' button').forEach(b=>b.onclick=()=>{document.querySelectorAll('#'+id+' button').forEach(x=>x.classList.toggle('on',x===b));fn(b.dataset.m)})}
seg('segm',m=>{modeM=m;todo()});seg('sege',m=>{modeE=m;showOne()});
// tabla
const cols=[['t','Emisora'],['q','Trimestre'],['av','Disponible desde'],['gm','M. bruto'],['om','M. operativo'],['nm','M. neto'],['ry','Ingresos YoY'],['ey','Utilidad YoY']];
function tabla(){const f=$('q').value.trim().toUpperCase();
 const rows=ultimo(F).filter(r=>r.t.includes(f)).sort((a,b)=>{const A=a[sk],B=b[sk];if(A==null)return 1;if(B==null)return -1;return(A>B?1:A<B?-1:0)*dir});
 $('th').innerHTML='<tr>'+cols.map(([k,n])=>`<th data-k="${k}">${n}${k===sk?(dir<0?' ↓':' ↑'):''}</th>`).join('')+'</tr>';
 $('tb').innerHTML=rows.map(r=>`<tr><td>${esc(r.t)}</td><td>${r.q}</td><td>${fmt(r.av)}</td>${['gm','om','nm','ry','ey'].map(k=>`<td class="${r[k]==null?'na':r[k]<0?'neg':''}">${pc(r[k])}</td>`).join('')}</tr>`).join('');
 document.querySelectorAll('#th th').forEach(h=>h.onclick=()=>{const k=h.dataset.k;if(k===sk)dir=-dir;else{sk=k;dir=-1}tabla()})}
$('q').oninput=tabla;
todo();
</script></body></html>
"""

if __name__ == "__main__":
    main(sys.argv[1:])
