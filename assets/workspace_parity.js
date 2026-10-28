/* Native Forecast target inspection; uses only the rendered selected series. */
(function(){
    "use strict";
    const graph=()=>document.querySelector("#outlook-chart .js-plotly-plot");
    const api=window.gridpulseReview=window.gridpulseReview||{};
    // EIA and cached Forecast/Plotly date coordinates represent UTC. An offsetless
    // producer timestamp must not be reinterpreted in the browser's local zone.
    const utcNumber=value=>{
        if(typeof value==="number")return Number.isFinite(value)?value:NaN;
        if(value instanceof Date)return value.getTime();
        if(typeof value!=="string")return NaN;
        const text=value.trim().replace(" ","T");
        return Date.parse(/(?:Z|[+-]\d{2}:?\d{2})$/i.test(text)?text:text+"Z");
    };
    api.forecastZoomRange=function(fig){
        const axis=fig&&fig.layout&&fig.layout.xaxis||{};
        const previous=axis.autorange===false&&Array.isArray(axis.range)?axis.range:null;
        const xs=(previous||((fig&&fig.data)||[]).filter(t=>t.visible!==false).flatMap(t=>t.x||[])).map(utcNumber).filter(Number.isFinite);
        if(xs.length<2)return null;
        const lo=Math.min(...xs),hi=Math.max(...xs),pad=(hi-lo)*.18;
        return [new Date(lo+pad).toISOString(),new Date(hi-pad).toISOString()];
    };
    const registered=new WeakSet();
    const resizeQueued=new WeakSet();
    const resizeBusy=new WeakSet();
    function resizeForecast(){
        const p=graph(),host=document.getElementById("outlook-chart");
        if(!p||!host||!window.Plotly||resizeQueued.has(p)||resizeBusy.has(p))return;
        resizeQueued.add(p);
        requestAnimationFrame(()=>{
            resizeQueued.delete(p);
            const box=host.getBoundingClientRect(),width=Math.round(box.width),height=Math.round(box.height);
            if(width<20||height<20||!p._fullLayout||resizeBusy.has(p))return;
            if(Math.abs(p._fullLayout.width-width)<=1&&Math.abs(p._fullLayout.height-height)<=1)return;
            resizeBusy.add(p);
            Promise.resolve(window.Plotly.relayout(p,{width:width,height:height,autosize:true})).then(()=>{resizeBusy.delete(p);resizeForecast();},()=>{resizeBusy.delete(p);});
        });
    }
    const sizeObserver=typeof ResizeObserver!=="undefined"?new ResizeObserver(resizeForecast):null;
    let observedHost=null;
    function attach(){
        const host=document.getElementById("outlook-chart"),p=graph();
        if(host&&host!==observedHost){if(observedHost&&sizeObserver)sizeObserver.unobserve(observedHost);observedHost=host;if(sizeObserver)sizeObserver.observe(host);}
        if(!p||!p.on)return;
        if(!registered.has(p)){registered.add(p);p.on("plotly_afterplot",()=>{sync(p);resizeForecast();});}
        sync(p);resizeForecast();
    }
    const selected=p=>(p.data||[]).findIndex(t=>typeof t.name==="string"&&t.name.endsWith(" Forecast"));
    const inspectedScopes=new WeakMap();
    function controlsScope(){
        const region=document.getElementById("region-selector"),model=document.getElementById("outlook-model"),hour=document.querySelector("#outlook-horizon input:checked");
        return [region&&region.value,model&&model.value,hour&&hour.value].join(":");
    }
    function currentScope(p){
        const m=p.layout&&p.layout.meta;
        if(!m||!m.forecast_region)return true;
        return [m.forecast_region,m.forecast_requested,String(m.forecast_hours)].join(":")===controlsScope();
    }
    function clearInspection(p,pending){
        const out=document.getElementById("forecast-chart-inspection"),host=document.getElementById("forecast-chart-keyboard");
        const caption=pending?"Updating selected forecast…":p&&selected(p)>=0?"Inspect with ← → · target timestamps UTC · MW":"Target timestamps UTC · MW";
        if(out&&out.textContent!==caption)out.textContent=caption;
        if(host){delete host.dataset.targetKey;delete host.dataset.targetIndex;}
        if(p&&p._fullLayout&&window.Plotly&&window.Plotly.Fx)window.Plotly.Fx.unhover(p);
    }
    function sync(p){
        const ci=selected(p),trace=ci>=0?p.data[ci]:null,prior=inspectedScopes.get(p),host=document.getElementById("outlook-chart");
        const pending=!!(host&&host.getAttribute("data-dash-is-loading")==="true")||!currentScope(p);
        const yText=trace?(prior&&prior.yRef===trace.y?prior.yText:JSON.stringify(trace.y)):"";
        const meta=p.layout&&p.layout.meta||{};
        const key=[controlsScope(),p.layout&&p.layout.uirevision,JSON.stringify(meta),trace&&trace.name,trace&&trace.x&&trace.x.length,trace&&trace.x&&trace.x[0],trace&&trace.x&&trace.x[trace.x.length-1],yText].join("|");
        inspectedScopes.set(p,{key:key,pending:pending,yRef:trace&&trace.y,yText:yText});
        if(!prior||prior.key!==key||prior.pending!==pending)clearInspection(p,pending);
        p.setAttribute("tabindex","-1");p.setAttribute("role","img");p.setAttribute("aria-label","Demand outlook chart. Arrow keys inspect selected forecast targets; Home and End jump to its bounds; Escape clears inspection.");p.setAttribute("aria-describedby","forecast-chart-inspection");
        const actual=(p.data||[]).find(t=>t.name==="Actual");
        const a=document.getElementById("forecast-legend-actual"), f=document.getElementById("forecast-legend-selected");
        if(a){a.setAttribute("aria-pressed",String(!!actual&&actual.visible!==false));a.disabled=!actual;}
        if(f){f.setAttribute("aria-pressed",String(ci>=0&&p.data[ci].visible!==false));f.disabled=ci<0;}
    }
    document.addEventListener("click",function(e){
        const context=e.target.closest&&e.target.closest("#forecast-context-open-explorer");if(context){const entry=document.getElementById("forecast-open-explorer");if(entry)entry.click();return;}
        const jump=e.target.closest&&e.target.closest("#overview-inspect-evidence");
        if(jump){const detail=document.getElementById("overview-evidence-disclosure");if(detail){detail.open=true;const summary=detail.querySelector("summary");if(summary)summary.focus({preventScroll:true});detail.scrollIntoView({block:"start",behavior:matchMedia("(prefers-reduced-motion:reduce)").matches?"auto":"smooth"});}return;}
        const button=e.target.closest&&e.target.closest("#forecast-legend-actual,#forecast-legend-selected"),p=graph();
        if(!button||!p||!window.Plotly)return;
        if(button.id==="forecast-legend-actual"){
            const check=document.getElementById("outlook-show-history");if(check)check.click();
        }else{
            const ci=selected(p);if(ci<0)return;
            const next=p.data[ci].visible===false, indexes=p.data.map((t,i)=>t.name!=="Actual"?i:-1).filter(i=>i>=0);
            window.Plotly.restyle(p,{visible:next},indexes).then(()=>sync(p));
        }
    });
    document.addEventListener("keydown",function(e){
        const host=e.target.closest&&e.target.closest("#forecast-chart-keyboard");
        if(!host||!["ArrowLeft","ArrowRight","Home","End","Escape"].includes(e.key))return;
        const p=graph(),out=document.getElementById("forecast-chart-inspection");
        if(!p||!out||!window.Plotly)return;
        e.preventDefault();
        if(e.key==="Escape"){clearInspection(p,false);return;}
        const plotHost=document.getElementById("outlook-chart");
        if(plotHost&&plotHost.getAttribute("data-dash-is-loading")==="true"||!currentScope(p)){clearInspection(p,true);return;}
        const ci=selected(p);if(ci<0){out.textContent="No forecast targets are available to inspect.";return;}
        const trace=p.data[ci],xs=trace.x||[], calc=(p.calcdata||[])[ci]||[];
        if(!xs.length||trace.visible===false){out.textContent="Enable the selected forecast in the legend to inspect its targets.";return;}
        const key=trace.name+":"+xs[0]+":"+xs.length;
        let index=host.dataset.targetKey===key?Number(host.dataset.targetIndex):-1;
        if(e.key==="Home")index=0;else if(e.key==="End")index=xs.length-1;else index+=e.key==="ArrowRight"?1:-1;
        index=Math.max(0,Math.min(xs.length-1,index));
        host.dataset.targetKey=key;host.dataset.targetIndex=String(index);
        const point=calc[index],value=point&&Number(point.y),date=new Date(xs[index]);
        const clock=Number.isFinite(date.getTime())?date.toISOString().slice(0,16).replace("T"," ")+" UTC":"Target clock unavailable";
        const demand=Number.isFinite(value)?value.toLocaleString("en-US",{maximumFractionDigits:6})+" MW":"Unavailable";
        out.textContent=clock+" · "+trace.name+": "+demand;
        window.Plotly.Fx.hover(p,[{curveNumber:ci,pointNumber:index}]);
    });
    document.addEventListener("change",function(e){
        if(e.target.id==="region-selector"||e.target.id==="outlook-model"||e.target.closest&&e.target.closest("#outlook-horizon"))clearInspection(graph(),true);
    });
    let scheduled=false;
    const observer=new MutationObserver(function(){
        if(scheduled)return;scheduled=true;requestAnimationFrame(function(){scheduled=false;attach();});
    });
    function start(){observer.observe(document.body,{childList:true,subtree:true,attributes:true,attributeFilter:["data-dash-is-loading"]});attach();window.addEventListener("resize",resizeForecast);window.addEventListener("orientationchange",resizeForecast);}
    if(document.readyState==="loading")document.addEventListener("DOMContentLoaded",start);else start();
})();
