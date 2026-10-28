/* Native Dash/Plotly inspection and accessible explicit-detail focus. No data sources here. */
(function () {
    "use strict";
    const api = window.gridpulseReview = window.gridpulseReview || {};
    const fmt = (v, digits) => v === null || v === undefined || !Number.isFinite(Number(v)) ? "Gap" : Number(v).toLocaleString(undefined, {maximumFractionDigits: digits});
    const clock = s => new Intl.DateTimeFormat("en-US", {month:"short",day:"numeric",hour:"2-digit",minute:"2-digit",hourCycle:"h23",timeZone:"UTC"}).format(new Date(s)) + " UTC";
    api.inspect = function (data, index, events, previousCursor) {
        const dateInput=document.getElementById("explorer-start");
        if(dateInput)dateInput.dataset.controlledStart=data&&data.window_start?new Date(data.window_start).toISOString().slice(0,16):"";
        if (!data || !Array.isArray(data.rows) || !data.rows.length) return [null,"No inspected source hour is available.",0];
        const ctx = window.dash_clientside.callback_context;
        const triggered = ctx && ctx.triggered && ctx.triggered[0] ? ctx.triggered[0].prop_id : "explorer-window.data";
        let selected = triggered === "explorer-window.data" ? data.selected_index || 0 : Number(index) || 0;
        if(triggered==="explorer-window.data" && previousCursor && previousCursor.region===data.region && previousCursor.mode===data.mode){const found=data.rows.findIndex(r=>r.timestamp===previousCursor.timestamp);if(found>=0)selected=found;}
        const eventIDs = ["explorer-demand.hoverData","explorer-weather.hoverData","explorer-error.hoverData","explorer-demand.clickData","explorer-weather.clickData","explorer-error.clickData"];
        const eventIndex = eventIDs.indexOf(triggered);
        if (eventIndex >= 0 && events[eventIndex] && events[eventIndex].points && events[eventIndex].points[0]) {
            const target = Date.parse(events[eventIndex].points[0].x);
            if (Number.isFinite(target)) {
                let best = Infinity;
                data.rows.forEach((r,i) => {const delta = Math.abs(Date.parse(r.timestamp)-target);if(delta<best){best=delta;selected=i;}});
            }
        }
        selected = Math.max(0,Math.min(data.rows.length-1,selected));
        const row=data.rows[selected];
        const text=[clock(row.timestamp), data.demand_label+": "+fmt(row[data.demand_column],0)+" MW",data.weather_label+": "+fmt(row[data.variable],1)+" "+data.unit];
        if(data.mode==="error")text.push("Forecast: "+fmt(row.forecast_mw,0)+" MW","Actual − forecast: "+fmt(row.error_mw,0)+" MW");
        if(data.mode==="future" && row.generated_before_target===false)text.push("Stored before run generation");
        const repaint = attempt => {
            ["demand","weather","error"].forEach(part => {
                const host=document.getElementById("explorer-"+part),gd=host&&host.querySelector(".js-plotly-plot");
                if(!gd || !gd.layout || !gd.layout.meta || gd.layout.meta.window_key!==data.key+":"+part || !window.Plotly)return;
                window.Plotly.relayout(gd,{shapes:[{type:"line",xref:"x",yref:"paper",x0:row.timestamp,x1:row.timestamp,y0:0,y1:1,line:{color:"#adc4d9",width:1,dash:"dot"}}]});
                gd.data.forEach((trace,i)=>{if(trace.meta&&trace.meta.cursor_column)window.Plotly.restyle(gd,{x:[[row.timestamp]],y:[[row[trace.meta.cursor_column]]] },[i]);});
                gd.setAttribute("tabindex","0");gd.setAttribute("role","img");gd.setAttribute("aria-label",part+" plot. Use arrow keys, Home and End to inspect the shared UTC hour.");
            });
            if(attempt<3 && triggered==="explorer-window.data")setTimeout(()=>repaint(attempt+1),100);
        };
        repaint(0);
        return [{region:data.region,mode:data.mode,timestamp:row.timestamp,index:selected},text.join(" · ")+" · Rounded display; full precision in Exact values",selected];
    };
    // dbc.Input debounces valid React values; native datetime segments can
    // remain blank/invalid without publishing that DOM edit. Commit the raw
    // edit only on blur/Enter, so validation never interrupts typing.
    function commitDate(el){
        if(!el || el.id!=="explorer-start")return;
        const raw=el.value;
        let scope={};try{scope=JSON.parse(el.dataset.editScope||"{}");}catch(e){}
        window.dash_clientside.set_props("explorer-date-edit",{data:{value:raw,scope:scope,nonce:Date.now()}});
        if(!raw || !el.validity.valid)el.value=el.dataset.controlledStart||"";
    }
    document.addEventListener("focusin",function(e){
        if(e.target.id!=="explorer-start")return;
        const selected=document.querySelector('#explorer-mode input:checked');
        e.target.dataset.editScope=JSON.stringify({region:document.getElementById("region-selector").value,mode:selected&&selected.value,source:document.getElementById("explorer-source").value,window_start:e.target.dataset.controlledStart||""});
    });
    document.addEventListener("focusout",function(e){commitDate(e.target);});
    document.addEventListener("keydown",function(e){
        if(e.key==="Enter" && e.target.id==="explorer-start")commitDate(e.target);
        const graph=e.target.closest&&e.target.closest(".gp-explorer-graph .js-plotly-plot");
        if(!graph || !["ArrowLeft","ArrowRight","Home","End"].includes(e.key))return;
        const input=document.getElementById("explorer-hour");if(!input)return;
        e.preventDefault();const max=Number(input.max)||0,value=Number(input.value)||0;
        const next=e.key==="Home"?0:e.key==="End"?max:Math.max(0,Math.min(max,value+(e.key==="ArrowRight"?1:-1)));
        window.dash_clientside.set_props("explorer-hour",{value:next});
    });
    api.reveal = function(targetID,focusID){
        requestAnimationFrame(()=>{const target=document.getElementById(targetID),focus=document.getElementById(focusID||targetID);if(!target)return;
            if(focus)focus.focus({preventScroll:true});const r=target.getBoundingClientRect();
            if(r.top<12 || r.bottom>innerHeight-16)target.scrollIntoView({block:"start",behavior:matchMedia("(prefers-reduced-motion: reduce)").matches?"auto":"smooth"});
        });
    };
}());
