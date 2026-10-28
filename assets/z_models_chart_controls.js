/* Native cached Models inspection. Decoded Plotly values; no source mutations. */
(function () {
    'use strict';
    const installed = new WeakMap();
    const numberText = value => value === null || value === undefined || value === '' || typeof value === 'boolean' ? 'Unavailable' : Number.isFinite(Number(value)) ? Number(value).toLocaleString(undefined, {maximumFractionDigits: 4}) : 'Unavailable';
    const at = (values, index) => values && typeof values.length === 'number' ? values[index] : undefined;
    const utcText = value => {
        const millis = typeof value === 'number' ? value : utcMillis(value);
        return Number.isFinite(millis) ? new Intl.DateTimeFormat('en-US', {month:'short',day:'numeric',year:'numeric',hour:'2-digit',minute:'2-digit',hourCycle:'h23',timeZone:'UTC'}).format(new Date(millis)) + ' UTC' : 'Clock unavailable';
    };
    function utcMillis(value) {
        if (typeof value === 'number') return value;
        let text = String(value).trim().replace(' ', 'T');
        // These date axes are emitted only from qualified UTC backtest targets.
        if (!/(Z|[+-]\d\d:\d\d)$/.test(text)) text += 'Z';
        return Date.parse(text);
    }
    function graphFor(target) {
        const wrapper = target.closest && target.closest('.gp-model-chart-keyboard');
        return wrapper && wrapper.querySelector('.js-plotly-plot');
    }
    function pointCount(gd) {
        const trace = gd._fullData && gd._fullData[0];
        if (!trace || trace.visible === false || trace.visible === 'legendonly') return 0;
        if (trace.type === 'histogram') return (gd.calcdata[0] || []).length;
        return (trace.x && trace.x.length) || (trace.y && trace.y.length) || 0;
    }
    function clearInspection(gd) {
        const host=gd.closest('.gp-models-graph'),output=host&&document.getElementById(host.id+'-inspected');
        const active=gd.dataset.inspectedIndex !== undefined || (output && output.textContent) || (gd._hoverdata && gd._hoverdata.length);
        delete gd.dataset.inspectedIndex;
        if(output && output.textContent)output.textContent='';
        if(active && window.Plotly)window.Plotly.Fx.unhover(gd);
    }
    function pendingOrDifferentRegion(gd) {
        const host=gd.closest('.gp-models-graph'),selector=document.getElementById('region-selector');
        const region=selector&&selector.value,revision=gd.layout&&gd.layout.uirevision;
        return (host && host.getAttribute('data-dash-is-loading')==='true') ||
            (region && (!revision || String(revision).split(':')[0]!==region));
    }
    function inspectedPoint(gd, index) {
        const trace = gd._fullData[0], calc = gd.calcdata && gd.calcdata[0] && gd.calcdata[0][index];
        let x = at(trace.x, index), y = at(trace.y, index);
        if (calc) {
            if (trace.type === 'bar' || trace.type === 'histogram') {
                if (trace.orientation === 'h') { x = calc.s !== undefined ? calc.s : x; }
                else { x = calc.p !== undefined ? calc.p : x; y = calc.s !== undefined ? calc.s : y; }
            } else {
                if (gd._fullLayout.xaxis.type !== 'category' && calc.x !== undefined) x = calc.x;
                if (gd._fullLayout.yaxis.type !== 'category' && calc.y !== undefined) y = calc.y;
            }
        }
        return {trace, x, y};
    }
    function inspect(gd, index) {
        const count = pointCount(gd); if (!count || pendingOrDifferentRegion(gd)) {clearInspection(gd);return;}
        index = Math.max(0, Math.min(count - 1, index));
        gd.dataset.inspectedIndex = String(index);
        if (window.Plotly) window.Plotly.Fx.hover(gd, [{curveNumber:0,pointNumber:index}]);
        const host = gd.closest('.gp-models-graph'), output = host && document.getElementById(host.id + '-inspected');
        if (!output) return;
        const point = inspectedPoint(gd, index), trace = point.trace;
        if (host.id === 'tab3-residuals-hist') {
            const bin = at(trace.customdata, index);
            output.textContent = Array.isArray(bin) ? numberText(bin[0]) + ' ≤ error ' + (index === count - 1 ? '≤ ' : '< ') + numberText(bin[1]) + ' MW · Count ' + numberText(point.y) : 'Error bin ' + numberText(point.x) + ' MW · Count ' + numberText(point.y);
        } else if (host.id === 'tab3-shap') {
            output.textContent = String(point.y === undefined ? 'Feature unavailable' : point.y) + ' · Relative importance ' + numberText(point.x);
        } else if (gd._fullLayout.xaxis.type === 'date') {
            output.textContent = (gd.layout.meta && gd.layout.meta.diagnostic_utc_qualified ? utcText(point.x) : 'Clock unavailable') + ' · Error ' + numberText(point.y) + ' MW · ' + trace.name;
        } else if (host.id === 'tab3-error-heatmap') {
            output.textContent = 'Target hour ' + numberText(point.x) + ':00 UTC · MAE ' + numberText(point.y) + ' MW · n = ' + numberText(at(trace.customdata, index));
        } else {
            output.textContent = 'Predicted demand ' + numberText(point.x) + ' MW · Error ' + numberText(point.y) + ' MW · ' + trace.name;
        }
    }
    function renderLegend(gd, state) {
        const host = gd.closest('.gp-models-graph'), legend = host && document.getElementById(host.id + '-legend');
        if (!legend || !gd._fullData) return;
        const entries = host.id === 'tab3-shap' ? [] : gd._fullData.filter(trace => trace.visible !== false && trace.name).map(trace => ({name:trace.name,color:(trace.line && trace.line.color) || (trace.marker && trace.marker.color) || '#5b9fff'}));
        const key = JSON.stringify(entries); if (state.legend === key) return; state.legend = key;
        legend.replaceChildren();
        entries.forEach(entry => {const item=document.createElement('span'),line=document.createElement('i');line.style.backgroundColor=entry.color;item.append(line,document.createTextNode(entry.name));legend.append(item);});
    }
    function figureSignature(gd) {
        return JSON.stringify({revision:gd.layout&&gd.layout.uirevision,meta:gd.layout&&gd.layout.meta,traces:(gd.data||[]).map(trace=>({name:trace.name,type:trace.type,x:trace.x,y:trace.y,customdata:trace.customdata}))});
    }
    function install(gd) {
        if (installed.has(gd) || typeof gd.on !== 'function') return;
        const host = gd.closest('.gp-models-graph'); if (!host) return;
        const state = {queued:false,running:false,legend:null,data:gd.data,revision:gd.layout&&gd.layout.uirevision,signature:figureSignature(gd)}; installed.set(gd,state);
        // A new Plotly host can reuse its server-rendered readout node. Clear it
        // even when the replacement figure has valid same-region values.
        clearInspection(gd);
        const resize = function () {
            if (state.queued) return; state.queued = true;
            requestAnimationFrame(function () {
                state.queued = false;
                if (!host.isConnected || !window.Plotly || !gd._fullLayout || state.running) return;
                const width = host.clientWidth, height = host.clientHeight;
                if (width < 1 || height < 1) return;
                if (Math.abs(gd._fullLayout.width - width) < 1 && Math.abs(gd._fullLayout.height - height) < 1) return;
                state.running = true;
                Promise.resolve(window.Plotly.relayout(gd, {width:width,height:height,autosize:true})).finally(function () {state.running=false;});
            });
        };
        const observer = new ResizeObserver(resize); observer.observe(host); state.observer = observer;
        gd.on('plotly_afterplot', function () {
            const revision=gd.layout&&gd.layout.uirevision,signature=figureSignature(gd);
            const changed=state.data!==gd.data || state.revision!==revision || state.signature!==signature;
            state.data=gd.data;state.revision=revision;state.signature=signature;
            if(changed || !pointCount(gd) || pendingOrDifferentRegion(gd))clearInspection(gd);
            renderLegend(gd,state);resize();
        });
        gd.tabIndex = -1; // The persistent server-rendered wrapper is the single Tab stop.
        renderLegend(gd,state); resize();
        if(!pointCount(gd) || pendingOrDifferentRegion(gd))clearInspection(gd);
    }
    function attach() {document.querySelectorAll('.gp-models-graph .js-plotly-plot').forEach(install);}
    document.addEventListener('keydown', function (event) {
        const gd = graphFor(event.target); if (!gd) return;
        if (event.key === 'Escape') {clearInspection(gd);return;}
        if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
        event.preventDefault(); const count=pointCount(gd);if(!count||pendingOrDifferentRegion(gd)){clearInspection(gd);return;}const index=Number(gd.dataset.inspectedIndex)||0;
        inspect(gd,event.key==='Home'?0:event.key==='End'?count-1:index+(event.key==='ArrowRight'?1:-1));
    });
    document.addEventListener('change',function(event){
        if(event.target.id==='region-selector')document.querySelectorAll('.gp-models-graph .js-plotly-plot').forEach(clearInspection);
    },true);
    document.addEventListener('click', function (event) {
        const button = event.target.closest && event.target.closest('[data-model-chart][data-chart-action]'); if (!button) return;
        const host=document.getElementById(button.dataset.modelChart),gd=host&&host.querySelector('.js-plotly-plot');if(!gd||!window.Plotly||!gd._fullLayout)return;
        if(button.dataset.chartAction==='reset'){window.Plotly.relayout(gd,{'xaxis.autorange':true,'yaxis.autorange':true});return;}
        const axis=gd._fullLayout.xaxis,range=axis.range;if(!range||range.length!==2)return;
        const date=axis.type==='date';if(date && !(gd.layout.meta && gd.layout.meta.diagnostic_utc_qualified))return;const convert=value=>date?(typeof axis.r2l==='function'?axis.r2l(value):utcMillis(value)):Number(value);
        const lo=convert(range[0]),hi=convert(range[1]);if(!Number.isFinite(lo)||!Number.isFinite(hi)||lo===hi)return;const delta=(hi-lo)/4;
        window.Plotly.relayout(gd,{'xaxis.range':date?[new Date(lo+delta).toISOString(),new Date(hi-delta).toISOString()]:[lo+delta,hi-delta],'xaxis.autorange':false});
    });
    let queued=false;new MutationObserver(function(records){
        records.forEach(function(record){
            if(record.type==='attributes' && record.target.getAttribute('data-dash-is-loading')==='true'){
                const host=record.target.closest&&record.target.closest('.gp-models-graph'),gd=host&&host.querySelector('.js-plotly-plot');
                if(gd)clearInspection(gd);
            }
        });
        if(queued)return;queued=true;requestAnimationFrame(function(){queued=false;attach();});
    }).observe(document.documentElement,{childList:true,subtree:true,attributes:true,attributeFilter:['data-dash-is-loading']});
    document.addEventListener('DOMContentLoaded',attach);attach();
}());
