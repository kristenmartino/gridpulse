/* motion.js — choreographed entrance + interaction motion for GridPulse.
 *
 * Zero-dependency, auto-loaded from assets/ (no build step). Mirrors the
 * accessibility.js observer pattern. THREE effects, all opt-in behind the
 * user's motion preference:
 *
 *   1. Count-up   — the Overview hero demand value animates 0 → its value,
 *                   then lands on the EXACT rendered string (honesty: the
 *                   number the DOM settles on is always the true value).
 *   2. Reveal     — top-level sections of the active tab fade + rise in a
 *                   short stagger as they enter the viewport.
 *   3. Ink-bar    — the active-tab underline slides between tabs like a
 *                   grid-frequency trace instead of hard-cutting.
 *
 * GUARDRAIL: every effect early-returns under prefers-reduced-motion. When
 * motion is reduced (or JS never runs), nothing is hidden and the static
 * CSS underline remains — pure progressive enhancement.
 */
(function () {
    "use strict";

    var mql = window.matchMedia ? window.matchMedia("(prefers-reduced-motion: reduce)") : null;
    function reduced() { return !!(mql && mql.matches); }

    // Signals the stylesheet that JS-driven motion is live, so the reveal
    // hidden-state (.gp-motion .gp-reveal { opacity: 0 }) only ever applies
    // when we are actually going to animate it back in.
    function enableMotionCss() {
        if (!reduced()) document.documentElement.classList.add("gp-motion");
    }

    var EASE_OUT = function (t) { return 1 - Math.pow(1 - t, 3); }; // cubic-out

    // ── 1. Hero count-up ────────────────────────────────────────────
    // The stable #overview-metrics-bar container survives re-renders; we
    // stamp the last-counted value on it so auto-refreshes of the SAME
    // reading don't re-trigger, while a genuine region/value change does.
    function runCountUp() {
        if (reduced()) return;
        var container = document.getElementById("overview-metrics-bar");
        if (!container) return;
        // Re-entrancy guard: the frame loop below writes textContent, which
        // trips our own MutationObserver → onDomChange → runCountUp. Without
        // this flag that re-entry reads a MID-animation value as the new
        // target and ratchets the hero down toward 0 (real bug seen: "1 MW").
        if (container.dataset.gpHeroAnimating === "1") return;
        var el = container.querySelector(".gp-metric-value--hero");
        if (!el) return;
        var finalText = el.textContent.trim();
        var num = parseFloat(finalText.replace(/[^0-9.\-]/g, ""));
        if (!isFinite(num) || num === 0) return;               // "—" / 0 → skip
        if (container.dataset.gpHeroCounted === finalText) return;
        container.dataset.gpHeroCounted = finalText;
        container.dataset.gpHeroAnimating = "1";

        var dur = 850, startTs = null;
        function frame(ts) {
            if (startTs === null) startTs = ts;
            var p = Math.min((ts - startTs) / dur, 1);
            if (p < 1) {
                el.textContent = Math.round(num * EASE_OUT(p)).toLocaleString("en-US");
                requestAnimationFrame(frame);
            } else {
                el.textContent = finalText;                    // land on the truth
                container.dataset.gpHeroAnimating = "0";
            }
        }
        requestAnimationFrame(frame);
    }

    // ── 2. Staggered section reveals ────────────────────────────────
    // Load-time entrance (NOT scroll-gated): each newly-rendered top-level
    // section of a tab is hidden, then released a beat later with a capped
    // per-index stagger. Deliberately observer-FREE — an IntersectionObserver
    // that never fires (0-size or backgrounded viewport) would strand content
    // invisible, so release runs on a timer that always fires.
    function scanReveals() {
        if (reduced()) return;
        var targets = document.querySelectorAll(
            ".gp-page .gp-section-stack > *:not(.gp-reveal)"
        );
        if (!targets.length) return;
        var fresh = [];
        targets.forEach(function (node, i) {
            node.classList.add("gp-reveal");
            // Cap the stagger so a long stack doesn't drift too far.
            node.style.transitionDelay = Math.min(i * 70, 420) + "ms";
            fresh.push(node);
        });
        // Release after a short beat so the hidden state paints first and the
        // transition plays. setTimeout (not rAF) keeps working in a
        // backgrounded tab, so content is never left stranded invisible.
        setTimeout(function () {
            fresh.forEach(function (n) { n.classList.add("in-view"); });
        }, 40);
    }

    // Ultimate safety net — nothing stays hidden past this, whatever happens.
    function revealAllFallback() {
        document.querySelectorAll(".gp-reveal:not(.in-view)").forEach(function (n) {
            n.classList.add("in-view");
        });
    }

    // ── 3. Sliding tab ink-bar ──────────────────────────────────────
    // Positions the .nav-tabs::after bar under the active pill by writing
    // --ink-x / --ink-w; CSS transitions the transform for the slide.
    function positionInk() {
        if (reduced()) return;
        var tabs = document.querySelector(".nav-tabs");
        if (!tabs) return;
        var active = tabs.querySelector(".nav-link.active");
        if (!active) return;
        var tabsRect = tabs.getBoundingClientRect();
        var aRect = active.getBoundingClientRect();
        // Bail on a not-yet-laid-out measurement (0-width) — engaging the
        // sliding bar with a zero rect would leave an invisible/mis-placed
        // indicator. The static per-link underline stays until we can measure.
        if (aRect.width < 1) return;
        var x = aRect.left - tabsRect.left + tabs.scrollLeft;
        tabs.style.setProperty("--ink-x", x.toFixed(1) + "px");
        tabs.style.setProperty("--ink-w", aRect.width.toFixed(1) + "px");
        tabs.classList.add("gp-ink-ready");                    // swap static→sliding
    }

    // ── 4. Chart morph gate ─────────────────────────────────────────
    // Python marks a figure as wanting a data morph by putting
    // `layout.transition` on it (components/_callbacks_shared._layout). That is
    // only a REQUEST: whether the morph should actually play depends on two
    // things Python cannot see, both of which live here.
    //
    // Why a patch and not CSS or a dcc.Store:
    //   - CSS can't touch it. A Plotly transition is d3/rAF-driven JS, so the
    //     @media (prefers-reduced-motion) block in custom.css — including the
    //     `* { transition-duration: 0.01ms }` catch-all — has no effect on it.
    //     Shipping the transition without this gate would ship motion that
    //     ignores the media query.
    //   - A dcc.Store flag would have to be threaded as an Input through every
    //     chart callback (re-firing them all) and would still race the first
    //     render. Dash routes every figure update through `Plotly.react`
    //     (dcc/async-graph.js calls the bare global; `Plotly.animate` is only
    //     used when the Graph sets animate=True, which this app never does), so
    //     one wrapper here is the whole surface.
    //
    // The two suppression rules:
    //   a. prefers-reduced-motion — the user asked for no motion.
    //   b. point-count change — d3 tweens a path's `d` attribute with
    //      interpolateString, which pairs numbers POSITIONALLY. Morphing a
    //      24-point curve into a 720-point one interpolates the ~24 vertices
    //      that pair up and leaves the remaining ~696 already at their final
    //      values — a torn curve, half mid-flight and half landed. So a morph
    //      is only coherent when each trace keeps its length (region switch,
    //      model switch, hourly refresh). Horizon changes and first paints of a
    //      shorter/longer history hard-cut instead, which is the honest
    //      rendering: every frame shows real values.
    //
    // HONESTY NOTE: a permitted morph tweens between two REAL states and lands
    // on real values — the same contract as the hero count-up above. The band
    // itself is never faked: its width is the empirically-calibrated P10–P90
    // interval computed in Python, and it is not grown, eased open, or revealed
    // progressively, which would understate uncertainty mid-flight.
    // A trace's y/x arrives in one of THREE encodings, and the point count has
    // to be read out of all of them:
    //   - plain Array          — e.g. string timestamps
    //   - TypedArray           — what plotly holds after it decodes
    //   - {dtype, bdata}       — plotly.py >=6 ships numpy arrays base64-encoded,
    //                            and this object has NO .length. Reading .length
    //                            here yields undefined, which would mark every
    //                            update a shape change and silently suppress
    //                            100% of transitions — the feature would look
    //                            wired up and simply never play.
    var BYTES_PER_ITEM = { i1: 1, u1: 1, i2: 2, u2: 2, i4: 4, u4: 4, i8: 8, u8: 8, f4: 4, f8: 8 };

    function arrayLength(v) {
        if (v == null) return -1;
        if (typeof v.length === "number") return v.length;
        // plotly caches its decode on the object it was handed; prefer it.
        if (v._inputArray && typeof v._inputArray.length === "number") return v._inputArray.length;
        if (typeof v.bdata === "string" && BYTES_PER_ITEM[v.dtype]) {
            var b64 = v.bdata;
            var pad = b64.charCodeAt(b64.length - 1) === 61 ? (b64.charCodeAt(b64.length - 2) === 61 ? 2 : 1) : 0;
            var bytes = (b64.length / 4) * 3 - pad;
            return Math.floor(bytes / BYTES_PER_ITEM[v.dtype]);
        }
        return -1;
    }

    // Same trace count AND same per-trace vertex count → interpolateString pairs
    // every number → coherent morph. Anything else would tear.
    function shapeMatches(gd, nextData) {
        var before = (gd && gd.data) || [];
        var after = nextData || [];
        if (!after.length || before.length !== after.length) return false;
        for (var i = 0; i < before.length; i++) {
            var by = arrayLength(before[i] && before[i].y);
            var ay = arrayLength(after[i] && after[i].y);
            if (by < 0 || ay < 0 || by !== ay) return false;
            if (arrayLength(before[i] && before[i].x) !== arrayLength(after[i] && after[i].x)) return false;
        }
        return true;
    }

    function suppressTransition(layout) {
        if (!layout || !layout.transition) return layout;
        // Copy — never mutate the figure Dash handed us (it is retained across
        // renders, so a mutation would permanently strip the request).
        var out = {};
        for (var k in layout) if (Object.prototype.hasOwnProperty.call(layout, k)) out[k] = layout[k];
        // Carry the rest of the transition through and override only the
        // duration — rebuilding it by hand silently drops sibling keys
        // (`ordering` is load-bearing; see CHART_TRANSITION).
        var t = { duration: 0 };
        for (var tk in layout.transition) {
            if (Object.prototype.hasOwnProperty.call(layout.transition, tk) && tk !== "duration") {
                t[tk] = layout.transition[tk];
            }
        }
        out.transition = t;
        return out;
    }

    function patchPlotlyReact(P) {
        if (!P || P.__gpMotionPatched) return false;
        var original = P.react;
        if (typeof original !== "function") return false;
        P.react = function (gd, dataOrFigure, layout) {
            try {
                // Dash calls react(gd, {data, layout, frames, config}); the
                // (gd, data, layout) form is plotly's other public signature.
                var isFigureObj =
                    dataOrFigure && !Array.isArray(dataOrFigure) && typeof dataOrFigure === "object";
                var nextData = isFigureObj ? dataOrFigure.data : dataOrFigure;
                var nextLayout = isFigureObj ? dataOrFigure.layout : layout;
                if (nextLayout && nextLayout.transition && (reduced() || !shapeMatches(gd, nextData))) {
                    if (isFigureObj) {
                        var figure = {};
                        for (var k in dataOrFigure) {
                            if (Object.prototype.hasOwnProperty.call(dataOrFigure, k)) {
                                figure[k] = dataOrFigure[k];
                            }
                        }
                        figure.layout = suppressTransition(nextLayout);
                        arguments[1] = figure;
                    } else {
                        arguments[2] = suppressTransition(nextLayout);
                    }
                }
            } catch (e) {
                /* Never let the gate break rendering — fall through to plotly. */
            }
            return original.apply(this, arguments);
        };
        P.__gpMotionPatched = true;
        return true;
    }

    // Dash loads plotly.js as a global <script> (window._dashPlotlyJSURL), so it
    // may not exist yet when this file runs. Poll briefly; the first figure
    // UPDATE (the only thing that can transition — the initial paint never
    // does) is always many seconds out, behind a user interaction.
    function watchForPlotly() {
        if (patchPlotlyReact(window.Plotly)) return;
        var tries = 0;
        var id = setInterval(function () {
            if (patchPlotlyReact(window.Plotly) || ++tries > 100) clearInterval(id);
        }, 100);
    }

    // ── Wiring ──────────────────────────────────────────────────────
    var rafPending = false;
    function onDomChange() {
        // Mark + hide new sections SYNCHRONOUSLY inside the mutation microtask,
        // before the browser paints them — otherwise they'd flash in visible
        // and then get hidden (a flicker). Layout-dependent work (ink position,
        // count-up) is deferred to the next frame where measurements are valid.
        scanReveals();
        if (rafPending) return;
        rafPending = true;
        requestAnimationFrame(function () {
            rafPending = false;
            positionInk();
            runCountUp();
        });
    }

    function start() {
        enableMotionCss();
        // Install the chart morph gate before any figure can update. Must run
        // even under reduced motion — suppressing the transition IS its job.
        watchForPlotly();
        onDomChange();
        var mo = new MutationObserver(onDomChange);
        mo.observe(document.body, { childList: true, subtree: true });
        window.addEventListener("resize", positionInk, { passive: true });
        // Slide the ink-bar on tab change. The MutationObserver only watches
        // childList, so the active-class flip (an attribute change, applied
        // asynchronously after Dash's callback round-trip) wouldn't otherwise
        // re-measure. Delegated click + a few retries catch it once settled.
        document.addEventListener("click", function (e) {
            if (e.target.closest && e.target.closest(".nav-tabs .nav-link")) {
                setTimeout(positionInk, 60);
                setTimeout(positionInk, 280);
                setTimeout(positionInk, 700);
            }
        }, true);
        // Ink-bar retries — the active pill may not be laid out on the first
        // pass; these best-effort re-measures engage the slide once it is.
        setTimeout(positionInk, 250);
        setTimeout(positionInk, 900);
        // Safety net: guarantee nothing rendered by initial load stays hidden.
        setTimeout(revealAllFallback, 1500);
        // If the user flips their motion preference mid-session, reveal any
        // elements we had hidden so nothing is left stranded invisible.
        if (mql && mql.addEventListener) {
            mql.addEventListener("change", function () {
                if (reduced()) {
                    document.documentElement.classList.remove("gp-motion");
                    document.querySelectorAll(".gp-reveal").forEach(function (n) {
                        n.classList.add("in-view");
                    });
                } else {
                    enableMotionCss();
                }
            });
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", start);
    } else {
        start();
    }
})();
