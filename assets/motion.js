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
