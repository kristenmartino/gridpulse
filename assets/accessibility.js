/* Keyboard shortcuts and accessibility enhancements */

// Alt+1..5: Switch between the five visible tabs, in shipped order:
//   1 Overview · 2 US Grid · 3 Forecast · 4 Risk · 5 Models
// (P2-42/#273: the old 4-entry map predated the US Grid tab — its keys
// gated a positional click, so Alt+2..4 landed one tab LEFT of the map's
// own intent and Models was unreachable. The key is used positionally
// against the visible pills, so this stays correct as long as the count
// below matches components/layout.py's visible tab order.)
document.addEventListener('keydown', function(e) {
    if (!e.altKey) return;

    const VISIBLE_TAB_COUNT = 5;
    const n = parseInt(e.key, 10);
    if (n >= 1 && n <= VISIBLE_TAB_COUNT) {
        e.preventDefault();
        // Click only visible tab pills — skip any hidden via .d-none.
        const visibleLinks = Array.from(
            document.querySelectorAll('.nav-tabs .nav-item:not(.d-none) .nav-link')
        );
        if (visibleLinks[n - 1]) {
            visibleLinks[n - 1].click();
        }
    }

    // Alt+R: Focus region selector
    if (e.key === 'r' || e.key === 'R') {
        e.preventDefault();
        const regionSelect = document.getElementById('region-selector');
        if (regionSelect) regionSelect.focus();
    }

    // Alt+P: Focus persona selector
    if (e.key === 'p' || e.key === 'P') {
        e.preventDefault();
        const personaSelect = document.getElementById('persona-selector');
        if (personaSelect) personaSelect.focus();
    }
});

// Add ARIA labels to charts after they render
const observer = new MutationObserver(function(mutations) {
    // Label all Plotly chart containers
    document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {
        if (!plot.getAttribute('role')) {
            plot.setAttribute('role', 'img');
            const title = plot.closest('.chart-container')?.querySelector('.chart-title');
            if (title) {
                plot.setAttribute('aria-label', 'Chart: ' + title.textContent);
            }
        }
    });

    // Label KPI cards
    document.querySelectorAll('.kpi-card').forEach(function(card) {
        if (!card.getAttribute('role')) {
            card.setAttribute('role', 'status');
            const label = card.querySelector('.kpi-label');
            const value = card.querySelector('.kpi-value');
            if (label && value) {
                card.setAttribute('aria-label', label.textContent + ': ' + value.textContent);
            }
        }
    });

    // Label alert cards
    document.querySelectorAll('.alert-card').forEach(function(card) {
        if (!card.getAttribute('role')) {
            card.setAttribute('role', 'alert');
        }
    });
});

// Start observing once the DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function() {
        observer.observe(document.body, { childList: true, subtree: true });
    });
} else {
    observer.observe(document.body, { childList: true, subtree: true });
}

// Skip-to-content link (screen readers)
(function() {
    const skip = document.createElement('a');
    // R5b: <main id="main-content"> now wraps the tab strip so the
    // skip-link lands the user past the header on activation.
    skip.href = '#main-content';
    skip.textContent = 'Skip to main content';
    skip.className = 'sr-only sr-only-focusable';
    skip.style.cssText = 'position:absolute;top:-40px;left:0;background:#3b82f6;color:#0a0a0b;' +
        'padding:8px 16px;z-index:10000;transition:top 0.2s;';
    skip.addEventListener('focus', function() { skip.style.top = '0'; });
    skip.addEventListener('blur', function() { skip.style.top = '-40px'; });
    document.body.insertBefore(skip, document.body.firstChild);
})();
