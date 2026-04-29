/* Keyboard shortcuts and accessibility enhancements */

// Alt+1..4: Switch between the four visible tabs.
// R3 reduced 9 → 4 (Overview / Forecast / Risk / Models). Hidden tabs
// stay rendered for callback safety but aren't surfaced via shortcuts.
document.addEventListener('keydown', function(e) {
    if (!e.altKey) return;

    const tabMap = {
        '1': 'tab-overview',
        '2': 'tab-outlook',
        '3': 'tab-alerts',
        '4': 'tab-models',
    };

    if (tabMap[e.key]) {
        e.preventDefault();
        // Click only visible tab pills — skip any hidden via .d-none.
        const visibleLinks = Array.from(
            document.querySelectorAll('.nav-tabs .nav-item:not(.d-none) .nav-link')
        );
        const tabIndex = parseInt(e.key) - 1;
        if (visibleLinks[tabIndex]) {
            visibleLinks[tabIndex].click();
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
    skip.href = '#tab-content';
    skip.textContent = 'Skip to main content';
    skip.className = 'sr-only sr-only-focusable';
    skip.style.cssText = 'position:absolute;top:-40px;left:0;background:#3b82f6;color:#0a0a0b;' +
        'padding:8px 16px;z-index:10000;transition:top 0.2s;';
    skip.addEventListener('focus', function() { skip.style.top = '0'; });
    skip.addEventListener('blur', function() { skip.style.top = '-40px'; });
    document.body.insertBefore(skip, document.body.firstChild);
})();
