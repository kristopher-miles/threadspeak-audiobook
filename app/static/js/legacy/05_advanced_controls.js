        // --- Advanced Controls toggle ---
        window.toggleAdvancedControls = function() {
            const panel = document.getElementById('advanced-controls-panel');
            const btn = document.getElementById('btn-advanced-controls');
            const visible = panel.style.display !== 'none';
            panel.style.display = visible ? 'none' : 'flex';
            btn.classList.toggle('active', !visible);
        };

