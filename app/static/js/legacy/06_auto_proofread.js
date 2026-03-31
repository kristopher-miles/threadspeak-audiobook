        // --- Auto Proofread toggle ---
        const AUTO_PROOFREAD_KEY = 'alexandria-auto-proofread';
        const savedAutoProofread = localStorage.getItem(AUTO_PROOFREAD_KEY);
        let autoProofreadEnabled = savedAutoProofread === null ? false : savedAutoProofread === 'true';
        let clipsAtLastAutoProofread = 0;

        function _applyAutoProofreadBtn() {
            const btn = document.getElementById('btn-auto-proofread');
            if (!btn) return;
            btn.classList.toggle('active', autoProofreadEnabled);
            btn.title = autoProofreadEnabled ? 'Auto Proofread: ON — runs every 25 clips' : 'Auto Proofread: OFF';
        }

        window.toggleAutoProofread = function() {
            autoProofreadEnabled = !autoProofreadEnabled;
            localStorage.setItem(AUTO_PROOFREAD_KEY, autoProofreadEnabled);
            _applyAutoProofreadBtn();
        };

        _applyAutoProofreadBtn();

