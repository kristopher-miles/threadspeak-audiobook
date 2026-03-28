        // --- Toast & Confirm utilities ---
        function showToast(message, type = 'info', duration = 4000) {
            const container = document.getElementById('toast-container');
            const bgClass = type === 'success' ? 'bg-success' :
                           type === 'error' ? 'bg-danger' :
                           type === 'warning' ? 'bg-warning text-dark' : 'bg-info';
            const id = 'toast-' + Date.now();
            const html = `
                <div id="${id}" class="toast align-items-center text-white ${bgClass} border-0" role="alert">
                    <div class="d-flex">
                        <div class="toast-body">${message}</div>
                        <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                    </div>
                </div>`;
            container.insertAdjacentHTML('beforeend', html);
            const el = document.getElementById(id);
            const toast = new bootstrap.Toast(el, { delay: duration });
            toast.show();
            el.addEventListener('hidden.bs.toast', () => el.remove());
        }

        function showConfirm(message) {
            return new Promise((resolve) => {
                const body = document.getElementById('confirmModalBody');
                body.textContent = message;
                const modal = new bootstrap.Modal(document.getElementById('confirmModal'));
                const okBtn = document.getElementById('confirmModalOk');
                const cancelBtn = document.getElementById('confirmModalCancel');

                function cleanup() {
                    okBtn.removeEventListener('click', onOk);
                    cancelBtn.removeEventListener('click', onCancel);
                    document.getElementById('confirmModal').removeEventListener('hidden.bs.modal', onHidden);
                }
                let resolved = false;
                function onOk() { resolved = true; cleanup(); modal.hide(); resolve(true); }
                function onCancel() { resolved = true; cleanup(); modal.hide(); resolve(false); }
                function onHidden() { if (!resolved) { cleanup(); resolve(false); } }

                okBtn.addEventListener('click', onOk);
                cancelBtn.addEventListener('click', onCancel);
                document.getElementById('confirmModal').addEventListener('hidden.bs.modal', onHidden);
                modal.show();
            });
        }

