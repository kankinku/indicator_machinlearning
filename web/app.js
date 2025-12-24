const app = {
    data: [],
    features: [],
    bestShortId: null,

    init: function () {
        const pageId = document.body.id;

        // Common: Window Resize for Charts
        window.addEventListener('resize', () => {
            const gd = document.getElementById('detail-chart');
            if (gd && gd.data) Plotly.Plots.resize(gd);

            const fp = document.getElementById('feature-preview-chart');
            if (fp && fp.data) Plotly.Plots.resize(fp);
        });

        // Page-Specific Init
        if (pageId === 'page-dashboard') {
            this.initDashboard();
        } else if (pageId === 'page-history') {
            this.initHistory();
        } else if (pageId === 'page-features') {
            this.initFeatures();
        } else if (pageId === 'page-detail') {
            this.initDetail();
        } else if (pageId === 'page-analysis') {
            this.initAnalysis();
        } else if (pageId === 'page-backtest') {
            this.initBacktest();
        }
    },
    formatRisk: function (d) {
        const target = (d.target_return_pct !== null && d.target_return_pct !== undefined)
            ? `${d.target_return_pct.toFixed(2)}%`
            : '-';
        const stop = (d.stop_loss_pct !== null && d.stop_loss_pct !== undefined)
            ? `${d.stop_loss_pct.toFixed(2)}%`
            : '-';
        const horizon = (d.horizon !== null && d.horizon !== undefined) ? d.horizon : '-';
        const profile = d.risk_profile || '-';
        return `Target ${target} | Stop ${stop} | Horizon ${horizon} | Profile ${profile}`;
    },
    fillKeyValues: function (container, items) {
        if (!container) return;
        container.innerHTML = '';
        items.forEach(item => {
            const row = document.createElement('div');
            row.className = 'detail-kv-label';
            row.textContent = item.label;
            const val = document.createElement('div');
            val.className = 'detail-kv-value';
            val.textContent = item.value;
            container.appendChild(row);
            container.appendChild(val);
        });
    },
    formatJson: function (obj) {
        if (!obj) return '-';
        try {
            return JSON.stringify(obj, null, 2);
        } catch (e) {
            return String(obj);
        }
    },
    formatNumber: function (value, decimals) {
        if (value === null || value === undefined) return '-';
        const num = Number(value);
        if (Number.isNaN(num)) return '-';
        return num.toFixed(decimals);
    },

    saveState: function (key, val) {
        localStorage.setItem(key, String(val));
    },

    restoreState: function (key) {
        return localStorage.getItem(key) === 'true';
    },

    calculateScore: function (d) {
        // Balanced Approach:
        // Return * 3.0 (Significant) + WinRate * 50 (Baseline) + Trade * 0.1 (Participation, max 5.0)
        return (d.total_return || 0) * 3.0 + ((d.win_rate || 0) * 50.0) + ((d.trades || 0) * 0.1);
    },

    // === Dashboard ===
    initDashboard: async function () {
        await this.fetchData();
        await this.initDiagnostics(); // [V14]

        const cb = document.getElementById('dash-include-rejected');
        if (cb) cb.checked = this.restoreState('dash-include-rejected');

        this.renderKPIs();
        this.renderHallOfFame();
        this.renderDashTable();
    },

    initDiagnostics: async function () {
        const diagData = await this.fetchDataDiagnostics();
        if (diagData && diagData.summary) {
            this.renderDiagnostics(diagData);
        }
    },

    fetchDataDiagnostics: async function () {
        try {
            const res = await fetch('/api/v1/diagnostics');
            if (!res.ok) throw new Error("Failed to fetch diagnostics");
            return await res.json();
        } catch (err) {
            console.error(err);
            return null;
        }
    },

    renderDiagnostics: function (data) {
        const section = document.getElementById('diag-section');
        if (!section) return;

        const summary = data.summary;
        if (!summary || summary.status === "UNKNOWN") {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';

        const statusEl = document.getElementById('diag-status');
        statusEl.textContent = summary.status;

        // Clean class list and add new status
        statusEl.className = 'status-badge';
        const cleanStatus = summary.status.toLowerCase().replace(/[^a-z0-9_]/g, '');
        statusEl.classList.add(`status-${cleanStatus}`);

        document.getElementById('diag-pass-rate').textContent = (summary.pass_rate * 100).toFixed(1) + '%';
        document.getElementById('diag-rej-rate').textContent = (summary.rej_rate * 100).toFixed(1) + '%';

        const simRate = (summary.similarity && summary.similarity.avg_jaccard !== undefined)
            ? (summary.similarity.avg_jaccard * 100).toFixed(1) + '%'
            : '0%';
        document.getElementById('diag-sim-rate').textContent = simRate;

        // Render Taxonomy Bars
        const container = document.getElementById('diag-taxonomy-bars');
        container.innerHTML = '';

        if (summary.taxonomy) {
            Object.entries(summary.taxonomy).forEach(([issue, count]) => {
                const pct = (count / (summary.rej_rate * 100 || 1)) * 100; // rough estimate or just raw count
                // Actually taxonomy in summary is counts. Let's show as bars relative to total rejections.
                const totalRej = Object.values(summary.taxonomy).reduce((a, b) => a + b, 0);
                const barPct = totalRej > 0 ? (count / totalRej) * 100 : 0;

                const item = document.createElement('div');
                item.className = 'tax-item';
                item.innerHTML = `
                    <span class="tax-label">${issue.replace('_', ' ')} (${count})</span>
                    <div class="tax-bar-bg">
                        <div class="tax-bar-fill" style="width: ${barPct}%"></div>
                    </div>
                `;
                container.appendChild(item);
            });
        }
    },

    renderKPIs: function () {
        if (this.data.length === 0) return;
        const total = this.data.length;
        const approved = this.data.filter(d => d.status === "Approved");
        const appRate = total > 0 ? ((approved.length / total) * 100).toFixed(1) : 0;

        let maxSharpe = 0, maxWin = 0, maxRet = 0;
        if (approved.length > 0) {
            maxSharpe = Math.max(...approved.map(d => d.sharpe));
            maxWin = Math.max(...approved.map(d => d.win_rate));
            maxRet = Math.max(...approved.map(d => d.total_return));
        }

        const kpiGrid = document.getElementById('kpi-grid');
        if (kpiGrid) {
            kpiGrid.innerHTML = `
                <div class="kpi-card"><div class="kpi-label">Total Exp</div><div class="kpi-value">${total}</div></div>
                <div class="kpi-card"><div class="kpi-label">Approval Rate</div><div class="kpi-value">${appRate}%</div></div>
                <div class="kpi-card"><div class="kpi-label">Top Sharpe</div><div class="kpi-value" style="color:var(--accent-blue)">${maxSharpe.toFixed(2)}</div></div>
                <div class="kpi-card"><div class="kpi-label">Top Win Rate</div><div class="kpi-value">${(maxWin * 100).toFixed(1)}%</div></div>
                <div class="kpi-card"><div class="kpi-label">Top Return</div><div class="kpi-value" style="color:var(--accent-green)">${maxRet.toFixed(1)}%</div></div>
            `;
        }
    },

    renderHallOfFame: function () {
        const approved = this.data.filter(d => d.status === "Approved");
        const section = document.getElementById('hof-section');
        if (!section) return;

        if (approved.length === 0) {
            section.style.display = 'none';
            return;
        }
        section.style.display = 'block';

        // Find Best by Holistic Score
        const best = approved.reduce((prev, current) => {
            return (this.calculateScore(prev) > this.calculateScore(current)) ? prev : current;
        });

        this.bestShortId = best.short_id;

        document.getElementById('hof-time').textContent = new Date(best.timestamp).toLocaleString();
        document.getElementById('hof-title').textContent = `${best.origin} Strategy`;
        document.getElementById('hof-desc').textContent = best.indicators;

        document.getElementById('hof-sharpe').textContent = best.sharpe.toFixed(2);
        document.getElementById('hof-win').textContent = (best.win_rate * 100).toFixed(1) + '%';
        document.getElementById('hof-trades').textContent = best.trades;
        document.getElementById('hof-ret').textContent = best.total_return.toFixed(2) + '%';
        const hofRisk = document.getElementById('hof-risk');
        if (hofRisk) {
            hofRisk.textContent = this.formatRisk(best);
        }

        // Click Handler
        const hofCard = document.getElementById('hof-card');
        if (hofCard) {
            hofCard.style.cursor = 'pointer';
            hofCard.onclick = () => {
                window.location.href = `detail.html?id=${best.short_id}`;
            };
        }
    },

    renderDashTable: function () {
        const tbody = document.getElementById('dash-table-body');
        if (!tbody) return;
        tbody.innerHTML = "";

        if (!this.data || this.data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" style="text-align:center; padding:20px; color:#86868B;">No data available or loading...</td></tr>';
            return;
        }

        const includeRejected = document.getElementById('dash-include-rejected')?.checked;
        if (includeRejected !== undefined) this.saveState('dash-include-rejected', includeRejected);

        let sorted = this.data.filter(d => includeRejected || d.status === "Approved");

        // Default Sort: Holistic Score
        sorted.sort((a, b) => this.calculateScore(b) - this.calculateScore(a));
        const top5 = sorted.slice(0, 5);
        this.fillTable(tbody, top5);
    },

    // === History ===
    initHistory: async function () {
        await this.fetchData();

        const cb = document.getElementById('hist-include-rejected');
        if (cb) cb.checked = this.restoreState('hist-include-rejected');

        this.renderHistoryTable();

        const searchInput = document.getElementById('search-input');
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.renderHistoryTable(e.target.value);
            });
        }
    },

    renderHistoryTable: function (filterText = null) {
        const tbody = document.getElementById('hist-table-body');
        if (!tbody) return;

        if (filterText === null || filterText === undefined) {
            const searchInput = document.getElementById('search-input');
            filterText = searchInput ? searchInput.value : "";
        }

        tbody.innerHTML = "";

        const includeRejected = document.getElementById('hist-include-rejected')?.checked;
        if (includeRejected !== undefined) this.saveState('hist-include-rejected', includeRejected);

        let displayData = this.data.filter(d => includeRejected || d.status === "Approved");

        if (filterText) {
            const lower = filterText.toLowerCase();
            displayData = displayData.filter(d =>
                d.origin.toLowerCase().includes(lower) ||
                d.indicators.toLowerCase().includes(lower) ||
                d.short_id.includes(lower)
            );
        }

        const sortSelect = document.getElementById('sort-select');
        const sortMode = sortSelect ? sortSelect.value : 'timestamp';

        displayData.sort((a, b) => {
            if (sortMode === 'timestamp') return new Date(b.timestamp) - new Date(a.timestamp);
            if (sortMode === 'holistic') {
                return this.calculateScore(b) - this.calculateScore(a);
            }
            return (b[sortMode] || 0) - (a[sortMode] || 0);
        });

        this.fillTable(tbody, displayData);
    },

    handleSortChange: function (val) {
        const searchInput = document.getElementById('search-input');
        this.renderHistoryTable(searchInput ? searchInput.value : "");
    },

    // === Features ===
    initFeatures: async function () {
        await this.fetchFeatures();
        this.renderFeatureStats();
        this.renderFeatureTable();
    },

    renderFeatureStats: function () {
        const statsContainer = document.getElementById('feature-stats');
        if (!statsContainer || this.features.length === 0) return;

        const total = this.features.length;
        const categories = [...new Set(this.features.map(f => f.category).filter(Boolean))];
        const categoryCounts = this.features.reduce((acc, f) => {
            if (f.category) {
                acc[f.category] = (acc[f.category] || 0) + 1;
            }
            return acc;
        }, {});

        // Sort categories by count
        const topCats = Object.entries(categoryCounts)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 4);

        let html = `
            <div class="kpi-card">
                <div class="kpi-label">Total Features</div>
                <div class="kpi-value">${total}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-label">Categories</div>
                <div class="kpi-value">${categories.length}</div>
            </div>
        `;

        topCats.forEach(([cat, count]) => {
            html += `
                <div class="kpi-card">
                    <div class="kpi-label">${cat}</div>
                    <div class="kpi-value" style="font-size: 18px;">${count} Indicators</div>
                </div>
            `;
        });

        statsContainer.innerHTML = html;
    },

    renderFeatureTable: function () {
        const tbody = document.getElementById('feat-table-body');
        if (!tbody) return;

        tbody.innerHTML = "";
        const sorted = [...this.features].sort((a, b) => a.feature_id.localeCompare(b.feature_id));

        sorted.forEach(f => {
            const tr = document.createElement('tr');

            let paramsHtml = "";
            if (f.params && f.params.length > 0) {
                paramsHtml = f.params.map(p =>
                    `<span class="feature-pill">${p.name}: [${p.min}, ${p.max}]</span>`
                ).join("");
            } else {
                paramsHtml = `<span class="param-pill">No params</span>`;
            }

            tr.innerHTML = `
                <td style="font-weight:600; font-family:'JetBrains Mono', monospace;">${f.feature_id}</td>
                <td style="color:#86868B">${f.category || '-'}</td>
                <td>${f.description || '-'}</td>
                <td>${paramsHtml}</td>
                <td style="text-align:center;"><button class="btn-micro" onclick="app.previewFeature('${f.feature_id}')">Visualize</button></td>
            `;
            tbody.appendChild(tr);
        });
    },

    previewFeature: async function (featureId) {
        const container = document.getElementById('feature-preview-container');
        container.style.display = 'block';
        container.scrollIntoView({ behavior: 'smooth' });

        document.getElementById('preview-title').textContent = `Preview: ${featureId}`;
        const chartDiv = document.getElementById('feature-preview-chart');
        chartDiv.innerHTML = '<div style="color:#86868B; text-align:center; padding-top:150px;">Loading Market Data...</div>';

        try {
            const res = await fetch(`/api/features/${featureId}/preview`, { method: 'POST' });

            if (!res.ok) {
                const errData = await res.json();
                throw new Error(errData.detail || "Preview failed");
            }
            const data = await res.json();
            this.renderPreviewChart(data, featureId);

        } catch (e) {
            chartDiv.innerHTML = `<div style="color:var(--accent-red); text-align:center; padding-top:150px;">Error: ${e.message}</div>`;
        }
    },

    renderPreviewChart: function (data, featureId) {
        const traces = [];
        const dates = data.dates;

        // Close Price Trace (Secondary Y-Axis, Dashed)
        if (data.close && data.close.length > 0) {
            traces.push({
                x: dates,
                y: data.close,
                name: 'Close Price',
                yaxis: 'y2',
                type: 'scatter',
                line: { color: '#555', width: 1, dash: 'dot' }
            });
        }

        // Indicator Traces (Primary Y-Axis)
        const colors = ['#2997FF', '#30D158', '#FF9F0A', '#BF5AF2', '#FF453A'];
        let colorIdx = 0;
        Object.keys(data.values).forEach(key => {
            traces.push({
                x: dates,
                y: data.values[key],
                name: key,
                type: 'scatter',
                line: { width: 2, color: colors[colorIdx % colors.length] }
            });
            colorIdx++;
        });

        const layout = {
            paper_bgcolor: '#1C1C1E',
            plot_bgcolor: '#1C1C1E',
            height: 420,
            showlegend: true,
            legend: { orientation: 'h', y: 1.12, x: 0.5, xanchor: 'center', font: { color: '#E5E5E5', size: 11 } },
            xaxis: {
                showgrid: false,
                color: '#86868B',
                tickfont: { family: 'JetBrains Mono, monospace', size: 10 }
            },
            yaxis: {
                title: featureId,
                titlefont: { color: '#86868B', size: 11 },
                gridcolor: '#2C2C2E',
                color: '#86868B'
            },
            yaxis2: {
                title: 'Price',
                titlefont: { color: '#555', size: 10 },
                overlaying: 'y',
                side: 'right',
                showgrid: false,
                color: '#555'
            },
            margin: { l: 50, r: 50, t: 50, b: 40 }
        };

        const config = { displayModeBar: false, responsive: true };
        Plotly.newPlot('feature-preview-chart', traces, layout, config);
    },

    closePreview: function () {
        document.getElementById('feature-preview-container').style.display = 'none';
    },

    // === Detail ===
    initDetail: async function () {
        const urlParams = new URLSearchParams(window.location.search);
        const id = urlParams.get('id');

        if (!id) {
            document.querySelector('.detail-view').innerHTML = '<p style="color:#86868B; padding:50px;">No strategy ID provided.</p>';
            return;
        }

        await this.fetchData();
        const item = this.data.find(d => d.short_id === id);

        if (!item) {
            document.querySelector('.detail-view').innerHTML = '<p style="color:#86868B; padding:50px;">Strategy not found.</p>';
            return;
        }

        this.renderDetailView(item);
    },

    renderDetailView: function (d) {
        document.title = `${d.origin} - Vibe Lab`;
        document.getElementById('detail-title').textContent = `${d.origin} Strategy`;
        document.getElementById('detail-id').textContent = d.short_id;
        document.getElementById('detail-date').textContent = new Date(d.timestamp).toLocaleString();

        const badge = document.getElementById('detail-badge');
        if (d.status === "Approved") {
            badge.className = "status-badge status-approved";
            badge.textContent = "APPROVED";
        } else {
            badge.className = "status-badge status-rejected";
            badge.textContent = "REJECTED";
        }

        document.getElementById('d-sharpe').textContent = d.sharpe.toFixed(2);
        document.getElementById('d-win').textContent = (d.win_rate * 100).toFixed(1) + "%";
        document.getElementById('d-trades').textContent = d.trades;
        document.getElementById('d-avg').textContent = d.return_mean.toFixed(2) + '%';
        document.getElementById('d-tot').textContent = d.total_return.toFixed(2) + " %";
        const detailRisk = document.getElementById('detail-risk');
        if (detailRisk) {
            detailRisk.textContent = this.formatRisk(d);
        }

        // Composition
        const grid = document.getElementById('detail-comp-grid');
        grid.innerHTML = "";
        if (d.genome_full && typeof d.genome_full === 'object') {
            Object.keys(d.genome_full).sort().forEach(key => {
                const params = d.genome_full[key];
                let paramsStr = "";
                if (typeof params === 'object') {
                    paramsStr = Object.entries(params).map(([k, v]) => `${k}: ${v}`).join(', ');
                } else {
                    paramsStr = String(params);
                }
                const item = document.createElement('div');
                item.className = 'comp-item';
                item.innerHTML = `
                    <div class="comp-name">${key}</div>
                    <div class="comp-params">${paramsStr}</div>
                `;
                grid.appendChild(item);
            });
        }

        const indicatorsPre = document.getElementById('detail-indicators');
        if (indicatorsPre) {
            indicatorsPre.textContent = this.formatJson(d.genome_full || {});
        }

        const overview = document.getElementById('detail-overview');
        this.fillKeyValues(overview, [
            { label: 'Template', value: d.template_id || d.origin || '-' },
            { label: 'Module Key', value: d.module_key || '-' },
            { label: 'Regime', value: (d.rl_meta && d.rl_meta.state_key) ? d.rl_meta.state_key : '-' },
            { label: 'Data Window', value: (d.data_window && d.data_window.lookback) ? String(d.data_window.lookback) : '-' },
            { label: 'Cost (bps)', value: (d.execution_assumption && d.execution_assumption.cost_bps !== undefined) ? String(d.execution_assumption.cost_bps) : '-' },
            { label: 'Eval Stage', value: d.eval_stage || '-' },
        ]);

        const entry = document.getElementById('detail-entry');
        const entryThreshold = (d.entry_threshold !== null && d.entry_threshold !== undefined) ? d.entry_threshold : '-';
        const entryMaxProb = (d.entry_max_prob !== null && d.entry_max_prob !== undefined) ? d.entry_max_prob : '-';
        const entryRule = (entryThreshold !== '-' && entryMaxProb !== '-')
            ? `prob >= ${entryThreshold} and prob <= ${entryMaxProb}`
            : '-';
        this.fillKeyValues(entry, [
            { label: 'Threshold', value: String(entryThreshold) },
            { label: 'Max Prob', value: String(entryMaxProb) },
            { label: 'Rule', value: entryRule },
        ]);

        const riskBlock = document.getElementById('detail-risk-block');
        this.fillKeyValues(riskBlock, [
            { label: 'Target (%)', value: this.formatNumber(d.target_return_pct, 2) !== '-' ? `${this.formatNumber(d.target_return_pct, 2)}%` : '-' },
            { label: 'Stop (%)', value: this.formatNumber(d.stop_loss_pct, 2) !== '-' ? `${this.formatNumber(d.stop_loss_pct, 2)}%` : '-' },
            { label: 'TP (%)', value: this.formatNumber(d.tp_pct * 100, 2) !== '-' ? `${this.formatNumber(d.tp_pct * 100, 2)}%` : '-' },
            { label: 'SL (%)', value: this.formatNumber(d.sl_pct * 100, 2) !== '-' ? `${this.formatNumber(d.sl_pct * 100, 2)}%` : '-' },
            { label: 'Horizon', value: (d.horizon !== null && d.horizon !== undefined) ? String(d.horizon) : '-' },
            { label: 'Risk Profile', value: d.risk_profile || '-' },
            { label: 'R/R', value: this.formatNumber(d.risk_reward_ratio, 2) },
            { label: 'k_up', value: (d.k_up !== null && d.k_up !== undefined) ? String(d.k_up) : '-' },
            { label: 'k_down', value: (d.k_down !== null && d.k_down !== undefined) ? String(d.k_down) : '-' },
        ]);

        const evalBox = document.getElementById('detail-eval');
        this.fillKeyValues(evalBox, [
            { label: 'Eval Score', value: this.formatNumber(d.eval_score, 3) },
            { label: 'Sample ID', value: d.sample_id || '-' },
            { label: 'Sample Window', value: d.sample_window_id || '-' },
            { label: 'Sample Return (%)', value: this.formatNumber(d.sample_total_return_pct, 2) },
            { label: 'Sample MDD (%)', value: this.formatNumber(d.sample_mdd_pct, 2) },
            { label: 'Sample R/R', value: this.formatNumber(d.sample_rr, 2) },
            { label: 'Sample Vol (%)', value: this.formatNumber(d.sample_vol_pct, 2) },
            { label: 'Sample Trades', value: (d.sample_trades !== null && d.sample_trades !== undefined) ? String(d.sample_trades) : '-' },
            { label: 'Sample Win Rate', value: this.formatNumber(d.sample_win_rate * 100, 1) !== '-' ? `${this.formatNumber(d.sample_win_rate * 100, 1)}%` : '-' },
            { label: 'CPCV Mean', value: this.formatNumber(d.cpcv_mean, 3) },
            { label: 'CPCV Worst', value: this.formatNumber(d.cpcv_worst, 3) },
        ]);

        // Chart
        this.loadDetailChart(d.id);
    },

    loadDetailChart: async function (expId) {
        try {
            const res = await fetch(`/api/experiments/${expId}/chart`);
            if (!res.ok) throw new Error("Chart not found");
            const data = await res.json();

            const trace = {
                x: data.dates,
                y: data.equity,
                mode: 'lines',
                fill: 'tozeroy',
                line: { color: '#2997FF', width: 2 },
                fillcolor: 'rgba(41, 151, 255, 0.1)'
            };
            const layout = {
                paper_bgcolor: '#1C1C1E',
                plot_bgcolor: '#1C1C1E',
                margin: { l: 50, r: 20, t: 20, b: 40 },
                xaxis: { showgrid: false, color: '#86868B', tickfont: { family: 'JetBrains Mono', size: 10 } },
                yaxis: { gridcolor: '#2C2C2E', color: '#86868B', title: 'Equity (%)', titlefont: { size: 11 } }
            };
            const config = { displayModeBar: false, responsive: true };
            Plotly.newPlot('detail-chart', [trace], layout, config);
        } catch (e) {
            document.getElementById('detail-chart').innerHTML = '<p style="color:#86868B; text-align:center; padding-top:100px;">Chart data unavailable.</p>';
        }
    },

    // === Shared Data Fetchers ===
    fetchData: async function () {
        try {
            const res = await fetch('/api/experiments');
            if (!res.ok) throw new Error("Failed to fetch experiments");
            this.data = await res.json();
            return this.data;
        } catch (err) {
            console.error(err);
            return [];
        }
    },

    fetchFeatures: async function () {
        try {
            const res = await fetch('/api/features');
            if (!res.ok) throw new Error("Failed to fetch features");
            this.features = await res.json();
            return this.features;
        } catch (err) {
            console.error(err);
            return [];
        }
    },

    // === Shared Table Filler ===
    fillTable: function (tbody, list) {
        list.forEach(d => {
            const tr = document.createElement('tr');
            tr.onclick = () => {
                window.location.href = `detail.html?id=${d.short_id}`;
            };

            const date = new Date(d.timestamp);
            const timeStr = `${date.getMonth() + 1}/${date.getDate()} ${String(date.getHours()).padStart(2, '0')}:${String(date.getMinutes()).padStart(2, '0')}`;

            // [V11] REJECTED 상태에 실패 사유 표시
            let statusBadge;
            if (d.status === "Approved") {
                statusBadge = `<span class="status-badge status-approved">APPROVED</span>`;
            } else {
                const failReason = d.fail_reason || 'Unknown';
                statusBadge = `<span class="status-badge status-rejected" title="${failReason}">REJECTED</span>`;
            }

            tr.innerHTML = `
                <td style="color:#86868B; font-family:'JetBrains Mono', monospace;">${timeStr}</td>
                <td class="mono" style="color:#86868B">#${d.generation}</td>
                <td>
                    <div style="font-weight:600; font-size:13px;">${d.origin}</div>
                    <div style="font-size:11px; color:#86868B; margin-top:2px; max-width:250px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; font-family:'JetBrains Mono', monospace;">${d.indicators}</div>
                </td>
                <td class="mono" style="font-weight:600; color:${d.sharpe >= 1.0 ? 'var(--accent-blue)' : 'inherit'}">${d.sharpe.toFixed(2)}</td>
                <td class="mono">${(d.win_rate * 100).toFixed(1)}%</td>
                <td class="mono">${d.trades}</td>
                <td class="mono" style="color:${d.total_return > 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${d.total_return.toFixed(2)} %</td>
                <td>${statusBadge}</td>
            `;
            tbody.appendChild(tr);
        });
    },

    // === Model Test (Backtest Page) ===
    selectedModel: null,

    initBacktest: async function () {
        await this.fetchData();
        const cb = document.getElementById('bt-include-rejected');
        if (cb) cb.checked = this.restoreState('bt-include-rejected');

        this.renderModelCards();
    },

    renderModelCards: function () {
        const container = document.getElementById('model-cards');
        if (!container) return;

        container.innerHTML = "";

        const includeRejected = document.getElementById('bt-include-rejected')?.checked;
        if (includeRejected !== undefined) this.saveState('bt-include-rejected', includeRejected);

        // Get top 5 models sorted by Sharpe
        const candidates = this.data.filter(d => includeRejected || d.status === "Approved");
        candidates.sort((a, b) => b.sharpe - a.sharpe);
        const top5 = candidates.slice(0, 5);

        if (top5.length === 0) {
            container.innerHTML = '<div style="color:#86868B; padding:20px;">No models available.</div>';
            return;
        }

        top5.forEach((model, idx) => {
            const card = document.createElement('div');
            card.className = 'model-card';
            card.dataset.id = model.short_id;

            card.innerHTML = `
                <div class="model-rank">#${idx + 1}</div>
                <div class="model-info">
                    <div class="model-name">${model.origin} Strategy</div>
                    <div class="model-indicators">${model.indicators}</div>
                </div>
                <div class="model-metrics">
                    <div class="model-metric">
                        <span class="metric-val" style="color:var(--accent-blue)">${model.sharpe.toFixed(2)}</span>
                        <span class="metric-lbl">Sharpe</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-val">${(model.win_rate * 100).toFixed(0)}%</span>
                        <span class="metric-lbl">Win</span>
                    </div>
                    <div class="model-metric">
                        <span class="metric-val" style="color:${model.total_return > 0 ? 'var(--accent-green)' : 'var(--accent-red)'}">${model.total_return.toFixed(1)}%</span>
                        <span class="metric-lbl">Return (%)</span>
                    </div>
                </div>
            `;

            card.onclick = () => this.selectModel(model);
            container.appendChild(card);
        });
    },

    selectModel: function (model) {
        this.selectedModel = model;

        // Update card selection
        document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
        const selectedCard = document.querySelector(`.model-card[data-id="${model.short_id}"]`);
        if (selectedCard) selectedCard.classList.add('selected');

        // Update labels
        const label = document.getElementById('selected-model-label');
        if (label) label.textContent = `Selected: ${model.origin}`;

        const title = document.getElementById('selected-model-title');
        if (title) title.textContent = `${model.origin} Strategy`;

        // Clear previous results
        ['m-total-return', 'm-entry-signals', 'm-trades', 'm-winrate', 'm-mdd'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '-';
        });

        // Show section
        const section = document.getElementById('model-detail-section');
        if (section) {
            section.style.display = 'block';
            section.scrollIntoView({ behavior: 'smooth' });
        }

        // Composition Grid
        const grid = document.getElementById('model-comp-grid');
        if (grid) {
            grid.innerHTML = "";
            if (model.genome_full && typeof model.genome_full === 'object') {
                Object.keys(model.genome_full).sort().forEach(key => {
                    const params = model.genome_full[key];
                    let paramsStr = "";
                    if (typeof params === 'object') {
                        paramsStr = Object.entries(params).map(([k, v]) => `${k}: ${v}`).join(', ');
                    } else {
                        paramsStr = String(params);
                    }
                    const item = document.createElement('div');
                    item.className = 'comp-item';
                    item.innerHTML = `<div class="comp-name">${key}</div><div class="comp-params">${paramsStr}</div>`;
                    grid.appendChild(item);
                });
            }
        }

        // Load Chart
        this.loadModelChart(model.id);

        // Hide summary until backtest runs
        const summary = document.getElementById('backtest-summary');
        if (summary) summary.style.display = 'none';
    },

    loadModelChart: async function (expId) {
        try {
            const res = await fetch(`/api/experiments/${expId}/chart`);
            if (!res.ok) throw new Error("Chart not found");
            const data = await res.json();

            const trace = {
                x: data.dates,
                y: data.equity,
                mode: 'lines',
                fill: 'tozeroy',
                line: { color: '#2997FF', width: 2 },
                fillcolor: 'rgba(41, 151, 255, 0.1)'
            };
            const layout = {
                paper_bgcolor: '#1C1C1E',
                plot_bgcolor: '#1C1C1E',
                height: 350,
                margin: { l: 50, r: 20, t: 20, b: 40 },
                xaxis: { showgrid: false, color: '#86868B', tickfont: { family: 'JetBrains Mono', size: 10 } },
                yaxis: { gridcolor: '#2C2C2E', color: '#86868B', title: 'Equity (%)', titlefont: { size: 11 } }
            };
            const config = { displayModeBar: false, responsive: true };
            if (document.getElementById('model-chart')) Plotly.newPlot('model-chart', [trace], layout, config);
        } catch (e) {
            if (document.getElementById('model-chart')) document.getElementById('model-chart').innerHTML = '<p style="color:#86868B; text-align:center; padding-top:100px;">Chart data unavailable.</p>';
        }
    },

    runModelBacktest: async function () {
        if (!this.selectedModel) {
            alert('Please select a model first.');
            return;
        }

        const btn = document.getElementById('run-test-btn');
        btn.disabled = true;
        btn.textContent = 'Running...';

        try {
            const res = await fetch(`/api/experiments/${this.selectedModel.id}/backtest`);

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || 'Test failed');
            }

            const data = await res.json();
            this.renderBacktestResult(data);

        } catch (e) {
            const summary = document.getElementById('backtest-summary');
            if (summary) {
                summary.style.display = 'block';
                document.getElementById('backtest-summary-text').textContent = `Error: ${e.message}`;
            }
        } finally {
            btn.disabled = false;
            btn.textContent = 'Run Backtest';
        }
    },

    renderBacktestResult: function (data) {
        if (!data || !data.metrics) return;
        const metrics = data.metrics;

        document.getElementById('m-total-return').textContent = metrics.total_return_pct.toFixed(2) + '%';
        document.getElementById('m-entry-signals').textContent = metrics.entry_signals;
        document.getElementById('m-trades').textContent = metrics.trade_count;
        document.getElementById('m-winrate').textContent = (metrics.win_rate * 100).toFixed(1) + '%';
        document.getElementById('m-mdd').textContent = metrics.mdd_pct.toFixed(2) + '%';

        const summary = document.getElementById('backtest-summary');
        const text = document.getElementById('backtest-summary-text');
        if (summary && text) {
            summary.style.display = 'block';
            text.textContent =
                `From ${metrics.start_date} to ${metrics.end_date}, total return is ` +
                `${metrics.total_return_pct.toFixed(2)}% with ${metrics.trade_count} trades ` +
                `(${metrics.entry_signals} entry signals). Win rate ${(
                    metrics.win_rate * 100
                ).toFixed(1)}% and max drawdown ${metrics.mdd_pct.toFixed(2)}%.`;
        }
    },

    // === Analysis ===
    initAnalysis: async function () {
        try {
            const res = await fetch('/api/v1/stats/regime');
            if (!res.ok) throw new Error("Failed to fetch stats");
            const data = await res.json();

            this.renderRejectionChart(data.rejections);
            this.renderPriorsChart(data.priors);
            this.renderRegimeLeaderboard(data.leaderboard);
        } catch (e) {
            console.error("Analysis Init Error:", e);
        }
    },

    renderRejectionChart: function (rejections) {
        if (!rejections || Object.keys(rejections).length === 0) return;

        const labels = Object.keys(rejections).slice(0, 10);
        const values = labels.map(l => rejections[l]);

        const data = [{
            values: values,
            labels: labels,
            type: 'pie',
            hole: 0.4,
            marker: {
                colors: ['#FF453A', '#FF9F0A', '#FFD60A', '#30D158', '#64D2FF', '#BF5AF2', '#86868B', '#5E5E62', '#414144', '#2C2C2E']
            },
            textinfo: 'percent',
            textposition: 'outside',
            automargin: true
        }];

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            showlegend: true,
            legend: {
                font: { color: '#86868B', size: 10 },
                orientation: 'v',
                x: 1.1,
                y: 0.5
            },
            margin: { t: 20, b: 20, l: 20, r: 120 },
            font: { color: '#F5F5F7', family: 'Inter' }
        };

        const config = { displayModeBar: false, responsive: true };
        if (document.getElementById('rejection-chart')) Plotly.newPlot('rejection-chart', data, layout, config);
    },

    renderPriorsChart: function (priors) {
        if (!priors || Object.keys(priors).length === 0) return;

        const regimes = Object.keys(priors);
        const categories = Object.keys(priors[regimes[0]] || {});

        const traces = regimes.map(regime => {
            return {
                x: categories,
                y: categories.map(cat => (priors[regime][cat] || 0) * 100),
                name: regime,
                type: 'bar'
            };
        });

        const layout = {
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'rgba(0,0,0,0)',
            barmode: 'group',
            xaxis: {
                tickfont: { color: '#86868B', size: 10 },
                showgrid: false
            },
            yaxis: {
                title: 'Weight (%)',
                titlefont: { color: '#86868B', size: 10 },
                gridcolor: '#38383A',
                color: '#86868B'
            },
            legend: {
                font: { color: '#F5F5F7', size: 10 },
                orientation: 'h',
                y: -0.3
            },
            margin: { t: 20, b: 80, l: 40, r: 20 },
            font: { family: 'Inter' }
        };

        const config = { displayModeBar: false, responsive: true };
        if (document.getElementById('priors-chart')) Plotly.newPlot('priors-chart', traces, layout, config);
    },

    renderRegimeLeaderboard: function (leaderboard) {
        const container = document.getElementById('regime-leaderboard');
        if (!container) return;

        container.innerHTML = "";

        const regimes = Object.keys(leaderboard);
        if (regimes.length === 0) {
            container.innerHTML = '<div class="status-text">No approved strategies yet to rank.</div>';
            return;
        }

        regimes.forEach(regime => {
            const group = leaderboard[regime];
            const card = document.createElement('div');
            card.className = "regime-card";

            let leadersHtml = "";
            group.forEach((item, idx) => {
                leadersHtml += `
                    <div class="leader-item">
                        <div class="leader-rank">RANK #${idx + 1} (${item.id})</div>
                        <div class="leader-info">${item.indicators}</div>
                        <div class="leader-stats">
                            <span>Ret: ${item.total_return.toFixed(1)}%</span>
                            Win: ${(item.win_rate * 100).toFixed(1)}%
                            Trades: ${item.trades}
                        </div>
                    </div>
                `;
            });

            card.innerHTML = `
                <div class="regime-name" style="color:${this.getRegimeColor(regime)}">${regime}</div>
                ${leadersHtml}
            `;
            container.appendChild(card);
        });
    },

    getRegimeColor: function (regime) {
        const r = regime.toUpperCase();
        if (r.includes("BULL") || r.includes("UP")) return "#30D158";
        if (r.includes("BEAR") || r.includes("DOWN")) return "#FF453A";
        if (r.includes("STAGNANT") || r.includes("FLAT") || r.includes("SIDE")) return "#FF9F0A";
        if (r.includes("VOLATILE") || r.includes("SHOCK")) return "#BF5AF2";
        return "#2997FF";
    }
};

// Auto Initialize - add backtest page check
// Initialize
app.init();
