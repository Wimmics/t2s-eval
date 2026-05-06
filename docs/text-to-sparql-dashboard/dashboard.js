(function () {
	const appData = window.__T2S_DASHBOARD_DATA__ || { datasets: {} };

	const refs = {
		fileSelector: document.getElementById('file-selector'),
		fileInfo: document.getElementById('file-info'),
		systemSelector: document.getElementById('system-selector'),
		systemsSelectAll: document.getElementById('systems-select-all'),
		systemsClear: document.getElementById('systems-clear'),
		categorySelector: document.getElementById('category-selector'),
		heatmapMetrics: document.getElementById('heatmap-metrics'),
		heatmapSelectAll: document.getElementById('heatmap-select-all'),
		heatmapClear: document.getElementById('heatmap-clear'),
		parallelMetrics: document.getElementById('parallel-metrics'),
		parallelSelectAll: document.getElementById('parallel-select-all'),
		parallelClear: document.getElementById('parallel-clear'),
		scatterMetrics: document.getElementById('scatter-metrics'),
		scatterSelectAll: document.getElementById('scatter-select-all'),
		scatterClear: document.getElementById('scatter-clear'),
		radarChart: document.getElementById('radar-chart'),
		barChart: document.getElementById('bar-chart'),
		heatmapChart: document.getElementById('correlation-heatmap'),
		parallelChart: document.getElementById('parallel-coords'),
		scatterChart: document.getElementById('scatter-matrix'),
		tabButtons: Array.from(document.querySelectorAll('.tab-button')),
		tabPanels: Array.from(document.querySelectorAll('.tab-content')),
	};

	const state = {
		file: null,
		selectedSystems: [],
		selectedCategory: null,
		selectedHeatmapMetrics: [],
		selectedParallelMetrics: [],
		selectedScatterMetrics: [],
	};

	const plotConfig = { responsive: true, displaylogo: false };

	function checkedValues(containerEl) {
		return Array.from(containerEl.querySelectorAll('input[type="checkbox"]:checked')).map((input) => input.value);
	}

	function setSelectOptions(selectEl, values, selected) {
		const selectedSet = new Set(selected || []);
		selectEl.innerHTML = '';

		values.forEach((value) => {
			const option = document.createElement('option');
			option.value = value;
			option.textContent = value;
			option.selected = selectedSet.has(value);
			selectEl.appendChild(option);
		});
	}

	function setCheckboxOptions(containerEl, values, selected, groupName) {
		const selectedSet = new Set(selected || []);
		containerEl.innerHTML = '';

		values.forEach((value, idx) => {
			const label = document.createElement('label');
			label.className = 'checkbox-item';

			const checkbox = document.createElement('input');
			checkbox.type = 'checkbox';
			checkbox.name = groupName;
			checkbox.id = `${groupName}-${idx}`;
			checkbox.value = value;
			checkbox.checked = selectedSet.has(value);

			const text = document.createElement('span');
			text.textContent = value;

			label.appendChild(checkbox);
			label.appendChild(text);
			containerEl.appendChild(label);
		});
	}

	function setAllChecked(containerEl, checked) {
		containerEl.querySelectorAll('input[type="checkbox"]').forEach((input) => {
			input.checked = checked;
		});
	}

	function msgFigure(message) {
		return {
			data: [],
			layout: {
				annotations: [{
					text: message,
					x: 0.5,
					y: 0.5,
					xref: 'paper',
					yref: 'paper',
					showarrow: false,
					font: { size: 16 },
				}],
				xaxis: { visible: false },
				yaxis: { visible: false },
			},
		};
	}

	function currentDataset() {
		return appData.datasets[state.file] || null;
	}

	function rowsForSelection(data) {
		const metricIndex = new Map(data.metrics.map((metric, idx) => [metric, idx]));
		const systemsSet = new Set(state.selectedSystems);
		const rows = [];

		data.systems.forEach((systemName, rowIndex) => {
			if (!systemsSet.has(systemName)) {
				return;
			}
			const vector = data.dataMatrix[rowIndex];
			const row = { system_name: systemName };
			data.metrics.forEach((metric) => {
				row[metric] = Number(vector[metricIndex.get(metric)] || 0);
			});
			rows.push(row);
		});
		return rows;
	}

	function ensureRange(minVal, maxVal) {
		if (!Number.isFinite(minVal) || !Number.isFinite(maxVal)) {
			return [0, 1];
		}
		if (minVal === maxVal) {
			const delta = minVal === 0 ? 0.5 : Math.abs(minVal) * 0.1;
			return [minVal - delta, maxVal + delta];
		}
		return [minVal, maxVal];
	}

	function pearson(xs, ys) {
		const n = Math.min(xs.length, ys.length);
		if (n < 2) {
			return 0;
		}

		let sx = 0;
		let sy = 0;
		for (let i = 0; i < n; i += 1) {
			sx += xs[i];
			sy += ys[i];
		}
		const mx = sx / n;
		const my = sy / n;

		let num = 0;
		let dx2 = 0;
		let dy2 = 0;

		for (let i = 0; i < n; i += 1) {
			const dx = xs[i] - mx;
			const dy = ys[i] - my;
			num += dx * dy;
			dx2 += dx * dx;
			dy2 += dy * dy;
		}

		const den = Math.sqrt(dx2 * dy2);
		if (den === 0) {
			return 0;
		}
		return num / den;
	}

	function renderRadarAndBar() {
		const data = currentDataset();
		if (!data || !state.selectedCategory || state.selectedSystems.length === 0) {
			Plotly.react(refs.radarChart, msgFigure('No data selected').data, msgFigure('No data selected').layout, plotConfig);
			Plotly.react(refs.barChart, msgFigure('No data selected').data, msgFigure('No data selected').layout, plotConfig);
			return;
		}

		const categoryMetrics = data.availableCategories[state.selectedCategory] || [];
		const rows = rowsForSelection(data);

		if (categoryMetrics.length === 0 || rows.length === 0) {
			Plotly.react(refs.radarChart, msgFigure('No matching metrics found').data, msgFigure('No matching metrics found').layout, plotConfig);
			Plotly.react(refs.barChart, msgFigure('No matching metrics found').data, msgFigure('No matching metrics found').layout, plotConfig);
			return;
		}

		const radarData = rows.map((row) => ({
			type: 'scatterpolar',
			r: categoryMetrics.map((metric) => row[metric]),
			theta: categoryMetrics,
			fill: 'toself',
			name: row.system_name,
		}));

		Plotly.react(
			refs.radarChart,
			radarData,
			{
				title: `Radar Chart - ${state.selectedCategory}`,
				height: 600,
				polar: { radialaxis: { visible: true, range: [0, 1] } },
			},
			plotConfig,
		);

		const barData = categoryMetrics.map((metric) => ({
			type: 'bar',
			name: metric,
			x: rows.map((row) => row.system_name),
			y: rows.map((row) => row[metric]),
			text: rows.map((row) => Number(row[metric] || 0).toFixed(3)),
			textposition: 'auto',
		}));

		Plotly.react(
			refs.barChart,
			barData,
			{
				title: `Bar Chart Comparison - ${state.selectedCategory}`,
				barmode: 'group',
				height: 600,
				xaxis: { tickangle: -45 },
			},
			plotConfig,
		);
	}

	function renderHeatmap() {
		const data = currentDataset();
		if (!data) {
			Plotly.react(refs.heatmapChart, msgFigure('No data selected').data, msgFigure('No data selected').layout, plotConfig);
			return;
		}

		if (state.selectedSystems.length < 2 || state.selectedHeatmapMetrics.length < 2) {
			Plotly.react(refs.heatmapChart, msgFigure('Select at least 2 systems and 2 metrics').data, msgFigure('Select at least 2 systems and 2 metrics').layout, plotConfig);
			return;
		}

		const rows = rowsForSelection(data);
		const metrics = state.selectedHeatmapMetrics;

		const columns = metrics.map((metric) => rows.map((row) => Number(row[metric] || 0)));
		const corr = metrics.map((_, i) => metrics.map((__, j) => {
			const value = pearson(columns[i], columns[j]);
			return Number(value.toFixed(2));
		}));

		Plotly.react(
			refs.heatmapChart,
			[{
				type: 'heatmap',
				z: corr,
				x: metrics,
				y: metrics,
				colorscale: 'Turbo',
				zmin: -1,
				zmax: 1,
				text: corr,
				texttemplate: '%{text}',
				textfont: { size: 10 },
			}],
			{
				title: 'Metric Correlation Heatmap',
				height: 700,
				xaxis: { tickangle: -45 },
			},
			plotConfig,
		);
	}

	function renderParallel() {
		const data = currentDataset();
		if (!data) {
			Plotly.react(refs.parallelChart, msgFigure('No data selected').data, msgFigure('No data selected').layout, plotConfig);
			return;
		}

		if (state.selectedSystems.length < 2 || state.selectedParallelMetrics.length < 2) {
			Plotly.react(refs.parallelChart, msgFigure('Select at least 2 systems and 2 metrics').data, msgFigure('Select at least 2 systems and 2 metrics').layout, plotConfig);
			return;
		}

		const rows = rowsForSelection(data);
		const metrics = state.selectedParallelMetrics;
		const dimensions = metrics.map((metric) => {
			const vals = rows.map((row) => Number(row[metric] || 0));
			return {
				label: metric,
				values: vals,
				range: ensureRange(Math.min(...vals), Math.max(...vals)),
			};
		});

		Plotly.react(
			refs.parallelChart,
			[{
				type: 'parcoords',
				line: { color: rows.map((_, idx) => idx), colorscale: 'Viridis' },
				dimensions,
			}],
			{
				title: 'Parallel Coordinates Plot',
				height: 600,
			},
			plotConfig,
		);
	}

	function renderScatterMatrix() {
		const data = currentDataset();
		if (!data) {
			Plotly.react(refs.scatterChart, msgFigure('No data selected').data, msgFigure('No data selected').layout, plotConfig);
			return;
		}

		if (
			state.selectedSystems.length < 2 ||
			state.selectedScatterMetrics.length < 2 ||
			state.selectedScatterMetrics.length > 5
		) {
			Plotly.react(refs.scatterChart, msgFigure('Select at least 2 systems and 2 metrics (max 5 for readability)').data, msgFigure('Select at least 2 systems and 2 metrics (max 5 for readability)').layout, plotConfig);
			return;
		}

		const rows = rowsForSelection(data);
		const metrics = state.selectedScatterMetrics;

		const trace = {
			type: 'splom',
			dimensions: metrics.map((metric) => ({
				label: metric,
				values: rows.map((row) => Number(row[metric] || 0)),
			})),
			text: rows.map((row) => row.system_name),
			marker: {
				color: rows.map((_, idx) => idx),
				colorscale: 'Viridis',
				size: 8,
				opacity: 0.8,
			},
			diagonal: { visible: false },
			hovertemplate: '%{text}<extra></extra>',
		};

		Plotly.react(
			refs.scatterChart,
			[trace],
			{
				title: 'Scatter Matrix',
				height: 800,
			},
			plotConfig,
		);
	}

	function renderAll() {
		renderRadarAndBar();
		renderHeatmap();
		renderParallel();
		renderScatterMatrix();
	}

	function setAdvancedTabState(disabled) {
		refs.tabButtons.forEach((button) => {
			const tab = button.dataset.tab;
			if (tab === 'heatmap' || tab === 'parallel' || tab === 'scatter') {
				button.disabled = disabled;
			}
		});
	}

	function switchTab(tabName) {
		refs.tabButtons.forEach((button) => {
			button.classList.toggle('active', button.dataset.tab === tabName);
		});

		refs.tabPanels.forEach((panel) => {
			panel.classList.toggle('active', panel.id === `tab-${tabName}`);
		});
	}

	function onFileChanged() {
		state.file = refs.fileSelector.value;
		const data = currentDataset();

		if (!data) {
			refs.fileInfo.textContent = 'No file selected';
			return;
		}

		refs.fileInfo.textContent = `File: ${data.file} | Systems: ${data.systems.length} | Metrics: ${data.metrics.length}`;

		setCheckboxOptions(refs.systemSelector, data.systems, data.systems, 'systems');
		state.selectedSystems = [...data.systems];

		const categories = Object.keys(data.availableCategories);
		setSelectOptions(refs.categorySelector, categories, categories.length ? [categories[0]] : []);
		state.selectedCategory = categories.length ? categories[0] : null;

		setCheckboxOptions(refs.heatmapMetrics, data.metrics, [], 'heatmap-metrics');
		state.selectedHeatmapMetrics = [];
		setCheckboxOptions(refs.parallelMetrics, data.metrics, [], 'parallel-metrics');
		state.selectedParallelMetrics = [];
		setCheckboxOptions(refs.scatterMetrics, data.metrics, [], 'scatter-metrics');
		state.selectedScatterMetrics = [];

		setAdvancedTabState(data.systems.length < 2);
		renderAll();
	}

	function init() {
		const files = Object.keys(appData.datasets || {});

		if (!files.length) {
			refs.fileInfo.textContent = 'No datasets found';
			renderAll();
			return;
		}

		setSelectOptions(refs.fileSelector, files, [files[0]]);
		state.file = files[0];

		refs.fileSelector.addEventListener('change', onFileChanged);
		refs.systemSelector.addEventListener('change', () => {
			state.selectedSystems = checkedValues(refs.systemSelector);
			renderAll();
		});
		refs.systemsSelectAll.addEventListener('click', () => {
			setAllChecked(refs.systemSelector, true);
			state.selectedSystems = checkedValues(refs.systemSelector);
			renderAll();
		});
		refs.systemsClear.addEventListener('click', () => {
			setAllChecked(refs.systemSelector, false);
			state.selectedSystems = checkedValues(refs.systemSelector);
			renderAll();
		});
		refs.categorySelector.addEventListener('change', () => {
			state.selectedCategory = refs.categorySelector.value || null;
			renderRadarAndBar();
		});
		refs.heatmapMetrics.addEventListener('change', () => {
			state.selectedHeatmapMetrics = checkedValues(refs.heatmapMetrics);
			renderHeatmap();
		});
		refs.heatmapSelectAll.addEventListener('click', () => {
			setAllChecked(refs.heatmapMetrics, true);
			state.selectedHeatmapMetrics = checkedValues(refs.heatmapMetrics);
			renderHeatmap();
		});
		refs.heatmapClear.addEventListener('click', () => {
			setAllChecked(refs.heatmapMetrics, false);
			state.selectedHeatmapMetrics = checkedValues(refs.heatmapMetrics);
			renderHeatmap();
		});
		refs.parallelMetrics.addEventListener('change', () => {
			state.selectedParallelMetrics = checkedValues(refs.parallelMetrics);
			renderParallel();
		});
		refs.parallelSelectAll.addEventListener('click', () => {
			setAllChecked(refs.parallelMetrics, true);
			state.selectedParallelMetrics = checkedValues(refs.parallelMetrics);
			renderParallel();
		});
		refs.parallelClear.addEventListener('click', () => {
			setAllChecked(refs.parallelMetrics, false);
			state.selectedParallelMetrics = checkedValues(refs.parallelMetrics);
			renderParallel();
		});
		refs.scatterMetrics.addEventListener('change', () => {
			state.selectedScatterMetrics = checkedValues(refs.scatterMetrics);
			renderScatterMatrix();
		});
		refs.scatterSelectAll.addEventListener('click', () => {
			setAllChecked(refs.scatterMetrics, true);
			state.selectedScatterMetrics = checkedValues(refs.scatterMetrics);
			renderScatterMatrix();
		});
		refs.scatterClear.addEventListener('click', () => {
			setAllChecked(refs.scatterMetrics, false);
			state.selectedScatterMetrics = checkedValues(refs.scatterMetrics);
			renderScatterMatrix();
		});

		refs.tabButtons.forEach((button) => {
			button.addEventListener('click', () => {
				if (button.disabled) {
					return;
				}
				switchTab(button.dataset.tab);
			});
		});

		onFileChanged();
		switchTab('radar');
	}

	init();
})();
