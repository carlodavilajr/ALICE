// ALICE front end. Holds the current dataset id and calls the Flask API.

const state = { datasetId: null, summary: null, transforms: {} };

const $ = (id) => document.getElementById(id);

// ---------------------------------------------------------------- helpers

function show(id, visible = true) { $(id).hidden = !visible; }

function notify(message, ok = false) {
  const n = $("notice");
  n.textContent = message;
  n.className = ok ? "notice ok" : "notice";
  n.hidden = !message;
}

async function api(path, body, method = "POST") {
  const opts = method === "POST"
    ? { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }
    : { method };
  const res = await fetch(path, opts);
  const data = await res.json();
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

function fmt(v) {
  if (v === null || v === undefined) return "";
  if (typeof v === "number") return Math.abs(v) >= 1000 ? v.toLocaleString(undefined, { maximumFractionDigits: 2 }) : Number(v.toPrecision(5)).toString();
  return String(v);
}

function renderTable(table, columns, rows) {
  table.innerHTML = "";
  const thead = table.createTHead().insertRow();
  columns.forEach((c) => { const th = document.createElement("th"); th.textContent = c; thead.appendChild(th); });
  const tbody = table.createTBody();
  rows.forEach((r) => {
    const tr = tbody.insertRow();
    columns.forEach((c) => { tr.insertCell().textContent = fmt(r[c]); });
  });
}

function fillSelect(select, options, selected) {
  select.innerHTML = "";
  options.forEach(([value, label]) => {
    const o = document.createElement("option");
    o.value = value; o.textContent = label;
    if (value === selected) o.selected = true;
    select.appendChild(o);
  });
}

function selectedColumns() {
  return [...document.querySelectorAll("#column-checklist input:checked")].map((i) => i.value);
}

function transformParams() {
  const t = $("transform").value;
  if (t === "moving_average") return { window: Number($("window").value) };
  if (t === "real") return { deflator: $("deflator").value, base_position: Number($("base-position").value) };
  if (t === "index") return { base_position: Number($("index-base").value) };
  return {};
}

function commonBody() {
  return { dataset_id: state.datasetId, time_column: $("time-column").value || null };
}

// ---------------------------------------------------------------- dataset

function loadDataset(data) {
  state.datasetId = data.dataset_id;
  state.summary = data.summary;
  state.transforms = data.transforms;
  $("dataset-name").textContent = `${data.name}: ${data.summary.rows} rows, ${data.summary.columns} columns`;

  const numeric = data.summary.numeric_columns;
  const timeCol = data.summary.time_column;
  const allCols = Object.keys(data.summary.dtypes);

  fillSelect($("time-column"), [["", "(none, use row order)"], ...allCols.map((c) => [c, c])], timeCol || "");

  const list = $("column-checklist");
  list.innerHTML = "";
  numeric.filter((c) => c !== timeCol).forEach((c, i) => {
    const label = document.createElement("label");
    const box = document.createElement("input");
    box.type = "checkbox"; box.value = c; box.checked = i === 0;
    label.appendChild(box); label.appendChild(document.createTextNode(c));
    list.appendChild(label);
  });

  const tOpts = Object.entries(data.transforms);
  ["transform", "reg-x-t", "reg-y-t"].forEach((id) => fillSelect($(id), tOpts, "level"));
  const numOpts = numeric.map((c) => [c, c]);
  fillSelect($("deflator"), numOpts, numeric.find((c) => /cpi|deflator|price/i.test(c)) || numeric[0]);
  fillSelect($("reg-x"), numOpts, numeric[0]);
  fillSelect($("reg-y"), numOpts, numeric[1] || numeric[0]);

  renderTable($("preview-table"), allCols, data.preview);
  $("preview-meta").textContent = `first ${data.preview.length} rows`;

  const statCols = ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max", "missing"];
  const statRows = Object.entries(data.summary.stats).map(([col, s]) => ({ column: col, ...s }));
  renderTable($("stats-table"), statCols, statRows);

  show("empty-state", false);
  ["column-block", "regression-block", "other-block", "preview-panel", "stats-panel"].forEach((id) => show(id));
  show("chart-panel", false); show("notes-panel", false);
  notify(`Loaded ${data.name}. Time column detected: ${timeCol || "none"}.`, true);
  updateParamVisibility();
}

$("file-input").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("file-label").textContent = file.name;
  const form = new FormData();
  form.append("file", file);
  try {
    const res = await fetch("/api/upload", { method: "POST", body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error);
    loadDataset(data);
  } catch (err) { notify(err.message); }
});

$("btn-sample").addEventListener("click", async () => {
  try { loadDataset(await api("/api/sample", null, "GET")); } catch (err) { notify(err.message); }
});

// ---------------------------------------------------------------- transforms UI

function updateParamVisibility() {
  const t = $("transform").value;
  show("param-window", t === "moving_average");
  show("param-deflator", t === "real");
  show("param-base", t === "index");
}
$("transform").addEventListener("change", updateParamVisibility);

// ---------------------------------------------------------------- actions

function showChart(title, image) {
  $("chart-title").textContent = title;
  $("chart-image").src = `data:image/png;base64,${image}`;
  $("chart-download").href = `data:image/png;base64,${image}`;
  $("chart-stats").innerHTML = "";
  show("chart-panel");
  notify("");
}

$("btn-series").addEventListener("click", async () => {
  const columns = selectedColumns();
  if (!columns.length) return notify("Select at least one series.");
  try {
    const data = await api("/api/series", { ...commonBody(), columns, transform: $("transform").value,
      chart: $("chart-kind").value, params: transformParams() });
    showChart(`Series: ${state.transforms[$("transform").value]}`, data.image);
    const rows = Object.entries(data.stats).map(([name, s]) => ({ series: name, ...s }));
    const table = document.createElement("table");
    renderTable(table, ["series", "mean", "std", "min", "max", "cagr"], rows);
    $("chart-stats").appendChild(table);
  } catch (err) { notify(err.message); }
});

$("btn-export").addEventListener("click", async () => {
  const columns = selectedColumns();
  if (!columns.length) return notify("Select at least one series.");
  try {
    const res = await fetch("/api/export", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...commonBody(), columns, transform: $("transform").value, params: transformParams() }) });
    if (!res.ok) throw new Error((await res.json()).error);
    const blob = await res.blob();
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = `alice_${$("transform").value}.csv`; a.click();
    URL.revokeObjectURL(a.href);
  } catch (err) { notify(err.message); }
});

$("btn-regression").addEventListener("click", async () => {
  try {
    const data = await api("/api/regression", { ...commonBody(), x: $("reg-x").value, y: $("reg-y").value,
      x_transform: $("reg-x-t").value, y_transform: $("reg-y-t").value });
    showChart(`Regression: ${data.y_label} on ${data.x_label}`, data.image);
    const f = data.fit;
    const table = document.createElement("table");
    renderTable(table, ["n", "slope", "intercept", "r_squared", "correlation", "slope_std_error", "t_statistic", "equation"], [f]);
    $("chart-stats").appendChild(table);
  } catch (err) { notify(err.message); }
});

$("btn-correlation").addEventListener("click", async () => {
  const columns = selectedColumns();
  if (columns.length < 2) return notify("Select at least two series for a correlation matrix.");
  try {
    const data = await api("/api/correlation", { ...commonBody(), columns, transform: $("transform").value });
    showChart(`Correlation: ${state.transforms[$("transform").value]}`, data.image);
  } catch (err) { notify(err.message); }
});

$("btn-histogram").addEventListener("click", async () => {
  const columns = selectedColumns();
  if (!columns.length) return notify("Select a series first.");
  try {
    const data = await api("/api/histogram", { ...commonBody(), column: columns[0], transform: $("transform").value });
    showChart(`Distribution: ${columns[0]}`, data.image);
  } catch (err) { notify(err.message); }
});

$("btn-interpret").addEventListener("click", async () => {
  const columns = selectedColumns();
  if (!columns.length) return notify("Select at least one series.");
  try {
    const data = await api("/api/interpret", { ...commonBody(), columns });
    const box = $("notes");
    box.innerHTML = "";
    Object.entries(data.series).forEach(([name, lines]) => {
      const h = document.createElement("h3"); h.textContent = name; box.appendChild(h);
      const ul = document.createElement("ul");
      lines.forEach((l) => { const li = document.createElement("li"); li.textContent = l; ul.appendChild(li); });
      box.appendChild(ul);
    });
    if (data.relationships.length) {
      const h = document.createElement("h3"); h.textContent = "Relationships"; box.appendChild(h);
      const ul = document.createElement("ul");
      data.relationships.forEach((l, i) => {
        const li = document.createElement("li"); li.textContent = l;
        if (i === data.relationships.length - 1) li.className = "caveat";
        ul.appendChild(li);
      });
      box.appendChild(ul);
    }
    show("notes-panel");
    notify("");
  } catch (err) { notify(err.message); }
});
