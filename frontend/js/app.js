const fmt = (n) => (n == null ? "—" : `$${Math.round(n).toLocaleString("en-US")}`);
const fmtPct = (n) => (n == null ? "—" : `${n.toFixed(1)}%`);
const fmtR2 = (n) => (n == null ? "—" : n.toFixed(4));

let baselineChart, algorithmsChart, scatterChart;

Chart.defaults.color = "#8b93a7";
Chart.defaults.borderColor = "rgba(255,255,255,0.06)";
Chart.defaults.font.family = "'Outfit', sans-serif";

async function fetchJSON(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} → ${res.status}`);
  return res.json();
}

function setHealth(data) {
  const badge = document.getElementById("health-badge");
  const ok = data.status === "healthy";
  badge.classList.toggle("healthy", ok);
  badge.querySelector("span:last-child").textContent = ok
    ? `Production · v${data.model_version || "1"}`
    : "Modelo indisponível";
}

function renderHero(overview) {
  document.getElementById("hero-tagline").textContent = overview.tagline;
  const bm = overview.best_model || {};
  document.getElementById("hero-best-mae").textContent = fmt(bm.mae);
  document.getElementById("hero-best-r2").textContent = `R² ${fmtR2(bm.r2)}`;

  const stats = overview.stats || {};
  document.getElementById("hero-metrics").innerHTML = [
    { label: "Experimentos MLflow", value: stats.total_experiments, gold: true },
    { label: "Features", value: stats.features_engineered },
    { label: "Algoritmos", value: stats.algorithms_count },
    { label: "MAE melhoria", value: `${Math.round(overview.baseline?.improvement_pct || 0)}%`, gold: true },
  ]
    .map(
      (m) => `
    <div class="metric-card">
      <span>${m.label}</span>
      <strong class="${m.gold ? "gold" : ""}">${m.value ?? "—"}</strong>
    </div>`
    )
    .join("");
}

function renderPipeline(stages) {
  const container = document.getElementById("pipeline-flow");
  container.innerHTML = stages
    .map((s, i) => {
      const arrow = i < stages.length - 1 ? '<div class="pipeline-arrow">›</div>' : "";
      return `
        ${arrow}
        <div class="pipeline-stage ${s.status === "active" ? "active" : ""}">
          <span class="phase">${s.phase}</span>
          <h3>${s.name}</h3>
          <p>${s.description}</p>
          <ul>${(s.outputs || []).map((o) => `<li>${o}</li>`).join("")}</ul>
        </div>`;
    })
    .join("");
}

function renderBaselineChart(overview) {
  const b = overview.baseline || {};
  const ctx = document.getElementById("baseline-chart");
  if (baselineChart) baselineChart.destroy();

  baselineChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["MAE", "RMSE"],
      datasets: [
        {
          label: "Naive (36 feat.)",
          data: [b.naive_mae, b.naive_rmse],
          backgroundColor: "rgba(139,147,167,0.5)",
          borderRadius: 8,
        },
        {
          label: "Engineered (277 feat.)",
          data: [b.engineered_mae, b.engineered_rmse],
          backgroundColor: "rgba(201,169,98,0.75)",
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (ctx) => `${ctx.dataset.label}: ${fmt(ctx.raw)}`,
          },
        },
      },
      scales: {
        y: {
          ticks: { callback: (v) => `$${(v / 1000).toFixed(0)}k` },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
      },
    },
  });

  document.getElementById("baseline-note").textContent =
    overview.baseline?.conclusion || "";
}

function renderAlgorithmsChart(algorithms) {
  const filtered = algorithms.filter(
    (a) => !["baseline_naive", "baseline_engineered"].includes(a.algorithm)
  );
  const ctx = document.getElementById("algorithms-chart");
  if (algorithmsChart) algorithmsChart.destroy();

  const colors = filtered.map((_, i) => {
    const hue = 160 + i * 18;
    return `hsla(${hue}, 55%, 55%, 0.75)`;
  });

  algorithmsChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: filtered.map((a) => a.label),
      datasets: [
        {
          label: "MAE ($)",
          data: filtered.map((a) => a.mae),
          backgroundColor: colors,
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (ctx) => fmt(ctx.raw) } },
      },
      scales: {
        x: {
          ticks: { callback: (v) => `$${(v / 1000).toFixed(0)}k` },
          grid: { color: "rgba(255,255,255,0.04)" },
        },
      },
    },
  });
}

function renderRegistry(overview) {
  const reg = overview.registry;
  const bm = overview.best_model || {};
  const el = document.getElementById("registry-banner");
  if (!reg) {
    el.innerHTML = "<p>Model Registry não configurado.</p>";
    return;
  }
  el.innerHTML = `
    <div>
      <span class="registry-badge">● Production</span>
      <strong style="display:block;margin-top:0.5rem">${reg.registered_model_name}</strong>
      <p style="color:var(--text-muted);font-size:0.85rem;margin-top:0.25rem">
        Run: ${reg.source_run_name} · MAE ${fmt(bm.mae)}
      </p>
    </div>
    <div style="text-align:right">
      <span style="font-size:0.75rem;color:var(--text-muted)">Versão</span>
      <div style="font-family:var(--font-display);font-size:2rem;color:var(--gold)">v${reg.model_version}</div>
    </div>`;
}

function renderAlgoTable(algorithms) {
  const tbody = document.querySelector("#algo-table tbody");
  const bestMae = Math.min(...algorithms.map((a) => a.mae));
  tbody.innerHTML = algorithms
    .map(
      (a) => `
    <tr>
      <td class="${a.mae === bestMae ? "best" : ""}">${a.label}</td>
      <td><code style="font-size:0.75rem">${a.run_name}</code></td>
      <td>${fmt(a.mae)}</td>
      <td>${fmt(a.rmse)}</td>
      <td>${fmtR2(a.r2)}</td>
      <td>${fmt(a.cv_mae)}</td>
    </tr>`
    )
    .join("");
}

function errorClass(pct) {
  const abs = Math.abs(pct);
  if (abs <= 5) return "good";
  if (abs <= 12) return "ok";
  return "warn";
}

function renderExamples(data) {
  const grid = document.getElementById("examples-grid");
  grid.innerHTML = (data.examples || [])
    .map(
      (ex) => `
    <div class="example-card">
      <span class="neighborhood">${ex.neighborhood}</span>
      <h4>${ex.house_style || "Residência"}</h4>
      <div class="example-meta">
        <span>${Math.round(ex.gr_liv_area)} sq ft</span>
        <span>Qual ${ex.overall_qual}</span>
        <span>${ex.year_built}</span>
      </div>
      <div class="price-row">
        <span class="label">Real</span>
        <span class="actual">${fmt(ex.actual_price)}</span>
      </div>
      <div class="price-row">
        <span class="label">Predito</span>
        <span class="predicted">${fmt(ex.predicted_price)}</span>
      </div>
      <span class="error-badge ${errorClass(ex.error_pct)}">
        Erro ${fmt(ex.error)} (${fmtPct(ex.error_pct)})
      </span>
    </div>`
    )
    .join("");

  renderScatter(data.scatter || []);
}

function renderScatter(points) {
  const ctx = document.getElementById("scatter-chart");
  if (scatterChart) scatterChart.destroy();

  const maxVal = Math.max(...points.flatMap((p) => [p.actual, p.predicted])) * 1.05;

  scatterChart = new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Holdout",
          data: points.map((p) => ({ x: p.actual, y: p.predicted })),
          backgroundColor: "rgba(201,169,98,0.55)",
          pointRadius: 4,
        },
        {
          label: "Ideal (y=x)",
          data: [
            { x: 0, y: 0 },
            { x: maxVal, y: maxVal },
          ],
          type: "line",
          borderColor: "rgba(52,211,153,0.5)",
          borderDash: [6, 4],
          pointRadius: 0,
          fill: false,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: "bottom" },
        tooltip: {
          callbacks: {
            label: (ctx) =>
              ctx.datasetIndex === 0
                ? `Real ${fmt(ctx.raw.x)} → Pred ${fmt(ctx.raw.y)}`
                : "",
          },
        },
      },
      scales: {
        x: {
          title: { display: true, text: "Preço real" },
          ticks: { callback: (v) => `$${(v / 1000).toFixed(0)}k` },
        },
        y: {
          title: { display: true, text: "Preço predito" },
          ticks: { callback: (v) => `$${(v / 1000).toFixed(0)}k` },
        },
      },
    },
  });
}

async function handlePredict(e) {
  e.preventDefault();
  const form = e.target;
  const fd = new FormData(form);
  const body = {};
  fd.forEach((v, k) => {
    body[k] = ["GrLivArea", "YearBuilt", "GarageCars", "FullBath", "OverallQual"].includes(k)
      ? Number(v)
      : v;
  });

  const priceEl = document.getElementById("result-price");
  priceEl.style.opacity = "0.4";

  try {
    const res = await fetch("/api/v1/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Erro na predição");

    priceEl.textContent = data.predicted_price_formatted;
    priceEl.style.opacity = "1";
    document.getElementById("result-interval").textContent =
      data.confidence_interval_low != null
        ? `Intervalo ±MAE: ${fmt(data.confidence_interval_low)} — ${fmt(data.confidence_interval_high)}`
        : "";
    document.getElementById("result-model").textContent = `Modelo: ${data.model_name}`;
  } catch (err) {
    priceEl.textContent = "Erro";
    priceEl.style.opacity = "1";
    document.getElementById("result-interval").textContent = err.message;
  }
}

async function init() {
  try {
    const [health, overview, algos, examples] = await Promise.all([
      fetchJSON("/api/v1/health"),
      fetchJSON("/api/v1/dashboard/overview"),
      fetchJSON("/api/v1/dashboard/algorithms"),
      fetchJSON("/api/v1/dashboard/examples"),
    ]);

    setHealth(health);
    renderHero(overview);
    renderPipeline(overview.pipeline_stages);
    renderBaselineChart(overview);
    renderAlgorithmsChart(algos.algorithms);
    renderRegistry(overview);
    renderAlgoTable(algos.algorithms);
    renderExamples(examples);
  } catch (err) {
    console.error(err);
    document.getElementById("health-badge").querySelector("span:last-child").textContent =
      "API offline";
  }

  document.getElementById("predict-form").addEventListener("submit", handlePredict);
}

init();
