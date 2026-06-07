console.log("✅ JS LOADED");

const API = "http://127.0.0.1:5000/api";

let loadChart = null;
let stressChart = null;
let currentView = "live";

// ===============================
// SAFE FETCH HELPER
// ===============================
async function safeFetch(url) {
    const res = await fetch(url);
    const text = await res.text();

    try {
        return JSON.parse(text.replace(/NaN/g, "0"));
    } catch (err) {
        console.error("❌ INVALID JSON FROM:", url);
        console.error(text);
        throw err;
    }
}

// ===============================
// TIME
// ===============================
function updateTime() {
    const now = new Date();
    document.getElementById("current-time").textContent =
        now.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });
}

// ===============================
// MAIN UPDATE
// ===============================
async function updateDashboard() {
    try {
        console.log("🔄 Updating dashboard...");
        if (currentView === "live") {
            await loadLiveView();
        } else {
            await loadTomorrowView();
        }
    } catch (err) {
        console.error("❌ DASHBOARD ERROR:", err);
    }
}

// ===============================
// LIVE VIEW
// ===============================
async function loadLiveView() {
    console.log("📡 Loading LIVE view...");

    const model = document.getElementById("model-select").value;
    const zone = document.getElementById("area-select").value;

    const status = await safeFetch(
        `${API}/status?model=${model}&zone=${zone}`
    );

    const adaptive = await safeFetch(
        `${API}/adaptive?zone=${zone}`
    );

    if (!adaptive || !adaptive.length) {
        console.warn("⚠️ No adaptive data received");
        return;
    }

    const last = adaptive[adaptive.length - 1];

    const currentLoad = Number(last["adaptive_net_load"] || 0);
    const peakLoad = Math.max(
        ...adaptive.slice(-24).map(x => Number(x["adaptive_net_load"] || 0))
    );

    document.getElementById("current-load").textContent = currentLoad.toFixed(2);
    document.getElementById("peak-load").textContent = peakLoad.toFixed(2);
    document.getElementById("forecast-error").textContent = "—";
    document.getElementById("monthly-cost").textContent =
        ((currentLoad * 2.4) / 100).toFixed(2);

    document.getElementById("risk-index").textContent =
        (status.risk * 100).toFixed(1);

    document.getElementById("adaptive-storage").textContent =
        status.storage.toFixed(2);

    document.getElementById("adaptive-shedding").textContent =
        status.shedding.toFixed(2);

    document.getElementById("grid-status").textContent =
        status.risk > 0.25 ? "RISK" : "OK";

    drawStressChart(adaptive.slice(-48));
}


// ===============================
// TOMORROW FORECAST
// ===============================
async function loadTomorrowView() {
    console.log("📡 Loading TOMORROW forecast...");

    const model = document.getElementById("model-select").value;
    const zone = document.getElementById("area-select").value;

    const forecast = await safeFetch(
        `${API}/forecast?model=${model}&zone=${zone}`
    );

    if (!forecast || !forecast.length) {
        console.warn("⚠️ No forecast data received");
        return;
    }

    // ✅ FIXED FIELD NAME
    const hourly = forecast.map(row => Number(row.load || 0)).slice(0, 24);

    const peak = Math.max(...hourly);

    document.getElementById("current-load").textContent = hourly[0].toFixed(2);
    document.getElementById("peak-load").textContent = peak.toFixed(2);
    document.getElementById("forecast-error").textContent = "±2.1";
    document.getElementById("monthly-cost").textContent =
        ((peak * 2.4) / 100).toFixed(2);

    updateChart(hourly);
    updateTable(hourly);
}


// ===============================
// FORECAST CHART
// ===============================
function updateChart(hourly) {
    const canvas = document.getElementById("loadChart");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (loadChart) loadChart.destroy();

    const labels = Array.from({ length: 24 }, (_, i) => `${i}:00`);

    loadChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: labels,
            datasets: [{
                label: "Predicted Load (MW)",
                data: hourly,
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        }
    });
}


// ===============================
// STRESS CHART
// ===============================
function drawStressChart(data) {
    const ctx = document.getElementById("stressChart").getContext("2d");
    if (stressChart) stressChart.destroy();

    const staticStress = data.map(x => Number(x.stress_res || 0));
    const adaptiveStress = data.map(x => Number(x.adaptive_stress || 0));

    stressChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: staticStress.map((_, i) => i),
            datasets: [
                { label: "Static Stress", data: staticStress },
                { label: "Adaptive Stress", data: adaptiveStress }
            ]
        }
    });
}

// ===============================
// TABLE
// ===============================
function updateTable(hourly) {
    const tbody = document.getElementById("table-body");
    tbody.innerHTML = "";

    hourly.forEach((load, i) => {
        let status = "NORMAL";
        if (load > hourly[0] * 1.2) status = "HIGH";
        if (load < hourly[0] * 0.7) status = "LOW";

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td>${i}:00</td>
            <td>${load.toFixed(2)}</td>
            <td>${status}</td>
        `;
        tbody.appendChild(tr);
    });
}

// ===============================
// EVENTS
// ===============================
document.addEventListener("DOMContentLoaded", () => {
    console.log("✅ DOM READY");

    document.querySelectorAll("input[name='view-mode']").forEach(radio => {
        radio.addEventListener("change", e => {
            currentView = e.target.value;
            updateDashboard();
        });
    });

    document.getElementById("update-btn").addEventListener("click", updateDashboard);

    document.getElementById("model-select")
    .addEventListener("change", updateDashboard);




    updateTime();
    setInterval(updateTime, 1000);

    updateDashboard();
});
