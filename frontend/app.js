// ═══════════════════════════════════════════════════════════
// GLOBAL STATE & MAP INITIALIZATION
// ═══════════════════════════════════════════════════════════
let liveMap, liveTrackPath;
let histMap, histPolyline;
let ws;

// Initialize Live Map (Called by DOMContentLoaded)
function initLiveMap() {
    const mapEl = document.getElementById('live-gps-map');
    if (!mapEl) return;
    liveMap = L.map('live-gps-map').setView([52.4862, -1.8904], 16);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(liveMap);
    liveTrackPath = L.polyline([], { color: '#00ff00', weight: 5 }).addTo(liveMap);
}

function initMap() {
    const mapEl = document.getElementById('gps-map');
    
    // Only proceed if the element exists on the CURRENT page
    if (!mapEl) {
        console.log("Map element not found on this page, skipping init.");
        return; 
    }

    // Now it's safe to initialize
    map = L.map('gps-map').setView([52.4862, -1.8904], 15);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
}

//Initialize Chart.js

function initChart() {
    const ctx = document.getElementById('telemetryChart');
    if (!ctx) return;
    telemChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels: [],
            datasets: [{ label: 'Speed (km/h)', data: [], borderColor: '#00ff00', tension: 0.1 }]
        }
    });
} 

function updateStatusDisplay(text, color) {
    const badge = document.getElementById("connection-status");
    if (badge) {
        badge.innerText = text;
        badge.style.backgroundColor = color;
    }
    const grid = document.getElementById("metrics-grid");
    if (grid) {
        grid.style.opacity = text !== "CONNECTED" ? "0.4" : "1.0";
    }
}

function resetTelemetryTimeout() {
    clearTimeout(window.telemetryTimeout);
    window.telemetryTimeout = setTimeout(() => {
        updateStatusDisplay("CAR OUT OF RANGE", "var(--warning)");
    }, 2500);
}

// ═══════════════════════════════════════════════════════════
// WEBSOCKET & LIVE DATA HANDLING
// ═══════════════════════════════════════════════════════════
function connectTelemetryWS() {
    if (!document.getElementById('live-gps-map')) return;

    // Fixed: Remove 'const' so we assign to the global 'ws' variable
    ws = new WebSocket(`ws://${window.location.host}/ws/telemetry`);

    ws.onopen = () => {
        updateStatusDisplay("CONNECTED", "var(--success)");
        resetTelemetryTimeout();
    };

    ws.onmessage = (event) => {
        try {
            const d = JSON.parse(event.data);
            resetTelemetryTimeout();

            if (d.latitude && d.longitude && d.latitude !== 0) {
                const newPoint = [d.latitude, d.longitude];
                if (liveTrackPath) liveTrackPath.addLatLng(newPoint);
                if (liveMap) liveMap.panTo(newPoint);
            }
        } catch (e) {
            console.error("Telemetry parse error:", e);
        }
    };

    ws.onclose = () => {
        updateStatusDisplay("BACKEND OFFLINE", "var(--danger)");
        setTimeout(connectTelemetryWS, 3000); 
    };
}
// ═══════════════════════════════════════════════════════════
// HISTORICAL MAPPING
// ═══════════════════════════════════════════════════════════
function renderHistoricalMap(rows) {
    const mapEl = document.getElementById('hist-gps-map');
    if (!mapEl) return;

    if (!histMap) {
        histMap = L.map('hist-gps-map').setView([52.4862, -1.8904], 15);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(histMap);
    } else {
        // Essential: Trigger this when the container becomes visible
        histMap.invalidateSize(); 
    }
    
    if (histPolyline) histMap.removeLayer(histPolyline);

    const pathCoords = rows
        .filter(r => r.latitude && r.longitude)
        .map(r => [parseFloat(r.latitude), parseFloat(r.longitude)]);

    if (pathCoords.length > 0) {
        histPolyline = L.polyline(pathCoords, { color: '#5F588A', weight: 4 }).addTo(histMap);
        histMap.fitBounds(histPolyline.getBounds());
    }
}

// ═══════════════════════════════════════════════════════════
// INITIALIZATION
// ═══════════════════════════════════════════════════════════

// Example of how to structure your initialization
function initApp() {
    connectTelemetryWS();
    // 1. Check if the element exists BEFORE initializing
    const liveChartEl = document.getElementById('telemetryChart');
    if (liveChartEl) {
        initLiveCharts();
        initLiveMap();
    }

    // 2. Check if the historical elements exist
    const histChartEl = document.getElementById('historicalChart');
    if (histChartEl) {
        initHistoricalCharts();
    }
}

// Ensure it runs after the DOM is ready
document.addEventListener('DOMContentLoaded', initApp);