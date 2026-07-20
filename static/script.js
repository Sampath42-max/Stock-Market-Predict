/**
 * StockPulse Main Interactive Script
 * Supports both Flask Server (port 5000) & Live Server (port 5500 / file://)
 */

const API_BASE = (window.location.port === '5500' || window.location.protocol === 'file:') ? 'http://127.0.0.1:5000' : '';

const FALLBACK_STOCKS = [
    { Ticker: "AAPL", Close: 234.50, Change: 2.30, ChangePct: 0.99, High: 236.00, Low: 232.10, Volume: 52400000 },
    { Ticker: "MSFT", Close: 448.90, Change: 4.10, ChangePct: 0.92, High: 450.20, Low: 445.00, Volume: 21300000 },
    { Ticker: "NVDA", Close: 128.40, Change: 4.35, ChangePct: 3.50, High: 130.10, Low: 124.50, Volume: 89100000 },
    { Ticker: "TSLA", Close: 245.10, Change: -3.48, ChangePct: -1.40, High: 251.00, Low: 242.30, Volume: 64200000 },
    { Ticker: "GOOGL", Close: 178.30, Change: 0.89, ChangePct: 0.50, High: 179.80, Low: 176.90, Volume: 28400000 },
    { Ticker: "AMZN", Close: 184.20, Change: 2.01, ChangePct: 1.10, High: 186.00, Low: 182.50, Volume: 34100000 },
    { Ticker: "PLTR", Close: 28.40, Change: 1.15, ChangePct: 4.22, High: 29.10, Low: 27.80, Volume: 45100000 },
    { Ticker: "META", Close: 495.20, Change: 6.80, ChangePct: 1.39, High: 498.00, Low: 489.10, Volume: 15600000 },
    { Ticker: "RELIANCE.NS", Close: 3120.50, Change: 24.30, ChangePct: 0.78, High: 3140.00, Low: 3100.00, Volume: 8400000 },
    { Ticker: "TCS.NS", Close: 4250.00, Change: 38.50, ChangePct: 0.91, High: 4280.00, Low: 4210.00, Volume: 3200000 },
    { Ticker: "INFY.NS", Close: 1835.40, Change: 14.20, ChangePct: 0.78, High: 1850.00, Low: 1820.00, Volume: 6500000 },
    { Ticker: "BTC-USD", Close: 64250.00, Change: 1420.00, ChangePct: 2.26, High: 65100.00, Low: 63800.00, Volume: 28400000000 }
];

document.addEventListener('DOMContentLoaded', () => {
    fixLiveServerLinks();
    initMarketStatus();
    initGlobalSearch();
    sanitizeAndLoadTable();
});

// Auto-fix navigation links if opened directly via Live Server
function fixLiveServerLinks() {
    if (window.location.port === '5500' || window.location.protocol === 'file:') {
        document.querySelectorAll('a[href="/"]').forEach(a => a.href = 'index.html');
        document.querySelectorAll('a[href="/analysis"]').forEach(a => a.href = 'market.html');
        document.querySelectorAll('a[href="/share-info"]').forEach(a => a.href = 'plot.html');
    }
}

// Fetch and update market open/closed status badge
async function initMarketStatus() {
    const statusBadge = document.getElementById('market-status');
    const statusText = document.getElementById('status-text');

    if (!statusBadge) return;

    try {
        const response = await fetch(`${API_BASE}/api/market-summary`);
        const data = await response.json();

        if (data.market_open) {
            statusBadge.className = 'status-badge open';
            if (statusText) statusText.innerText = 'Market Open';
        } else {
            statusBadge.className = 'status-badge closed';
            if (statusText) statusText.innerText = 'Market Closed';
        }
    } catch (err) {
        statusBadge.className = 'status-badge closed';
        if (statusText) statusText.innerText = 'Market Closed';
    }
}

// Global Ticker Live Search Handler
function initGlobalSearch() {
    const searchInput = document.getElementById('home-search-input');
    const dropdown = document.getElementById('search-dropdown');

    if (!searchInput || !dropdown) return;

    let debounceTimer;

    searchInput.addEventListener('input', (e) => {
        clearTimeout(debounceTimer);
        const query = e.target.value.trim();

        if (query.length < 1) {
            dropdown.style.display = 'none';
            return;
        }

        debounceTimer = setTimeout(async () => {
            try {
                const response = await fetch(`${API_BASE}/api/search?q=${encodeURIComponent(query)}`);
                const matches = await response.json();

                if (matches.length === 0) {
                    dropdown.innerHTML = `<div class="search-item" style="color: var(--text-dim)">No matching stocks found</div>`;
                } else {
                    dropdown.innerHTML = matches.map(ticker => `
                        <div class="search-item" onclick="navigateToStock('${ticker}')">
                            <span class="ticker-name">${ticker}</span>
                            <span style="font-size: 0.8rem; color: var(--primary-cyan)">Analyze & Predict &rarr;</span>
                        </div>
                    `).join('');
                }
                dropdown.style.display = 'block';
            } catch (err) {
                console.error("Search fetch error:", err);
            }
        }, 250);
    });

    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !dropdown.contains(e.target)) {
            dropdown.style.display = 'none';
        }
    });
}

function navigateToStock(ticker) {
    const target = (window.location.port === '5500' || window.location.protocol === 'file:') ? 'plot.html' : '/share-info';
    window.location.href = `${target}?ticker=${encodeURIComponent(ticker)}`;
}

// Sanitizes table and loads stock rows client-side if running in Live Server or if Jinja was unparsed
async function sanitizeAndLoadTable() {
    // 1. Remove foster-parented Jinja text nodes from table containers & headers
    document.querySelectorAll(".table-responsive, .glass-panel, table, thead, tbody, tr").forEach(parent => {
        parent.childNodes.forEach(node => {
            if (node.nodeType === Node.TEXT_NODE && (node.nodeValue.includes("{%") || node.nodeValue.includes("{{"))) {
                node.nodeValue = "";
            }
        });
    });

    const tbody = document.getElementById("table-body");
    if (!tbody) return;

    // Check if table contains raw Jinja text or no valid <tr> elements
    const isUnparsedJinja = tbody.innerHTML.includes("{%") || tbody.innerHTML.includes("{{") || !tbody.querySelector("tr");

    if (isUnparsedJinja || window.location.port === '5500' || window.location.protocol === 'file:') {
        tbody.innerHTML = ""; // Clear raw text node clutter

        let stocksToDisplay = FALLBACK_STOCKS;

        try {
            const res = await fetch(`${API_BASE}/api/market-summary`);
            if (res.ok) {
                const data = await res.json();
                if (data.stocks && data.stocks.length > 0) {
                    stocksToDisplay = data.stocks;
                }
            }
        } catch (e) {
            console.log("Using offline stock sample data");
        }

        renderStockTableRows(tbody, stocksToDisplay);
    }
}

function renderStockTableRows(tbody, stocks) {
    let html = '';
    stocks.forEach((stock, idx) => {
        const changeClass = stock.ChangePct >= 0 ? 'up' : 'down';
        const priceClass = stock.ChangePct >= 0 ? 'price-up' : 'price-down';
        const sign = stock.ChangePct >= 0 ? '+' : '';
        const targetPage = (window.location.port === '5500' || window.location.protocol === 'file:') ? 'plot.html' : '/share-info';

        html += `
            <tr data-ticker="${stock.Ticker}" data-change="${stock.ChangePct}">
                <td style="color: var(--text-dim);">${idx + 1}</td>
                <td>
                    <div class="ticker-badge">
                        <div class="ticker-icon">${stock.Ticker.substring(0, 2)}</div>
                        <span>${stock.Ticker}</span>
                    </div>
                </td>
                <td style="font-weight: 700; color: var(--text-main);">$${stock.Close.toFixed(2)}</td>
                <td><span class="${priceClass}">${sign}$${stock.Change.toFixed(2)}</span></td>
                <td><span class="change-pill ${changeClass}">${sign}${stock.ChangePct.toFixed(2)}%</span></td>
                <td>$${stock.High.toFixed(2)}</td>
                <td>$${stock.Low.toFixed(2)}</td>
                <td>${stock.Volume ? stock.Volume.toLocaleString() : 'N/A'}</td>
                <td style="text-align: center;">-</td>
                <td style="text-align: center;">
                    <a href="${targetPage}?ticker=${stock.Ticker}" class="action-link">Predict &rarr;</a>
                </td>
            </tr>
        `;
    });
    tbody.innerHTML = html;
}
