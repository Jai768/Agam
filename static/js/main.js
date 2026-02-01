// --- DOM ELEMENTS ---
const inputArea = document.getElementById('journal-input');
const highlightLayer = document.getElementById('highlight-layer');
const tooltip = document.getElementById('hover-tooltip');
const modal = document.getElementById('courtroom-modal');

let activeDistortions = [];
let currentDistortion = null;

// --- DEBOUNCE HELPER ---
function debounce(fn, wait) {
    let t;
    return function(...args) {
        clearTimeout(t);
        t = setTimeout(() => fn.apply(this, args), wait);
    };
}

// --- SCROLL SYNC (Critical) ---
inputArea.addEventListener('scroll', () => {
    highlightLayer.scrollTop = inputArea.scrollTop;
});

// --- REAL-TIME ANALYSIS (debounced) ---
const debouncedAnalyze = debounce(analyzeEntry, 350);
inputArea.addEventListener('input', () => {
    // Update word count (in case editor.html didn't bind)
    const countEl = document.getElementById('word-count');
    if (countEl) {
        const count = inputArea.value.trim().split(/\s+/).filter(word => word.length > 0).length;
        countEl.innerText = count;
    }

    // Trigger analysis
    debouncedAnalyze();
});

// Trigger initial analysis for default textarea content
window.addEventListener('DOMContentLoaded', () => {
    analyzeEntry();
});

// --- API CALL ---
async function analyzeEntry() {
    const text = inputArea.value;
    // If empty, clear highlights and avoid request
    if (!text || text.trim().length === 0) {
        activeDistortions = [];
        highlightLayer.innerHTML = '';
        return;
    }

    try {
        const response = await fetch('/api/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await response.json();

        activeDistortions = data.distortions;
        renderHighlights(text, activeDistortions);
    } catch (e) {
        console.error("API Error:", e);
        // Don't spam alerts while typing; just clear highlights on error
        highlightLayer.innerHTML = '';
    }
}

// --- RENDERER ---
function renderHighlights(text, distortions) {
    if (!distortions || !distortions.length) {
        highlightLayer.innerHTML = ''; // Clear if none
        return;
    }

    let html = '';
    let lastIndex = 0;

    // Sort to be safe
    distortions.sort((a, b) => a.start - b.start);

    distortions.forEach((dist, index) => {
        // 1. Text before (Plain)
        html += escapeHtml(text.substring(lastIndex, dist.start));

        // 2. The Highlight (Span)
        // Note: The text inside is transparent in CSS, but background is visible
        html += `<span
            onmouseover="showTooltip(event, '${escapeHtml(dist.type)}')"
            onmouseout="hideTooltip()"
            onclick="openCourtroom(${index})">${escapeHtml(dist.text)}</span>`;

        lastIndex = dist.end;
    });

    // 3. Remaining text
    html += escapeHtml(text.substring(lastIndex));

    highlightLayer.innerHTML = html;
}

// Helper to escape HTML so user input doesn't break the highlight layer
function escapeHtml(str) {
    return str
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;')
        .replace(/\n/g, '<br>');
}

// --- TOOLTIP ---
function showTooltip(e, text) {
    const rect = e.target.getBoundingClientRect();
    const tooltipText = document.getElementById('tooltip-text');
    tooltipText.innerText = text;

    tooltip.style.display = 'block';
    tooltip.style.left = `${rect.left + (rect.width / 2)}px`;
    tooltip.style.top = `${rect.top}px`;
}

function hideTooltip() {
    tooltip.style.display = 'none';
}

// --- COURTROOM LOGIC ---
function openCourtroom(index) {
    currentDistortion = activeDistortions[index];

    // Set Header
    document.getElementById('court-claim').innerText = `"${currentDistortion.text}"`;
    document.getElementById('court-verdict').innerText = "Agents are deliberating...";

    // Clear Lists
    document.getElementById('prosecution-list').innerHTML = '';
    document.getElementById('defense-list').innerHTML = '';

    // Show Modal
    modal.classList.remove('hidden');

    // Start Simulation
    simulateAgents();
}

function closeCourtroom() {
    modal.classList.add('hidden');
}

async function simulateAgents() {
    // Fake streaming data
    const pros = ["It felt terrible.", "I saw them frowning.", "I feel anxious."];
    const def = ["Feelings aren't facts.", "No feedback given yet.", "I can improve."];

    // Add Prosecution
    for (let t of pros) {
        await new Promise(r => setTimeout(r, 500));
        addCard('prosecution-list', t, 'bg-white border-red-200 text-red-800');
    }

    // Add Defense
    for (let t of def) {
        await new Promise(r => setTimeout(r, 600));
        addCard('defense-list', t, 'bg-white border-teal-200 text-teal-800');
    }

    // Verdict
    await new Promise(r => setTimeout(r, 500));
    document.getElementById('court-verdict').innerText = "VERDICT: This thought is a distortion. Reframe suggesting a learning opportunity.";
}

function addCard(id, text, classes) {
    const div = document.createElement('div');
    div.className = `p-3 rounded border shadow-sm text-sm animate-in ${classes}`;
    div.innerText = text;
    document.getElementById(id).appendChild(div);
}

function acceptVerdict() {
    const reframe = "I am learning from this experience.";
    inputArea.value = inputArea.value.replace(currentDistortion.text, reframe);
    highlightLayer.innerHTML = ''; // Clear highlights
    closeCourtroom();
}