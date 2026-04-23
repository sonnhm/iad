/**
 * IAD Demo — Client-side JavaScript (v2 — Batch Processing)
 * Supports: multi-file upload, gallery preview, batch inference, summary dashboard
 */

const MAX_IMAGES = 12;

// ============================================================
// DOM Elements
// ============================================================

const uploadArea      = document.getElementById("uploadArea");
const fileInput       = document.getElementById("fileInput");
const previewSection  = document.getElementById("previewSection");
const previewGallery  = document.getElementById("previewGallery");
const selectedCount   = document.getElementById("selectedCount");
const clearBtn        = document.getElementById("clearBtn");
const predictBtn      = document.getElementById("predictBtn");
const resultsPanel    = document.getElementById("resultsPanel");
const resultsContainer = document.getElementById("resultsContainer");
const categorySelect  = document.getElementById("category");
const modelSelect     = document.getElementById("model");

let selectedFiles = []; // Array of File objects

// ============================================================
// Upload Handling
// ============================================================

uploadArea.addEventListener("click", () => fileInput.click());

fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
        handleFiles(Array.from(e.target.files));
    }
});

uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
});

uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
});

uploadArea.addEventListener("drop", (e) => {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files.length > 0) {
        handleFiles(Array.from(e.dataTransfer.files));
    }
});

function handleFiles(files) {
    // Filter to only images
    const imageFiles = files.filter(f => f.type.startsWith("image/"));
    if (imageFiles.length === 0) {
        alert("Vui lòng chọn file ảnh (PNG, JPG, BMP)!");
        return;
    }

    // Enforce limit
    if (imageFiles.length > MAX_IMAGES) {
        alert(`Tối đa ${MAX_IMAGES} ảnh mỗi lượt. Hệ thống sẽ chỉ lấy ${MAX_IMAGES} ảnh đầu tiên.`);
        selectedFiles = imageFiles.slice(0, MAX_IMAGES);
    } else {
        selectedFiles = imageFiles;
    }

    renderPreviewGallery();
    uploadArea.style.display = "none";
    previewSection.style.display = "block";
    predictBtn.disabled = false;

    // Reset old results
    resultsPanel.style.display = "none";
    resultsContainer.innerHTML = "";
}

function renderPreviewGallery() {
    previewGallery.innerHTML = "";
    selectedCount.textContent = `${selectedFiles.length}/${MAX_IMAGES}`;

    selectedFiles.forEach((file) => {
        const thumb = document.createElement("div");
        thumb.className = "preview-thumb";

        const img = document.createElement("img");
        const reader = new FileReader();
        reader.onload = (e) => { img.src = e.target.result; };
        reader.readAsDataURL(file);

        const nameEl = document.createElement("div");
        nameEl.className = "thumb-name";
        nameEl.textContent = file.name;

        thumb.appendChild(img);
        thumb.appendChild(nameEl);
        previewGallery.appendChild(thumb);
    });
}

// Clear
clearBtn.addEventListener("click", () => {
    selectedFiles = [];
    fileInput.value = "";
    uploadArea.style.display = "block";
    previewSection.style.display = "none";
    previewGallery.innerHTML = "";
    selectedCount.textContent = "";
    predictBtn.disabled = true;
    resultsPanel.style.display = "none";
    resultsContainer.innerHTML = "";
});

// ============================================================
// Predict — Batch Submission
// ============================================================

predictBtn.addEventListener("click", async () => {
    if (selectedFiles.length === 0) return;

    // UI: loading state
    const btnText    = predictBtn.querySelector(".btn-text");
    const btnLoading = predictBtn.querySelector(".btn-loading");
    btnText.style.display    = "none";
    btnLoading.style.display = "inline";
    predictBtn.disabled = true;

    // Show progress section immediately
    resultsContainer.innerHTML = "";
    resultsPanel.style.display = "block";
    resultsPanel.scrollIntoView({ behavior: "smooth", block: "start" });

    const progressCard = renderProgressCard(0, selectedFiles.length);
    resultsContainer.appendChild(progressCard);

    try {
        const formData = new FormData();
        selectedFiles.forEach(file => formData.append("image", file));
        formData.append("category", categorySelect.value);
        formData.append("model",    modelSelect.value);
        if (categorySelect.value === "auto") {
            formData.append("auto_detect", "true");
        }

        // Animate progress bar during fetch (indeterminate feel)
        let fakeProgress = 10;
        const fakeInterval = setInterval(() => {
            fakeProgress = Math.min(fakeProgress + 3, 85);
            updateProgressCard(progressCard, fakeProgress, selectedFiles.length,
                `Đang xử lý ${selectedFiles.length} ảnh...`);
        }, 400);

        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        clearInterval(fakeInterval);
        updateProgressCard(progressCard, 100, selectedFiles.length,
            `Hoàn tất — ${selectedFiles.length} ảnh đã phân tích.`);

        const data = await response.json();

        if (data.error) {
            showError(data.error);
        } else {
            // Small delay to show 100% before replacing
            setTimeout(() => {
                renderBatchResults(data.batch_results);
            }, 500);
        }
    } catch (error) {
        showError("Lỗi kết nối server: " + error.message);
    } finally {
        btnText.style.display    = "inline";
        btnLoading.style.display = "none";
        predictBtn.disabled = false;
    }
});

// ============================================================
// Progress Card
// ============================================================

function renderProgressCard(current, total) {
    const card = document.createElement("div");
    card.className = "batch-progress-card";
    card.innerHTML = `
        <h3> Đang phân tích lô ảnh...</h3>
        <div class="progress-bar-wrap">
            <div class="progress-bar-fill" id="progressBarFill" style="width: 5%;"></div>
        </div>
        <p class="progress-status-text" id="progressStatusText">Chuẩn bị gửi ${total} ảnh đến máy chủ...</p>
    `;
    return card;
}

function updateProgressCard(card, pct, total, msg) {
    const fill = card.querySelector("#progressBarFill");
    const txt  = card.querySelector("#progressStatusText");
    if (fill) fill.style.width = `${pct}%`;
    if (txt)  txt.textContent  = msg;
}

// ============================================================
// Render Batch Results
// ============================================================

function renderBatchResults(batchResults) {
    resultsContainer.innerHTML = "";

    if (!batchResults || batchResults.length === 0) {
        showError("Không có kết quả nào được trả về.");
        return;
    }

    // ——— 1. Compute summary stats ———
    let normalCount  = 0;
    let anomalyCount = 0;
    let errorCount   = 0;

    batchResults.forEach(item => {
        if (item.error) {
            errorCount++;
            return;
        }
        // Determine verdict from first available result
        const primaryResult = item.results && item.results[0];
        if (primaryResult) {
            if (primaryResult.is_anomaly) anomalyCount++;
            else normalCount++;
        }
    });

    const validCount  = normalCount + anomalyCount;
    const healthPct   = validCount > 0 ? Math.round((normalCount / validCount) * 100) : 0;

    // ——— 2. Batch Summary Dashboard ———
    const summaryCard = document.createElement("div");
    summaryCard.className = "batch-summary-card";
    summaryCard.innerHTML = `
        <p class="batch-summary-title"> Tổng hợp lô phân tích</p>
        <div class="batch-summary-stats">
            <div class="stat-box total">
                <div class="stat-value">${batchResults.length}</div>
                <div class="stat-label"> Tổng ảnh</div>
            </div>
            <div class="stat-box normal">
                <div class="stat-value">${normalCount}</div>
                <div class="stat-label"> Bình thường</div>
            </div>
            <div class="stat-box anomaly">
                <div class="stat-value">${anomalyCount}</div>
                <div class="stat-label"> Phát hiện lỗi</div>
            </div>
            <div class="stat-box health">
                <div class="stat-value">${healthPct}%</div>
                <div class="stat-label"> Tỉ lệ đạt</div>
            </div>
        </div>
        <div class="batch-health-bar-wrap">
            <div class="batch-health-bar-fill" id="batchHealthBar" style="width: 0%;"></div>
        </div>
    `;
    resultsContainer.appendChild(summaryCard);

    // Animate health bar
    requestAnimationFrame(() => {
        setTimeout(() => {
            const bar = document.getElementById("batchHealthBar");
            if (bar) bar.style.width = `${healthPct}%`;
        }, 100);
    });

    // ——— 3. Set chatbot context to first anomaly (or first result) ———
    const firstAnomaly = batchResults.find(item =>
        item.results && item.results[0] && item.results[0].is_anomaly
    );
    const contextSource = firstAnomaly || batchResults.find(item => item.results && item.results[0]);
    if (contextSource) {
        const r = contextSource.results[0];
        window.chatContext = {
            category:   contextSource.category || categorySelect.value,
            model:      r.model,
            score:      r.score,
            threshold:  r.threshold,
            is_anomaly: r.is_anomaly,
        };
    }

    // ——— 4. Per-image result blocks ———
    batchResults.forEach((item, index) => {
        const block = document.createElement("div");
        block.className = "image-result-block";
        block.style.animationDelay = `${index * 0.06}s`;

        // Determine header badge
        let headerBadgeHTML = "";
        if (item.error) {
            headerBadgeHTML = `<span class="result-badge badge-anomaly"> Lỗi xử lý</span>`;
        } else {
            const primary = item.results && item.results[0];
            if (primary) {
                const isAno = primary.is_anomaly;
                headerBadgeHTML = `<span class="result-badge ${isAno ? 'badge-anomaly' : 'badge-normal'}">
                    ${isAno ? ' ANOMALY' : ' NORMAL'}
                </span>`;
            }
        }

        // Warning
        const warningHTML = item.warning
            ? `<div class="result-error" style="margin-bottom:1rem;"> ${item.warning}</div>`
            : "";

        // Model result cards inside the block
        let modelCardsHTML = "";
        if (item.error) {
            modelCardsHTML = `<div class="result-error">${item.error}</div>`;
        } else if (item.results && item.results.length > 0) {
            item.results.forEach((result, ri) => {
                modelCardsHTML += buildModelCardHTML(result, ri);
            });
        } else {
            modelCardsHTML = `<div class="result-error">Không có kết quả từ model.</div>`;
        }

        const autoTag = item.auto_detected
            ? `<span style="font-size:0.75rem;color:var(--accent);margin-left:6px;">(YOLO: ${item.auto_detected.category} ${(item.auto_detected.confidence*100).toFixed(0)}%)</span>`
            : "";

        block.innerHTML = `
            <div class="image-result-header" onclick="toggleBlock(this.parentElement)">
                <div class="image-result-filename">
                    <span class="img-index-badge">#${index + 1}</span>
                    ${item.filename}${autoTag}
                </div>
                <div style="display:flex;align-items:center;gap:0.75rem;">
                    ${headerBadgeHTML}
                    <span class="collapse-icon">▼</span>
                </div>
            </div>
            <div class="image-result-body">
                ${warningHTML}
                <div class="result-card" style="margin:0;padding:0;border:none;box-shadow:none;">
                    ${modelCardsHTML}
                </div>
            </div>
        `;

        resultsContainer.appendChild(block);
    });

    // Trigger bar fill animations now that all DOM is ready
    animateRiskMeters();
}

function toggleBlock(blockEl) {
    blockEl.classList.toggle("collapsed");
}

// ============================================================
// Build individual model result card HTML
// ============================================================

function buildModelCardHTML(result, index) {
    if (result.error) {
        return `<div class="result-error" style="margin-bottom:1rem;"> ${result.error}</div>`;
    }

    const isAnomaly  = result.is_anomaly !== undefined ? result.is_anomaly : result.score > 0.5;
    const badgeClass = isAnomaly ? "badge-anomaly" : "badge-normal";
    const badgeText  = isAnomaly ? " ANOMALY" : " NORMAL";

    // ——— Risk Meter ———
    let riskMeterHTML = "";
    if (result.anomaly_index !== null && result.anomaly_index !== undefined) {
        const idx = result.anomaly_index;

        // Severity class and label
        let severity, verdictText;
        if (idx < 0.8) {
            severity    = "safe";
            verdictText = " AN TOÀN — Sản phẩm đạt chuẩn";
        } else if (idx < 1.0) {
            severity    = "warning";
            verdictText = " CẢNH BÁO — Gần ngưỡng, cần theo dõi";
        } else {
            severity    = "critical";
            verdictText = " NGUY HIỂM — Vượt ngưỡng, dừng dây chuyền";
        }

        // Fill bar: index=1.0 → 50%, index=2.0+ → 100%
        // This ensures scores well above threshold look clearly "full red"
        const fillPct = Math.min((idx / 2.0) * 100, 100).toFixed(1);
        const idxDisplay = idx.toFixed(2) + "x";

        riskMeterHTML = `
            <div class="risk-meter-wrap">
                <div class="risk-meter-header">
                    <span class="risk-meter-label"> Chỉ số rủi ro (Anomaly Index)</span>
                    <span class="risk-meter-value ${severity}">${idxDisplay}</span>
                </div>
                <div class="risk-meter-bar-track" style="margin-top: 1rem;">
                    <div class="risk-meter-bar-fill" id="rmFill_${Date.now()}_${index}"
                         style="width: 0%; clip-path: inset(0 0 0 0 round 20px);"
                         data-target="${fillPct}"></div>
                </div>
                <div class="risk-meter-scale">
                    <span>0.0 (Tối ưu)</span>
                    <span>1.0 (Ngưỡng giới hạn)</span>
                    <span>2.0+</span>
                </div>
                <span class="risk-meter-verdict ${severity}">${verdictText}</span>
            </div>
        `;
    }

    // ——— Images ———
    let imagesHTML = "";
    if (result.input_b64) {
        imagesHTML += `<div class="result-img-container">
            <img src="data:image/png;base64,${result.input_b64}" alt="Input">
            <p class="result-img-label">Ảnh gốc</p>
        </div>`;
    }
    if (result.recon_b64) {
        imagesHTML += `<div class="result-img-container">
            <img src="data:image/png;base64,${result.recon_b64}" alt="Reconstruction">
            <p class="result-img-label">Reconstruction</p>
        </div>`;
    }
    if (result.heatmap_b64) {
        imagesHTML += `<div class="result-img-container">
            <img src="data:image/png;base64,${result.heatmap_b64}" alt="Heatmap">
            <p class="result-img-label">Anomaly Heatmap</p>
        </div>`;
    }
    if (result.overlay_b64) {
        imagesHTML += `<div class="result-img-container">
            <img src="data:image/png;base64,${result.overlay_b64}" alt="Overlay">
            <p class="result-img-label">Overlay</p>
        </div>`;
    }

    const borderTop = index > 0 ? "border-top: 1px solid var(--border-glass); padding-top: 1.25rem; margin-top: 1.25rem;" : "";

    const html = `
        <div style="${borderTop}">
            <div class="result-header" style="margin-bottom:1rem;">
                <span class="result-model-name">${result.model}</span>
                <span class="result-badge ${badgeClass}">${badgeText}</span>
            </div>
            ${riskMeterHTML}
            <div class="result-score" style="margin-bottom:1rem;">
                <span class="score-label">Raw Score:</span>
                <span class="score-value" style="font-size:1.1rem;">${result.score.toFixed(6)}
                    <span style="font-size:0.7em;opacity:0.5;margin-left:8px;">
                        ${result.threshold !== undefined ? '(Ngưỡng Youden: ' + result.threshold.toFixed(4) + ')' : ''}
                    </span>
                </span>
            </div>
            <div class="result-images">${imagesHTML}</div>
        </div>
    `;
    return html;
}

// Animate all risk meter bars after DOM insertion
function animateRiskMeters() {
    document.querySelectorAll(".risk-meter-bar-fill[data-target]").forEach(el => {
        const target = parseFloat(el.dataset.target);
        requestAnimationFrame(() => {
            setTimeout(() => { el.style.width = target + "%"; }, 80);
        });
    });
}



// ============================================================
// Error display
// ============================================================

function showError(message) {
    resultsContainer.innerHTML = `
        <div class="result-card">
            <div class="result-error"> ${message}</div>
        </div>
    `;
    resultsPanel.style.display = "block";
}

// ============================================================
// Chatbot XAI Logic
// ============================================================

const chatToggleBtn = document.getElementById("chatToggleBtn");
const chatWindow    = document.getElementById("chatWindow");
const chatCloseBtn  = document.getElementById("chatCloseBtn");
const chatMessages  = document.getElementById("chatMessages");
const chatInput     = document.getElementById("chatInput");
const chatSendBtn   = document.getElementById("chatSendBtn");
const chatMode      = document.getElementById("chatMode");

let chatHistory = [];

chatMode.addEventListener("change", () => {
    if (chatMode.value === "generative") {
        appendMessage(" **Lưu ý:** Chế độ *AI Generative* gọi Gemini API, có thể giới hạn quota.", "bot");
    } else {
        appendMessage(" **Phân tích cứng (Offline).** Không gọi Internet, bảo mật tuyệt đối.", "bot");
    }
});

chatToggleBtn.addEventListener("click", () => {
    chatWindow.style.display = chatWindow.style.display === "none" ? "flex" : "none";
});

chatCloseBtn.addEventListener("click", () => {
    chatWindow.style.display = "none";
});

function parseMarkdown(text) {
    let html = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
    html = html.replace(/\*(.*?)\*/g, "<em>$1</em>");
    html = html.replace(/\n/g, "<br>");
    return html;
}

function appendMessage(text, sender) {
    const msgDiv = document.createElement("div");
    msgDiv.className = `chat-message ${sender}`;
    msgDiv.innerHTML = parseMarkdown(text);
    chatMessages.appendChild(msgDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    if (text !== "Chuyên gia đang gõ...") {
        chatHistory.push({ role: sender, content: text });
        if (chatHistory.length > 10) chatHistory.shift();
    }
}

async function sendChatMessage() {
    const msg = chatInput.value.trim();
    if (!msg) return;

    appendMessage(msg, "user");
    chatInput.value = "";

    const typingDiv = document.createElement("div");
    typingDiv.className = "typing-indicator";
    typingDiv.innerText = "Chuyên gia đang gõ...";
    typingDiv.style.display = "block";
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;

    try {
        const response = await fetch("/api/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                message: msg,
                mode:    chatMode.value,
                history: chatHistory.slice(0, -1),
                context: window.chatContext || {
                    category: "Chưa xác định", model: "Chưa phân tích",
                    score: 0, threshold: 0, is_anomaly: false,
                },
            }),
        });

        const data = await response.json();
        chatMessages.removeChild(typingDiv);

        if (data.error) {
            appendMessage(` Lỗi hệ thống: ${data.error}`, "bot");
        } else {
            appendMessage(data.response, "bot");
        }
    } catch (err) {
        chatMessages.removeChild(typingDiv);
        appendMessage(" Mất kết nối tới API Chuyên gia.", "bot");
    }
}

chatSendBtn.addEventListener("click", sendChatMessage);
chatInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendChatMessage();
});
