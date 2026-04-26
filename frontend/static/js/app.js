const state = {
  analytics: null,
  charts: {},
  chartObservers: new Map(),
  chartResizeFrame: null,
  chartResizeTimeout: null,
  typingFrames: new Map(),
  lastPrediction: null,
};

const byId = (id) => document.getElementById(id);

function escapeHtml(value = "") {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function toneFromRiskLabel(label = "") {
  const normalized = String(label || "").toLowerCase();
  if (normalized.includes("low")) return "low";
  if (normalized.includes("medium")) return "medium";
  if (normalized.includes("moderate")) return "medium";
  return "high";
}

function applyToneClass(element, tone) {
  if (!element) return;
  element.classList.remove("tone-low", "tone-medium", "tone-high");
  element.classList.add(`tone-${tone}`);
}

function formatSignedPercent(value) {
  const numericValue = Number(value || 0);
  if (!Number.isFinite(numericValue)) return "0%";
  return `${numericValue > 0 ? "+" : ""}${numericValue}%`;
}

function toneFromDirection(direction = "") {
  const normalized = String(direction || "").toLowerCase();
  if (normalized.includes("improv")) return "low";
  if (normalized.includes("declin")) return "high";
  return "medium";
}

function formatTrendStatus(direction = "steady") {
  const normalized = String(direction || "steady").toLowerCase();
  if (normalized.includes("improv")) return "Improving";
  if (normalized.includes("declin")) return "Declining";
  if (normalized.includes("steady") || normalized.includes("stable")) return "Stable";

  const label = normalized.replace(/_/g, " ").trim();
  return label ? label.charAt(0).toUpperCase() + label.slice(1) : "Stable";
}

function hasMeaningfulAnalyticsValue(value, emptyLabels = []) {
  const normalized = String(value || "").trim();
  if (!normalized) return false;
  return !emptyLabels.map((item) => String(item).toLowerCase()).includes(normalized.toLowerCase());
}

function buildAnalyticsInsightMessage(analytics) {
  const explicitMessage = String(analytics?.smart_insight_message || "").trim();
  if (explicitMessage) return explicitMessage;

  const mostCommonBias = String(analytics?.most_common_bias || "").trim();
  const mostUsedWord = String(analytics?.most_used_word || "").trim();
  const trendLabel = formatTrendStatus(analytics?.weekly_progress?.direction || "steady").toLowerCase();
  const riskLabel = String(analytics?.risk_level || "Medium Risk").trim();

  const hasBias = hasMeaningfulAnalyticsValue(mostCommonBias, ["No analyses yet"]);
  const hasTriggerWord = hasMeaningfulAnalyticsValue(mostUsedWord, ["No clear trigger yet"]);

  if (hasBias && hasTriggerWord) {
    return `Your thinking shows frequent ${mostCommonBias.toLowerCase()} patterns, and repeated words like "${mostUsedWord}" may be reinforcing them. Right now the overall trend looks ${trendLabel}.`;
  }

  if (hasBias) {
    return `Your thinking shows frequent ${mostCommonBias.toLowerCase()} patterns across recent entries. The current trend looks ${trendLabel}, so this is a good moment to keep challenging that pattern.`;
  }

  return `Your current analytics suggest a ${riskLabel.toLowerCase()} profile with a ${trendLabel} pattern. Keep analyzing thoughts to unlock more personalized guidance.`;
}

function setTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  localStorage.setItem("cb_theme", theme);
}

function showToast(message, kind = "error") {
  const toast = document.createElement("div");
  toast.className = `toast ${kind}`;
  toast.textContent = message;
  document.body.appendChild(toast);

  requestAnimationFrame(() => toast.classList.add("visible"));

  window.setTimeout(() => {
    toast.classList.remove("visible");
    window.setTimeout(() => toast.remove(), 240);
  }, 2800);
}

async function parseApiResponse(response) {
  let payload = {};
  try {
    payload = await response.json();
  } catch (_error) {
    payload = {};
  }

  if (response.status === 401) {
    window.location.href = "/login";
    throw new Error("Authentication required.");
  }

  if (!response.ok || !payload.success) {
    throw new Error(payload?.error?.message || payload?.message || "Request failed.");
  }

  return payload.data;
}

function readChartData(containerId) {
  const node = byId(containerId);
  if (!node) return [];
  try {
    return JSON.parse(node.dataset.chart || "[]");
  } catch (_error) {
    return [];
  }
}

function readJsonDataset(id, key) {
  const node = byId(id);
  if (!node) return [];
  try {
    return JSON.parse(node.dataset[key] || "[]");
  } catch (_error) {
    return [];
  }
}

function destroyChart(chartId) {
  if (!state.charts[chartId]) return;
  state.charts[chartId].destroy();
  delete state.charts[chartId];
}

function pulseClass(element, className) {
  if (!element) return;
  element.classList.remove(className);
  void element.offsetWidth;
  element.classList.add(className);
}

function resolveChartShell(containerId) {
  return byId(`${containerId}Shell`) || byId(containerId)?.parentElement || null;
}

function ensureChartCanvas(containerId) {
  const current = byId(containerId);
  if (current) return current;

  const shell = resolveChartShell(containerId);
  if (!shell) return null;

  const canvas = document.createElement("canvas");
  canvas.id = containerId;
  canvas.className = "chart-canvas";
  shell.appendChild(canvas);
  return canvas;
}

function queueChartResize() {
  if (state.chartResizeFrame) {
    window.cancelAnimationFrame(state.chartResizeFrame);
  }

  state.chartResizeFrame = window.requestAnimationFrame(() => {
    Object.values(state.charts).forEach((chart) => chart?.resize());
    state.chartResizeFrame = null;
  });
}

function queueDelayedChartResize(delay = 160) {
  if (state.chartResizeTimeout) {
    window.clearTimeout(state.chartResizeTimeout);
  }

  state.chartResizeTimeout = window.setTimeout(() => {
    state.chartResizeTimeout = null;
    queueChartResize();
  }, delay);
}

function observeChartShell(containerId, shell) {
  if (!shell || !window.ResizeObserver || state.chartObservers.has(containerId)) return;

  // Keep charts synced when grid cards expand after the initial paint.
  const observer = new window.ResizeObserver(() => {
    queueChartResize();
  });

  observer.observe(shell);
  state.chartObservers.set(containerId, observer);
}

function fallbackChart(containerId, items) {
  destroyChart(containerId);
  byId(containerId)?.remove();

  const shell = resolveChartShell(containerId);
  if (!shell) return;

  shell.querySelector(".chart-fallback")?.remove();
  shell.querySelector(".empty-state")?.remove();

  if (!items.length) {
    shell.insertAdjacentHTML("beforeend", '<div class="empty-state">No data available yet.</div>');
    return;
  }

  const maxValue = Math.max(...items.map((item) => Number(item.count ?? item.value ?? 0)), 1);
  shell.insertAdjacentHTML(
    "beforeend",
    `
      <div class="chart-fallback">
        ${items
          .map((item) => {
            const value = Number(item.count ?? item.value ?? 0);
            const width = Math.max(10, Math.round((value / maxValue) * 100));
            return `
              <article class="chart-fallback-item">
                <div class="chart-fallback-head">
                  <strong>${escapeHtml(item.label)}</strong>
                  <span>${escapeHtml(value)}</span>
                </div>
                <div class="chart-fallback-track"><span style="width:${width}%"></span></div>
              </article>
            `;
          })
          .join("")}
      </div>
    `
  );
}

function chartOptions() {
  const dark = document.documentElement.getAttribute("data-theme") === "dark";
  return {
    textColor: dark ? "#edf5f1" : "#17231f",
    gridColor: dark ? "rgba(141, 224, 206, 0.12)" : "rgba(17, 73, 63, 0.12)",
    palette: ["#34c8a0", "#65a8ff", "#f1aa58", "#ef6f7c", "#8f7bff", "#51d0e4", "#9ddf8f"],
  };
}

function chartTooltipMeaning(containerId, type) {
  const meanings = {
    biasChart: "How often this bias appeared in recent analyses.",
    insightsBiasChart: "How often this bias appeared in recent analyses.",
    sentimentChart: "How your recent entries split between negative and balanced language.",
    insightsSentimentChart: "How your recent entries split between negative and balanced language.",
    timelineChart: "How many times you used the analyzer in each recent period.",
    insightsTimelineChart: "How many times you used the analyzer in each recent period.",
    weeklyProgressChart: "Lower percentages suggest fewer negative thinking patterns over time.",
  };

  return meanings[containerId] || (type === "line" ? "Lower values suggest a healthier trend." : "");
}

function normalizeTriggerWords(analytics) {
  const rawWords = Array.isArray(analytics?.top_triggers) ? analytics.top_triggers : [];
  const words = rawWords
    .map((item) => {
      if (typeof item === "string") return item;
      return item?.label || item?.term || item?.word || "";
    })
    .map((item) => String(item || "").trim())
    .filter(Boolean);

  const uniqueWords = [...new Set(words)];
  if (uniqueWords.length) return uniqueWords.slice(0, 4);

  const fallback = String(analytics?.most_used_word || "").trim();
  if (fallback && fallback.toLowerCase() !== "no clear trigger yet") {
    return [fallback];
  }

  return [];
}

function renderTriggerWords(containerId, words) {
  const container = byId(containerId);
  if (!container) return;

  container.dataset.triggers = JSON.stringify(words || []);
  container.innerHTML = (words || []).length
    ? words
        .map((word) => `<span class="analytics-trigger-chip">${escapeHtml(word)}</span>`)
        .join("")
    : '<span class="analytics-trigger-chip analytics-trigger-chip-muted">No strong trigger words yet</span>';
}

function renderChart(containerId, items, type = "bar") {
  if (!resolveChartShell(containerId)) return;

  if (!window.Chart || !items.length) {
    fallbackChart(containerId, items);
    return;
  }

  const canvas = ensureChartCanvas(containerId);
  if (!canvas) return;
  const shell = resolveChartShell(containerId);
  observeChartShell(containerId, shell);

  canvas.dataset.chart = JSON.stringify(items);
  destroyChart(containerId);
  canvas.classList.remove("chart-canvas-live");
  void canvas.offsetWidth;
  canvas.classList.add("chart-canvas-live");
  pulseClass(shell, "chart-shell-live");

  const { textColor, gridColor, palette } = chartOptions();
  const labels = items.map((item) => item.label);
  const values = items.map((item) => Number(item.count ?? item.value ?? 0));
  const total = values.reduce((sum, value) => sum + value, 0);
  const dataset =
    type === "line"
      ? {
          data: values,
          label: "Value",
          borderColor: palette[0],
          backgroundColor: "rgba(80, 214, 173, 0.16)",
          fill: true,
          tension: 0.36,
          pointRadius: 4,
          pointHoverRadius: 6,
          pointBackgroundColor: palette[0],
          pointBorderColor: "#ffffff",
          pointBorderWidth: 2,
        }
      : {
          data: values,
          label: "Value",
          backgroundColor: palette,
          borderColor: type === "bar" ? palette[0] : "rgba(255,255,255,0.06)",
          borderWidth: type === "bar" ? 2 : 1,
          borderRadius: type === "bar" ? 14 : 0,
          hoverOffset: type === "doughnut" ? 12 : 0,
          hoverBorderWidth: type === "doughnut" ? 2 : 1,
        };

  state.charts[containerId] = new window.Chart(canvas, {
    type,
    data: {
      labels,
      datasets: [dataset],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 850,
        easing: "easeOutQuart",
      },
      interaction: {
        mode: "nearest",
        intersect: false,
      },
      plugins: {
        legend: {
          display: type === "doughnut",
          labels: {
            color: textColor,
            usePointStyle: true,
            padding: 18,
          },
        },
        tooltip: {
          backgroundColor: "rgba(9, 18, 17, 0.94)",
          titleColor: "#ffffff",
          bodyColor: "#e9f4ef",
          borderColor: "rgba(109, 228, 191, 0.18)",
          borderWidth: 1,
          displayColors: type !== "line",
          callbacks: {
            label(context) {
              const value = Number(context.raw || 0);
              if (type === "doughnut" && total > 0) {
                const percentage = Math.round((value / total) * 100);
                return `${context.label}: ${value} (${percentage}%)`;
              }
              if (type === "line") {
                return `${context.label}: ${value}% negative`;
              }
              return `${context.label}: ${value}`;
            },
            afterLabel() {
              return chartTooltipMeaning(containerId, type);
            },
          },
        },
      },
      scales:
        type === "bar" || type === "line"
          ? {
              x: {
                ticks: { color: textColor },
                grid: { color: gridColor, drawBorder: false },
              },
              y: {
                beginAtZero: true,
                suggestedMax: type === "line" ? 100 : undefined,
                ticks: { color: textColor },
                grid: { color: gridColor, drawBorder: false },
              },
            }
          : {},
    },
  });
}

function typeText(element, text) {
  if (!element) return;

  const existingFrame = state.typingFrames.get(element);
  if (existingFrame) {
    window.cancelAnimationFrame(existingFrame);
    state.typingFrames.delete(element);
  }

  const value = String(text || "");
  if (!value) {
    element.textContent = "";
    return;
  }

  let index = 0;
  element.textContent = "";

  const step = () => {
    index += Math.max(1, Math.ceil(value.length / 38));
    element.textContent = value.slice(0, index);
    if (index < value.length) {
      const nextFrame = window.requestAnimationFrame(step);
      state.typingFrames.set(element, nextFrame);
    }
  };

  const firstFrame = window.requestAnimationFrame(step);
  state.typingFrames.set(element, firstFrame);
}

function setInstantText(element, text) {
  if (!element) return;

  const existingFrame = state.typingFrames.get(element);
  if (existingFrame) {
    window.cancelAnimationFrame(existingFrame);
    state.typingFrames.delete(element);
  }

  element.textContent = String(text || "");
}

function collectHighlightRanges(text, terms) {
  const lower = String(text || "").toLowerCase();
  const uniqueTerms = [...new Set((terms || []).map((term) => String(term).toLowerCase()).filter(Boolean))].sort(
    (left, right) => right.length - left.length
  );
  const ranges = [];

  uniqueTerms.forEach((term) => {
    let fromIndex = 0;
    while (fromIndex < lower.length) {
      const start = lower.indexOf(term, fromIndex);
      if (start === -1) break;
      const end = start + term.length;
      const overlaps = ranges.some((range) => start < range.end && end > range.start);
      if (!overlaps) {
        ranges.push({ start, end });
      }
      fromIndex = start + term.length;
    }
  });

  return ranges.sort((left, right) => left.start - right.start);
}

function buildHighlightedHtml(text, terms) {
  const safeText = String(text || "");
  const ranges = collectHighlightRanges(safeText, terms);
  if (!ranges.length) return escapeHtml(safeText);

  const parts = [];
  let cursor = 0;

  ranges.forEach((range) => {
    if (cursor < range.start) {
      parts.push(escapeHtml(safeText.slice(cursor, range.start)));
    }
    parts.push(`<mark>${escapeHtml(safeText.slice(range.start, range.end))}</mark>`);
    cursor = range.end;
  });

  if (cursor < safeText.length) {
    parts.push(escapeHtml(safeText.slice(cursor)));
  }

  return parts.join("");
}

function renderHighlightedText(element, text, terms) {
  if (!element) return;
  element.dataset.text = text || "";
  element.dataset.terms = JSON.stringify(terms || []);
  element.innerHTML = buildHighlightedHtml(text || "", terms || []);
}

function renderAlternatives(items) {
  const container = byId("alternatives");
  if (!container) return;

  container.innerHTML = items.length
    ? items
        .map(
          (item) => `
            <article class="detail-card">
              <strong>${escapeHtml(item.bias)}</strong>
              <span>${Math.round(Number(item.confidence) * 100)}% confidence</span>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">Alternative predictions will appear after analysis.</div>';
}

function renderMatchedTerms(items) {
  const container = byId("matchedTerms");
  if (!container) return;

  container.innerHTML = items.length
    ? items.map((term) => `<span class="pill">${escapeHtml(term)}</span>`).join("")
    : '<span class="pill">No trigger terms yet</span>';
}

function renderHighlightDetails(items) {
  const container = byId("triggerExplanations");
  if (!container) return;

  container.innerHTML = items.length
    ? items
        .map(
          (item) => `
            <article class="trigger-note">
              <strong>${escapeHtml(item.term)}</strong>
              <span>${escapeHtml(item.explanation)}</span>
            </article>
          `
        )
        .join("")
    : '<div class="empty-state">Trigger explanations will appear after analysis.</div>';
}

function confidenceTone(confidenceValue) {
  const value = Number(confidenceValue || 0);
  if (value >= 0.8) return "#2fca8f";
  if (value >= 0.55) return "#f1aa58";
  return "#ef6f7c";
}

function renderPrediction(prediction) {
  const panel = byId("resultPanel");
  if (!panel) return;
  state.lastPrediction = prediction;

  panel.classList.remove("is-empty");
  panel.classList.remove("is-loading");
  panel.classList.add("is-ready");
  panel.classList.remove("is-live");
  void panel.offsetWidth;
  panel.classList.add("is-live");

  byId("biasBadge").textContent = prediction.bias;
  byId("confidenceBadge").textContent = prediction.confidence_percent;
  byId("biasValue").textContent = prediction.bias;
  byId("thinkingScoreValue").textContent = `${prediction.thinking_score ?? "--"} / 100`;
  byId("thinkingBandValue").textContent = prediction.thinking_band || "-";
  byId("sentimentValue").textContent = prediction.sentiment_bucket;
  byId("confidenceMeterLabel").textContent = prediction.confidence_percent;
  const thinkingTone = prediction.thinking_tone || toneFromRiskLabel(prediction.thinking_band);
  applyToneClass(byId("thinkingScoreCard"), thinkingTone);
  applyToneClass(byId("thinkingBandCard"), thinkingTone);
  applyToneClass(byId("thinkingBandValue"), thinkingTone);

  const progress = byId("confidenceProgress");
  if (progress) {
    progress.style.width = `${Math.max(4, Math.round(Number(prediction.confidence) * 100))}%`;
    progress.style.background = `linear-gradient(90deg, ${confidenceTone(prediction.confidence)}, #65a8ff)`;
  }

  typeText(byId("explanationValue"), prediction.explanation || "");
  typeText(
    byId("balancedThoughtValue"),
    prediction.balanced_thought || "A grounded reframe will appear here after analysis."
  );
  typeText(byId("suggestionValue"), prediction.suggestion || "A next-step suggestion will appear here after analysis.");
  renderMatchedTerms(prediction.highlight_terms || prediction.matched_terms || []);
  renderHighlightDetails(prediction.highlight_details || []);
  renderAlternatives(prediction.alternative_biases || []);
  renderHighlightedText(byId("highlightedText"), prediction.text || "", prediction.highlight_terms || []);
  byId("continueChatBtn")?.removeAttribute("disabled");

  panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

function resetResultUI({ loading = false } = {}) {
  const panel = byId("resultPanel");
  if (!panel) return;
  state.lastPrediction = null;

  panel.classList.remove("is-ready", "is-live");
  panel.classList.add("is-empty");
  panel.classList.toggle("is-loading", loading);

  setInstantText(byId("biasBadge"), loading ? "Analyzing..." : "Awaiting analysis");
  setInstantText(byId("confidenceBadge"), "0%");
  setInstantText(byId("biasValue"), "-");
  setInstantText(byId("thinkingScoreValue"), "-- / 100");
  setInstantText(byId("thinkingBandValue"), "-");
  setInstantText(byId("sentimentValue"), "-");
  setInstantText(byId("confidenceMeterLabel"), "0%");
  setInstantText(
    byId("explanationValue"),
    loading ? "Scanning language patterns and model confidence..." : "Run an analysis to get a model explanation."
  );
  setInstantText(
    byId("balancedThoughtValue"),
    loading ? "Preparing a balanced rewrite..." : "A grounded reframe will appear here after analysis."
  );
  setInstantText(
    byId("suggestionValue"),
    loading ? "Building a helpful next step..." : "A next-step suggestion will appear here after analysis."
  );
  applyToneClass(byId("thinkingScoreCard"), "medium");
  applyToneClass(byId("thinkingBandCard"), "medium");
  applyToneClass(byId("thinkingBandValue"), "medium");

  const progress = byId("confidenceProgress");
  if (progress) {
    progress.style.width = "0%";
    progress.style.background = "linear-gradient(90deg, #ef6f7c, #65a8ff)";
  }

  renderMatchedTerms(loading ? ["Scanning cues"] : []);
  renderHighlightDetails([]);
  renderAlternatives([]);
  renderHighlightedText(
    byId("highlightedText"),
    loading ? byId("textInput")?.value.trim() || "Analyzing your thought..." : "Analyze a thought to see highlighted trigger terms.",
    []
  );
  byId("continueChatBtn")?.setAttribute("disabled", "disabled");
}

function renderSavedItems(items) {
  const container = byId("savedList");
  if (!container) return;

  container.innerHTML = items.length
    ? items
        .map(
          (item) => `
            <button class="saved-item" type="button" data-text="${escapeHtml(item.text)}">
              <strong>${escapeHtml(item.timestamp)}</strong>
              <span>${escapeHtml(item.text)}</span>
            </button>
          `
        )
        .join("")
    : '<div class="empty-state">Saved drafts will appear here once you store them.</div>';

  const savedCount = byId("savedCountValue");
  if (savedCount) savedCount.textContent = String(items.length);
}

function renderHistoryItems(items) {
  const container = byId("historyList");
  if (!container) return;

  container.innerHTML = items.length
    ? items
        .map((item) => {
          if (item.latest_text) {
            const tone = toneFromRiskLabel(item.thinking_band || "");
            return `
              <button class="history-item timeline-item" type="button" data-text="${escapeHtml(item.latest_text)}">
                <div class="history-item-head">
                  <strong>${escapeHtml(item.bias)}</strong>
                  <span class="history-count">&times;${escapeHtml(item.count)}</span>
                </div>
                <span>${escapeHtml(item.latest_text)}</span>
                <div class="history-item-meta">
                  <em>Latest | ${escapeHtml(item.latest_timestamp || "")}</em>
                  <span class="history-chip tone-${tone}">${escapeHtml(item.thinking_band || "Medium Risk")}</span>
                </div>
              </button>
            `;
          }

          return `
            <button class="history-item timeline-item" type="button" data-text="${escapeHtml(item.text)}">
              <strong>${escapeHtml(item.bias)}</strong>
              <span>${escapeHtml(item.text)}</span>
              <em>${escapeHtml(item.confidence_percent)} | ${escapeHtml(item.timestamp)}</em>
            </button>
          `;
        })
        .join("")
    : '<div class="empty-state">Your thought history will appear here after the first analysis.</div>';
}

function renderChatMessages(messages) {
  const container = byId("chatMessages");
  if (!container) return;

  container.innerHTML = messages.length
    ? messages
        .map((message) => {
          const meta =
            message.role === "assistant" && message.bias
              ? `
                  <div class="chat-meta">
                    <span>${escapeHtml(message.bias)}</span>
                    <span>${escapeHtml(message.confidence_percent || "")}</span>
                  </div>
                `
              : "";

          return `
            <article class="chat-message ${escapeHtml(message.role)}">
              <span class="chat-role">${message.role === "assistant" ? "AI Guide" : "You"}</span>
              <p>${escapeHtml(message.content)}</p>
              ${meta}
            </article>
          `;
        })
        .join("")
    : '<div class="empty-state">Start the conversation with a thought like "I always mess things up."</div>';

  container.scrollTop = container.scrollHeight;

  const chatCount = byId("chatCountValue");
  if (chatCount) chatCount.textContent = String(messages.length);
}

function findUsageValue(analytics, label) {
  return analytics?.usage_statistics?.find((item) => item.label === label)?.value ?? null;
}

function updateAnalytics(analytics) {
  if (!analytics) return;

  state.analytics = analytics;

  const cognitiveScoreValue = Math.max(0, Math.min(100, Math.round(Number(analytics.avg_thinking_score || 0))));
  const riskTone = toneFromRiskLabel(analytics.risk_level);
  const trendDirection = analytics.weekly_progress?.direction || "steady";
  const trendTone = toneFromDirection(trendDirection);
  const trendLabel = formatTrendStatus(trendDirection);
  const triggerWords = normalizeTriggerWords(analytics);
  const insightMessage = buildAnalyticsInsightMessage(analytics);

  renderChart("biasChart", analytics.bias_distribution || [], "doughnut");
  renderChart("insightsBiasChart", analytics.bias_distribution || [], "doughnut");
  renderChart("sentimentChart", analytics.negative_vs_balanced || analytics.sentiment_distribution || [], "doughnut");
  renderChart("insightsSentimentChart", analytics.negative_vs_balanced || analytics.sentiment_distribution || [], "doughnut");
  renderChart("timelineChart", analytics.timeline || [], "bar");
  renderChart("insightsTimelineChart", analytics.timeline || [], "bar");
  renderChart("weeklyProgressChart", analytics.weekly_progress?.series || [], "line");
  queueChartResize();
  queueDelayedChartResize();

  const analysisCount = byId("analysisCountValue");
  if (analysisCount) analysisCount.textContent = String(analytics.history_count ?? 0);

  const commonBias = byId("commonBiasValue");
  if (commonBias) commonBias.textContent = analytics.most_common_bias || "No analyses yet";

  const analyticsCommonBias = byId("analyticsMostCommonBiasValue");
  if (analyticsCommonBias) analyticsCommonBias.textContent = analytics.most_common_bias || "No analyses yet";

  const sidebarCommonBias = byId("sidebarCommonBias");
  if (sidebarCommonBias) sidebarCommonBias.textContent = analytics.most_common_bias || "No analyses yet";

  const balancedRatio = byId("balancedRatioValue");
  if (balancedRatio) balancedRatio.textContent = `${analytics.balanced_ratio || 0}%`;

  const avgConfidence = byId("avgConfidenceValue");
  if (avgConfidence) avgConfidence.textContent = `${analytics.avg_confidence || 0}%`;

  const trendSummary = byId("trendSummaryValue");
  if (trendSummary) trendSummary.textContent = analytics.trend_summary || "";

  const mostUsedWord = byId("mostUsedWordValue");
  if (mostUsedWord) mostUsedWord.textContent = analytics.most_used_word || "No clear trigger yet";

  const analyticsTriggerWord = byId("analyticsTriggerWordValue");
  if (analyticsTriggerWord) analyticsTriggerWord.textContent = analytics.most_used_word || "No clear trigger yet";

  const riskLevel = byId("riskLevelValue");
  if (riskLevel) {
    riskLevel.textContent = analytics.risk_level || "Low Risk";
    applyToneClass(riskLevel, riskTone);
  }

  const riskScore = byId("riskScoreValue");
  if (riskScore) riskScore.textContent = `${analytics.risk_score || 0} / 100`;

  const analyticsRiskScore = byId("analyticsRiskScoreBadgeValue");
  if (analyticsRiskScore) analyticsRiskScore.textContent = `${analytics.risk_score || 0} / 100`;

  const riskSummary = byId("riskSummaryValue");
  if (riskSummary) riskSummary.textContent = analytics.risk_summary || "";

  const analyticsRiskSummary = byId("analyticsRiskSummaryInline");
  if (analyticsRiskSummary) analyticsRiskSummary.textContent = analytics.risk_summary || "";

  const smartInsightMessage = byId("smartInsightMessageValue");
  if (smartInsightMessage) setInstantText(smartInsightMessage, insightMessage);

  const analyticsInsightMessage = byId("analyticsInsightMessageValue");
  if (analyticsInsightMessage) typeText(analyticsInsightMessage, insightMessage);

  const riskProgress = byId("riskProgress");
  if (riskProgress) {
    riskProgress.style.width = `${Math.max(6, Number(analytics.risk_score || 0))}%`;
    const gradients = {
      low: "linear-gradient(90deg, #2fca8f, #65a8ff)",
      medium: "linear-gradient(90deg, #f1aa58, #65a8ff)",
      high: "linear-gradient(90deg, #ef6f7c, #f1aa58)",
    };
    riskProgress.style.background = gradients[riskTone] || gradients.medium;
  }

  const analyticsCognitiveScore = byId("analyticsCognitiveScoreValue");
  if (analyticsCognitiveScore) analyticsCognitiveScore.textContent = `${cognitiveScoreValue} / 100`;

  const analyticsCognitiveProgress = byId("analyticsCognitiveProgress");
  if (analyticsCognitiveProgress) analyticsCognitiveProgress.style.width = `${Math.max(8, cognitiveScoreValue)}%`;

  const analyticsRiskLevel = byId("analyticsRiskLevelBadge");
  if (analyticsRiskLevel) {
    analyticsRiskLevel.textContent = analytics.risk_level || "Low Risk";
    applyToneClass(analyticsRiskLevel, riskTone);
  }

  const analyticsTrendStatus = byId("analyticsTrendStatusValue");
  if (analyticsTrendStatus) {
    analyticsTrendStatus.textContent = trendLabel;
    applyToneClass(analyticsTrendStatus, trendTone);
  }

  const analyticsTrendNarrative = byId("analyticsTrendNarrativeValue");
  if (analyticsTrendNarrative) analyticsTrendNarrative.textContent = analytics.trend_summary || "";

  const analyticsTrendSummary = byId("analyticsTrendSummaryValue");
  if (analyticsTrendSummary) analyticsTrendSummary.textContent = analytics.trend_summary || "";

  renderTriggerWords("analyticsTriggerWordsList", triggerWords);

  if (Array.isArray(analytics.grouped_history) && (analytics.grouped_history.length || Number(analytics.history_count || 0) === 0)) {
    renderHistoryItems(analytics.grouped_history);
  }

  const weeklyImprovement = byId("weeklyImprovementValue");
  if (weeklyImprovement) {
    const improvementValue = Number(analytics.weekly_progress?.improvement || 0);
    weeklyImprovement.dataset.value = String(improvementValue);
    weeklyImprovement.dataset.direction = trendDirection;
    weeklyImprovement.textContent = formatSignedPercent(improvementValue);
    applyToneClass(weeklyImprovement, trendTone);
  }

  const weeklyDirection = byId("weeklyDirectionValue");
  if (weeklyDirection) {
    weeklyDirection.textContent = trendLabel;
    applyToneClass(weeklyDirection, trendTone);
  }

  const weeklySummary = byId("weeklyProgressSummaryValue");
  if (weeklySummary) weeklySummary.textContent = analytics.weekly_progress?.summary || "";

  const analyticsWeeklySummary = byId("analyticsWeeklySummaryValue");
  if (analyticsWeeklySummary) analyticsWeeklySummary.textContent = analytics.weekly_progress?.summary || "";

  const savedCount = byId("savedCountValue");
  if (savedCount) {
    const savedValue = findUsageValue(analytics, "Saved Drafts");
    if (savedValue !== null) savedCount.textContent = String(savedValue);
  }

  const chatCount = byId("chatCountValue");
  if (chatCount) {
    const chatValue = findUsageValue(analytics, "Chat Turns");
    if (chatValue !== null) chatCount.textContent = String(chatValue);
  }
}

function setButtonLoading(buttonId, isLoading, loadingText, defaultText) {
  const button = byId(buttonId);
  if (!button) return;
  button.classList.toggle("loading", isLoading);
  button.textContent = isLoading ? loadingText : defaultText;
}

function setAnalyzeLoading(isLoading) {
  byId("loader")?.classList.toggle("active", isLoading);
  setButtonLoading("analyzeBtn", isLoading, "Analyzing...", "Analyze Text");
}

async function analyzeText() {
  const text = byId("textInput")?.value.trim();
  if (!text) {
    showToast("Please enter a thought before analyzing.");
    return;
  }

  resetResultUI({ loading: true });
  setAnalyzeLoading(true);

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await parseApiResponse(response);
    renderPrediction(data.prediction);
    renderHistoryItems(data.history || []);
    renderSavedItems(data.saved_items || []);
    if (Array.isArray(data.chat_messages) && data.chat_messages.length) {
      renderChatMessages(data.chat_messages);
    }
    updateAnalytics(data.analytics || null);
  } catch (error) {
    showToast(error.message || "Prediction failed.");
  } finally {
    setAnalyzeLoading(false);
  }
}

async function saveText() {
  const text = byId("textInput")?.value.trim();
  if (!text) {
    showToast("Write something before saving a draft.");
    return;
  }

  try {
    const response = await fetch("/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await parseApiResponse(response);
    renderSavedItems(data.saved_items || []);
    updateAnalytics(data.analytics || null);
    showToast("Draft saved successfully.", "success");
  } catch (error) {
    showToast(error.message || "Save failed.");
  }
}

async function sendChat(event) {
  event.preventDefault();
  const input = byId("chatInput");
  const text = input?.value.trim();
  if (!text) {
    showToast("Enter a thought before sending it to the assistant.");
    return;
  }

  setButtonLoading("chatSendBtn", true, "Sending...", "Send");

  const chatMessages = byId("chatMessages");
  if (chatMessages) {
    const emptyState = chatMessages.querySelector(".empty-state");
    if (emptyState) emptyState.remove();

    chatMessages.insertAdjacentHTML('beforeend', `
      <article class="chat-message user">
        <span class="chat-role">You</span>
        <p>${escapeHtml(text)}</p>
      </article>
      <article class="chat-message assistant typing-indicator-msg" id="typingIndicator">
        <span class="chat-role">AI Guide</span>
        <div class="typing-dots"><span></span><span></span><span></span></div>
      </article>
    `);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await parseApiResponse(response);
    renderChatMessages(data.messages || []);
    updateAnalytics(data.analytics || null);
    if (input) input.value = "";
  } catch (error) {
    const typingIndicator = byId("typingIndicator");
    if (typingIndicator) typingIndicator.remove();
    showToast(error.message || "Assistant request failed.");
  } finally {
    setButtonLoading("chatSendBtn", false, "Sending...", "Send");
  }
}

function randomExampleFrom(button) {
  const textarea = byId("textInput");
  if (!textarea) return;

  let options = [];
  try {
    options = JSON.parse(button.dataset.examples || "[]");
  } catch (_error) {
    options = [];
  }

  if (!options.length) return;

  const nextValue = options[Math.floor(Math.random() * options.length)];
  textarea.value = nextValue;
  textarea.focus();
}

function setupReveals() {
  document.body.classList.add("is-ready");

  const nodes = document.querySelectorAll(".reveal");
  if (!("IntersectionObserver" in window)) {
    nodes.forEach((node) => node.classList.add("is-visible"));
    return;
  }

  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("is-visible");
          observer.unobserve(entry.target);
        }
      });
    },
    { threshold: 0.14 }
  );

  nodes.forEach((node) => observer.observe(node));
}

function syncAuthShellHeight() {
  const card = byId("authCard");
  const shell = card?.querySelector(".auth-form-shell");
  const mode = card?.dataset.authMode || "login";
  const activeForm = card?.querySelector(`[data-auth-form="${mode}"]`);
  if (!shell || !activeForm) return;

  const nextHeight = Math.ceil(activeForm.scrollHeight);
  shell.style.height = `${Math.max(320, nextHeight)}px`;
}

function setAuthMode(mode) {
  const card = byId("authCard");
  if (!card) return;

  const nextMode = mode === "signup" ? "signup" : "login";
  card.dataset.authMode = nextMode;

  document.querySelectorAll("[data-auth-tab]").forEach((button) => {
    button.classList.toggle("active", button.dataset.authTab === nextMode);
  });

  const params = new URLSearchParams(window.location.search);
  const nextTarget = params.get("next");
  const nextUrl = new URL(nextMode === "signup" ? "/login?mode=signup" : "/login", window.location.origin);
  if (nextTarget) {
    nextUrl.searchParams.set("next", nextTarget);
  }
  window.history.replaceState({}, "", nextUrl);

  window.requestAnimationFrame(() => {
    syncAuthShellHeight();
  });
}

function initAuthSwitching() {
  const card = byId("authCard");
  if (!card) return;

  setAuthMode(card.dataset.authMode || "login");

  document.querySelectorAll("[data-auth-tab]").forEach((button) => {
    button.addEventListener("click", () => setAuthMode(button.dataset.authTab || "login"));
  });

  document.querySelectorAll("[data-auth-toggle]").forEach((button) => {
    button.addEventListener("click", () => setAuthMode(button.dataset.authToggle || "login"));
  });

  syncAuthShellHeight();
  window.addEventListener("resize", syncAuthShellHeight);
  window.setTimeout(syncAuthShellHeight, 120);
}

function bindStaticInteractions() {
  document.querySelectorAll("[data-examples]").forEach((button) => {
    button.addEventListener("click", () => randomExampleFrom(button));
  });

  document.addEventListener("click", (event) => {
    const fillable = event.target.closest("[data-text]");
    const textInput = byId("textInput");
    if (fillable?.dataset.text && textInput && fillable.id !== "highlightedText") {
      textInput.value = fillable.dataset.text;
      textInput.focus();
    }
  });
}

function continueInChat() {
  const sourceText = byId("highlightedText")?.dataset.text || byId("textInput")?.value.trim() || "";
  const chatInput = byId("chatInput");
  if (!chatInput) return;

  if (state.lastPrediction?.bias) {
    chatInput.value = `Help me continue working through this thought: "${sourceText}". The analyzer flagged ${state.lastPrediction.bias}. Balanced thought: "${state.lastPrediction.balanced_thought}".`;
  } else {
    chatInput.value = sourceText || "Help me continue working through this thought.";
  }
  chatInput.focus();
  chatInput.scrollIntoView({ behavior: "smooth", block: "center" });
}

async function copyAnalysis() {
  if (!state.lastPrediction) {
    showToast("No analysis available to copy.");
    return;
  }
  const p = state.lastPrediction;
  const report = `Cognitive Bias Analysis
-------------------------
Original Thought: "${p.text || ""}"
Detected Bias: ${p.bias} (${p.confidence_percent} confidence)
Thinking Score: ${p.thinking_score || "--"} / 100 (${p.thinking_band || "-"})

Explanation:
${p.explanation}

Balanced Reframe:
${p.balanced_thought || "N/A"}

Suggested Action:
${p.suggestion}
`;
  try {
    await navigator.clipboard.writeText(report);
    showToast("Analysis copied to clipboard!", "success");
  } catch (err) {
    showToast("Failed to copy analysis.");
  }
}

function renderStaticHighlights() {
  document.querySelectorAll("[data-text][data-terms]").forEach((element) => {
    if (element.id === "highlightedText" || element.dataset.highlightPreview === "true") {
      let terms = [];
      try {
        terms = JSON.parse(element.dataset.terms || "[]");
      } catch (_error) {
        terms = [];
      }
      renderHighlightedText(element, element.dataset.text || "", terms);
    }
  });
}

function bootInitialState() {
  const weeklySeries = readChartData("weeklyProgressChart");
  const analytics = {
    bias_distribution: readChartData("biasChart").length ? readChartData("biasChart") : readChartData("insightsBiasChart"),
    negative_vs_balanced: readChartData("sentimentChart").length
      ? readChartData("sentimentChart")
      : readChartData("insightsSentimentChart"),
    timeline: readChartData("timelineChart").length ? readChartData("timelineChart") : readChartData("insightsTimelineChart"),
    history_count: Number(byId("analysisCountValue")?.textContent || 0),
    avg_thinking_score: Number((byId("analyticsCognitiveScoreValue")?.textContent || "0").split("/")[0].trim()) || 0,
    balanced_ratio: Number((byId("balancedRatioValue")?.textContent || "0").replace("%", "")) || 0,
    avg_confidence: Number((byId("avgConfidenceValue")?.textContent || "0").replace("%", "")) || 0,
    most_common_bias: byId("commonBiasValue")?.textContent || "No analyses yet",
    most_used_word: byId("mostUsedWordValue")?.textContent || "No clear trigger yet",
    top_triggers: readJsonDataset("analyticsTriggerWordsList", "triggers"),
    risk_level: byId("riskLevelValue")?.textContent || "Low Risk",
    risk_score: Number((byId("riskScoreValue")?.textContent || "0").split("/")[0].trim()) || 0,
    risk_summary: byId("riskSummaryValue")?.textContent || "",
    smart_insight_message: byId("smartInsightMessageValue")?.textContent || "",
    trend_summary: byId("trendSummaryValue")?.textContent || "",
    grouped_history: [],
    weekly_progress: {
      series: weeklySeries,
      improvement: Number(byId("weeklyImprovementValue")?.dataset.value || 0),
      direction: byId("weeklyImprovementValue")?.dataset.direction || "steady",
      summary: byId("weeklyProgressSummaryValue")?.textContent || "",
    },
    usage_statistics: [
      { label: "Saved Drafts", value: Number(byId("savedCountValue")?.textContent || 0) },
      { label: "Chat Turns", value: Number(byId("chatCountValue")?.textContent || 0) },
    ],
  };

  updateAnalytics(analytics);
}

document.addEventListener("DOMContentLoaded", () => {
  const theme = localStorage.getItem("cb_theme") || "dark";
  setTheme(theme);

  byId("themeToggle")?.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") === "dark" ? "dark" : "light";
    setTheme(current === "dark" ? "light" : "dark");
    if (state.analytics) {
      updateAnalytics(state.analytics);
    }
  });

  setupReveals();
  initAuthSwitching();
  bindStaticInteractions();
  renderStaticHighlights();
  bootInitialState();
  window.addEventListener("load", () => queueDelayedChartResize(0));
  window.addEventListener("resize", queueChartResize);

  if (document.body.dataset.page === "dashboard") {
    resetResultUI();
  }

  byId("analyzeBtn")?.addEventListener("click", analyzeText);
  byId("saveBtn")?.addEventListener("click", saveText);
  byId("clearBtn")?.addEventListener("click", () => {
    const textarea = byId("textInput");
    if (!textarea) return;
    textarea.value = "";
    textarea.focus();
  });
  byId("continueChatBtn")?.addEventListener("click", continueInChat);
  byId("copyAnalysisBtn")?.addEventListener("click", copyAnalysis);
  byId("chatForm")?.addEventListener("submit", sendChat);
});
