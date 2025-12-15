class StatusNotifier {
  constructor() {
    this.toast = null;
    this.hideTimer = null;
    this.styleInjected = false;
  }

  ensureStyle() {
    if (this.styleInjected) return;
    const style = document.createElement("style");
    style.textContent = `
      .sem-highlight-toast {
        position: fixed;
        right: 16px;
        bottom: 16px;
        padding: 10px 12px;
        border-radius: 6px;
        background: rgba(32, 34, 37, 0.92);
        color: #f7f7f7;
        font-size: 13px;
        line-height: 1.4;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);
        opacity: 0;
        transform: translateY(8px);
        transition: opacity 160ms ease, transform 160ms ease;
        pointer-events: none;
        z-index: 2147483647;
      }
      .sem-highlight-toast.visible {
        opacity: 1;
        transform: translateY(0);
      }
      .sem-highlight-toast.info { background: rgba(36, 107, 255, 0.94); }
      .sem-highlight-toast.success { background: rgba(0, 148, 82, 0.94); }
      .sem-highlight-toast.error { background: rgba(187, 41, 55, 0.94); }
    `;
    document.head.appendChild(style);
    this.styleInjected = true;
  }

  ensureToast() {
    if (this.toast) return;
    this.toast = document.createElement("div");
    this.toast.className = "sem-highlight-toast";
    document.body.appendChild(this.toast);
  }

  show(message, type = "info", duration = 2500) {
    this.ensureStyle();
    this.ensureToast();

    this.toast.textContent = message;
    this.toast.classList.remove("info", "success", "error", "visible");
    this.toast.classList.add(type);

    // Allow transition to apply when toggling visibility.
    requestAnimationFrame(() => {
      this.toast.classList.add("visible");
    });

    clearTimeout(this.hideTimer);
    this.hideTimer = setTimeout(() => this.hide(), duration);
  }

  hide() {
    if (!this.toast) return;
    this.toast.classList.remove("visible");
  }
}

const DEFAULT_SETTINGS = {
  enabled: true,
  intensity: 0.35
};

const MIN_WORDS = 15;
const MAX_TARGETS = 3;
const MARK_CLASS = "sem-highlight-mark";
const ATTR_HIGHLIGHTED = "data-sem-highlighted";
const DATA_HIGHLIGHTED = "semHighlighted";
const DATA_PROCESSING = "semHighlightProcessing";

function initialize() {
  loadSettings().then(() => {
    attachStorageListener();
    scanAndHighlight();
    startObservation();
    updateDebugState({ state: "idle", activeAnalyses: 0 });
    warmupModel();
  });
}

function loadSettings() {
  return new Promise(resolve => {
    chrome.storage.sync.get(DEFAULT_SETTINGS, stored => {
      settings = { ...DEFAULT_SETTINGS, ...stored };
      resolve();
    });
  });
}

function attachStorageListener() {
  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "sync") return;

    if (changes.enabled) {
      settings.enabled = changes.enabled.newValue;
    }

    if (changes.intensity) {
      settings.intensity = changes.intensity.newValue;
    }

    if (!settings.enabled) {
      highlightManager.clearAll();
    } else {
      scanAndHighlight(true);
    }
  });
}

function startObservation() {
  if (observer || !document.body) return;

  observer = new MutationObserver(debounce((mutations) => {
    // Ignore mutations that are only about our highlight marks
    const hasRelevantMutation = mutations.some(mutation => {
      // Ignore if only mark elements changed
      if (mutation.type === 'childList') {
        const relevantNodes = Array.from(mutation.addedNodes).concat(Array.from(mutation.removedNodes))
          .filter(node => {
            if (node.nodeType === Node.ELEMENT_NODE) {
              return node.tagName !== 'MARK' || !node.classList.contains(MARK_CLASS);
            }
            return true;
          });
        return relevantNodes.length > 0;
      }
      return true;
    });
    
    if (hasRelevantMutation) {
      scanAndHighlight();
    }
  }, 800));
  observer.observe(document.body, { childList: true, subtree: true });
}

function scanAndHighlight(forceRefresh = false) {
  if (!settings.enabled) {
    return;
  }

  try {
    const { targets, maxWords } = findTargets(forceRefresh);
    updateDebugState({
      lastScanCandidates: targets.length,
      lastScanSelected: Math.min(targets.length, MAX_TARGETS),
      lastScanMaxWords: maxWords,
      state: targets.length ? "queued" : "idle"
    });

    console.info(
      "[semantic-highlight] scan candidates:",
      targets.length,
      "selected:",
      Math.min(targets.length, MAX_TARGETS),
      "max words:",
      maxWords
    );

    targets.slice(0, MAX_TARGETS).forEach(element => analyzeElement(element));
  } catch (error) {
    console.warn("semantic-highlight: scan failed", error);
    updateDebugState({ state: "error", lastError: String(error) });
  }
}

function findTargets(forceRefresh) {
  const selectors = ["p"];
  const candidates = Array.from(
    document.querySelectorAll(selectors.join(","))
  );

  let maxWords = 0;
  const filtered = candidates
    .filter(el => {
      const ok = isEligible(el, forceRefresh);
      if (ok) {
        maxWords = Math.max(maxWords, getWordCount(el));
      }
      return ok;
    })
    .sort((a, b) => getWordCount(b) - getWordCount(a));

  return { targets: filtered, maxWords };
}

function isEligible(element, forceRefresh) {
  if (!element || element.hasAttribute("aria-hidden")) return false;
  if (!isVisible(element)) return false;

  const words = getWordCount(element);
  if (words < MIN_WORDS) return false;

  // Skip if already processing or highlighted (unless forcing refresh)
  if (element.dataset[DATA_PROCESSING]) return false;
  if (!forceRefresh && element.dataset[DATA_HIGHLIGHTED]) return false;

  const tagName = element.tagName.toLowerCase();
  if (["nav", "header", "footer", "aside"].includes(tagName)) return false;

  return true;
}

function isVisible(element) {
  const rect = element.getBoundingClientRect();
  if (rect.width === 0 || rect.height === 0) return false;

  const style = window.getComputedStyle(element);
  return style && style.visibility !== "hidden" && style.display !== "none";
}

function getWordCount(element) {
  const text = element.textContent || "";
  return text.trim().split(/\s+/).filter(Boolean).length;
}

async function analyzeElement(element) {
  // Skip if already highlighted or processing
  if (element.dataset[DATA_HIGHLIGHTED] || element.dataset[DATA_PROCESSING]) {
    return;
  }
  
  element.dataset[DATA_PROCESSING] = "1";
  highlightManager.clearElement(element);
  activeAnalyses++;
  updateDebugState({ state: "analyzing", activeAnalyses });

  const collected = collectText(element);
  if (collected.wordCount < MIN_WORDS) {
    delete element.dataset[DATA_PROCESSING];
    activeAnalyses = Math.max(0, activeAnalyses - 1);
    updateDebugState({ activeAnalyses });
    if (activeAnalyses === 0) notifier.hide();
    console.info(
      "[semantic-highlight] Skipping element: too short",
      collected.wordCount,
      "words"
    );
    return;
  }
  console.info(
    "[semantic-highlight] Analyzing element with",
    collected.wordCount,
    "words and",
    collected.text.length,
    "chars"
  );

  const payload = {
    type: "analyze",
    text: collected.text,
    intensity: settings.intensity
  };

  console.log("[DEBUG] Sending analyze request:", {
    textLength: collected.text.length,
    wordCount: collected.wordCount,
    intensity: settings.intensity
  });

  chrome.runtime.sendMessage(payload, response => {
    console.log("[DEBUG] Received response from background:", {
      ok: response?.ok,
      hasError: !!response?.error,
      error: response?.error,
      spansCount: response?.spans?.length
    });
    console.log("[DEBUG] chrome.runtime.lastError:", chrome.runtime.lastError);

    delete element.dataset[DATA_PROCESSING];
    activeAnalyses = Math.max(0, activeAnalyses - 1);
    updateDebugState({ activeAnalyses });

    if (!settings.enabled) {
      console.info("[semantic-highlight] aborted: disabled");
      return;
    }

    if (chrome.runtime.lastError || !response?.ok) {
      const lastError = chrome.runtime.lastError;
      const responseError = response?.error;
      console.warn("[DEBUG] Error details:", {
        lastError,
        lastErrorMsg: lastError?.message,
        responseError,
        responseErrorType: typeof responseError
      });
      console.warn(
        "semantic-highlight: analysis failed",
        lastError || responseError
      );
      updateDebugState({
        state: "error",
        lastError: (lastError?.message || (typeof responseError === 'string' ? responseError : String(responseError)) || "Unknown error")
      });
      // Mark element to avoid immediate retry on the same node.
      element.dataset[DATA_HIGHLIGHTED] = "error";
      return;
    }

    const applied = highlightManager.applyHighlights(
      element,
      response.spans,
      collected.mapping
    );
    console.info(
      "[semantic-highlight] Response spans:",
      Array.isArray(response.spans) ? response.spans.length : 0,
      "applied:",
      applied
    );

    if (!applied) {
      element.dataset[DATA_HIGHLIGHTED] = "error";
      return;
    }

    const now = Date.now();
    updateDebugState({
      state: "done",
      lastAppliedSpans: Array.isArray(response.spans) ? response.spans.length : 0,
      lastError: null
    });
  });
}

function collectText(element) {
  const mapping = [];
  const walker = document.createTreeWalker(
    element,
    NodeFilter.SHOW_TEXT,
    {
      acceptNode(node) {
        const parent = node.parentElement;
        if (!parent) return NodeFilter.FILTER_REJECT;
        if (
          parent.closest(
            "script, style, noscript, textarea, svg, code, pre, [aria-hidden='true'], .sr-only"
          )
        ) {
          return NodeFilter.FILTER_REJECT;
        }
        const style = window.getComputedStyle(parent);
        if (style.display === "none" || style.visibility === "hidden") {
          return NodeFilter.FILTER_REJECT;
        }
        return NodeFilter.FILTER_ACCEPT;
      }
    }
  );

  let text = "";
  while (walker.nextNode()) {
    const node = walker.currentNode;
    const content = node.textContent || "";
    const start = text.length;
    text += content;
    mapping.push({ node, start, end: text.length });
  }

  return {
    text,
    mapping,
    wordCount: text.trim().split(/\s+/).filter(Boolean).length
  };
}

class HighlightManager {
  constructor() {
    this.styleInjected = false;
    this.tagBanner = null;
    this.allKeyphrases = new Set();
  }

  ensureStyle() {
    if (this.styleInjected) return;

    const style = document.createElement("style");
    style.textContent = `
      .sem-highlight-banner {
        position: fixed;
        bottom: 12px;
        left: 12px;
        max-width: 280px;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(12px);
        color: #333;
        padding: 8px 10px;
        border-radius: 8px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        z-index: 2147483646;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        display: flex;
        flex-direction: column;
        gap: 6px;
        font-size: 12px;
        border: 1px solid rgba(0,0,0,0.04);
      }
      .sem-highlight-banner-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .sem-highlight-banner-title {
        font-weight: 600;
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
      }
      .sem-highlight-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 4px;
      }
      .sem-highlight-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.12);
        padding: 3px 8px;
        border-radius: 10px;
        font-size: 11px;
        font-weight: 500;
        color: #000;
        border: 1px solid rgba(102, 126, 234, 0.2);
        cursor: pointer;
        transition: all 0.15s ease;
      }
      .sem-highlight-tag:hover {
        background: rgba(102, 126, 234, 0.2);
        border-color: rgba(102, 126, 234, 0.3);
        transform: translateY(-1px);
      }
      .sem-highlight-close {
        background: transparent;
        border: none;
        color: #999;
        width: 16px;
        height: 16px;
        cursor: pointer;
        font-size: 14px;
        line-height: 1;
        padding: 0;
        transition: color 0.2s ease;
      }
      .sem-highlight-close:hover {
        color: #333;
      }
    `;
    document.head.appendChild(style);
    this.styleInjected = true;
  }

  clearElement(element) {
    element.removeAttribute(ATTR_HIGHLIGHTED);
    delete element.dataset[DATA_HIGHLIGHTED];
  }

  clearAll() {
    this.allKeyphrases.clear();
    if (this.tagBanner) {
      this.tagBanner.remove();
      this.tagBanner = null;
    }
    
    document
      .querySelectorAll(`[${ATTR_HIGHLIGHTED}]`)
      .forEach(el => {
        el.removeAttribute(ATTR_HIGHLIGHTED);
        delete el.dataset[DATA_HIGHLIGHTED];
      });
  }

  createBanner() {
    if (this.tagBanner) return;
    
    this.ensureStyle();
    this.tagBanner = document.createElement("div");
    this.tagBanner.className = "sem-highlight-banner";
    
    const header = document.createElement("div");
    header.className = "sem-highlight-banner-header";
    
    const title = document.createElement("span");
    title.className = "sem-highlight-banner-title";
    title.textContent = "Key Topics";
    header.appendChild(title);
    
    const closeBtn = document.createElement("button");
    closeBtn.className = "sem-highlight-close";
    closeBtn.textContent = "×";
    closeBtn.onclick = () => {
      settings.enabled = false;
      chrome.storage.sync.set({ enabled: false });
      this.clearAll();
    };
    header.appendChild(closeBtn);
    
    this.tagBanner.appendChild(header);
    
    const tagsContainer = document.createElement("div");
    tagsContainer.className = "sem-highlight-tags";
    this.tagBanner.appendChild(tagsContainer);
    
    document.body.insertBefore(this.tagBanner, document.body.firstChild);
  }

  updateBanner() {
    if (!this.tagBanner || this.allKeyphrases.size === 0) return;
    
    const tagsContainer = this.tagBanner.querySelector(".sem-highlight-tags");
    if (!tagsContainer) return;
    
    // Clear existing tags
    tagsContainer.innerHTML = "";
    
    // Sort keyphrases alphabetically
    const keyphrases = Array.from(this.allKeyphrases).sort();
    
    keyphrases.forEach(phrase => {
      const tag = document.createElement("span");
      tag.className = "sem-highlight-tag";
      tag.textContent = phrase;
      tag.onclick = (e) => {
        e.preventDefault();
        const searchUrl = `https://www.google.com/search?q=${encodeURIComponent(phrase)}`;
        window.open(searchUrl, '_blank');
      };
      tagsContainer.appendChild(tag);
    });
  }

  applyHighlights(element, spans, mapping) {
    if (!Array.isArray(spans) || !spans.length) {
      return false;
    }

    this.ensureStyle();
    
    // Extract keyphrase texts and add to collection
    spans.forEach(span => {
      if (span.text && span.text.trim()) {
        this.allKeyphrases.add(span.text.trim());
      }
    });
    
    // Create or update banner
    this.createBanner();
    this.updateBanner();

    element.setAttribute(ATTR_HIGHLIGHTED, "1");
    element.dataset[DATA_HIGHLIGHTED] = "1";
    
    return true;
  }
}

function locatePosition(mapping, targetOffset) {
  for (let i = 0; i < mapping.length; i++) {
    const entry = mapping[i];
    if (targetOffset < entry.start) {
      break;
    }
    if (targetOffset <= entry.end) {
      return {
        node: entry.node,
        offset: targetOffset - entry.start
      };
    }
  }
  return null;
}

function debounce(callback, delay) {
  let timeout = null;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => callback(...args), delay);
  };
}

function updateDebugState(partial) {
  Object.assign(debugState, partial, { lastUpdated: Date.now() });
  chrome.storage.local.set({ semHighlightDebug: debugState });
}

function warmupModel() {
  console.log("[DEBUG] Sending warmup request");
  chrome.runtime.sendMessage({ type: "warmup" }, response => {
    console.log("[DEBUG] Warmup response received:", {
      ok: response?.ok,
      error: response?.error,
      lastError: chrome.runtime.lastError
    });
    if (chrome.runtime.lastError || !response?.ok) {
      const errorMsg = chrome.runtime.lastError?.message || response?.error || "Warmup failed";
      console.log("[DEBUG] Warmup failed:", errorMsg);
      updateDebugState({
        state: "error",
        lastError: errorMsg
      });
    } else {
      console.log("[DEBUG] Warmup successful");
    }
  });
}

const notifier = new StatusNotifier();
let activeAnalyses = 0;
let lastSuccessToastAt = 0;
const debugState = {
  state: "idle",
  activeAnalyses: 0,
  lastAppliedSpans: 0,
  lastError: null,
  lastUpdated: Date.now(),
  lastScanCandidates: 0,
  lastScanSelected: 0,
  lastScanMaxWords: 0
};
const highlightManager = new HighlightManager();
let settings = { ...DEFAULT_SETTINGS };
let observer = null;

initialize();
