const DEFAULT_SETTINGS = {
  enabled: true,
  intensity: 0.35
};

const toggle = document.getElementById("toggle-enabled");
const intensitySlider = document.getElementById("intensity");
const intensityLabel = document.getElementById("intensity-label");
const debugStatus = document.getElementById("debug-status");
const debugActive = document.getElementById("debug-active");
const debugSpans = document.getElementById("debug-spans");
const debugError = document.getElementById("debug-error");
const debugUpdated = document.getElementById("debug-updated");

initialize();

function initialize() {
  chrome.storage.sync.get(DEFAULT_SETTINGS, stored => {
    const settings = { ...DEFAULT_SETTINGS, ...stored };
    toggle.checked = !!settings.enabled;
    intensitySlider.value = settings.intensity;
    renderIntensity(settings.intensity);
  });

  toggle.addEventListener("change", persistSettings);
  intensitySlider.addEventListener("input", () =>
    renderIntensity(intensitySlider.value)
  );
  intensitySlider.addEventListener("change", persistSettings);

  // Load initial debug info.
  chrome.storage.local.get("semHighlightDebug", data => {
    renderDebug(data.semHighlightDebug);
  });

  chrome.storage.onChanged.addListener((changes, area) => {
    if (area !== "local" || !changes.semHighlightDebug) return;
    renderDebug(changes.semHighlightDebug.newValue);
  });
}

function renderIntensity(value) {
  const percent = Math.round(Number(value) * 100);
  intensityLabel.textContent = `${percent}% of model-picked spans`;
}

function persistSettings() {
  const payload = {
    enabled: toggle.checked,
    intensity: Number(intensitySlider.value)
  };
  chrome.storage.sync.set(payload);
}

function renderDebug(debug) {
  const state = debug || {};
  debugStatus.textContent = state.state || "idle";
  debugActive.textContent = state.activeAnalyses ?? 0;
  debugSpans.textContent = state.lastAppliedSpans ?? 0;
  debugError.textContent = state.lastError || "None";
  debugUpdated.textContent = state.lastUpdated
    ? new Date(state.lastUpdated).toLocaleTimeString()
    : "–";
  const scanInfo = [];
  if (state.lastScanCandidates != null)
    scanInfo.push(`${state.lastScanCandidates} candidates`);
  if (state.lastScanSelected != null)
    scanInfo.push(`${state.lastScanSelected} selected`);
  if (state.lastScanMaxWords != null)
    scanInfo.push(`max words ${state.lastScanMaxWords}`);
  debugStatus.title = scanInfo.join(" · ");
}
