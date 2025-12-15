// Background service worker - communicates with Python backend for text analysis

const BACKEND_URL = "http://127.0.0.1:5000";
const DEFAULT_SETTINGS = {
  enabled: true,
  intensity: 0.35
};

let backendAvailable = false;

// Queue to manage concurrent requests
const analysisQueue = [];
let isProcessingAnalysis = false;

async function checkBackendHealth() {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 2000);

    const response = await fetch(`${BACKEND_URL}/health`, {
      signal: controller.signal
    });
    clearTimeout(timeoutId);

    const data = await response.json();
    backendAvailable = data.status === "ok" && data.model_loaded;
    console.log("[DEBUG] Backend health check:", {
      available: backendAvailable,
      status: data.status,
      modelLoaded: data.model_loaded
    });
    return backendAvailable;
  } catch (error) {
    backendAvailable = false;
    console.error("[DEBUG] Backend health check failed:", error.message);
    return false;
  }
}

async function queueAnalysis(message) {
  return new Promise((resolve) => {
    analysisQueue.push({ message, resolve });
    processAnalysisQueue();
  });
}

async function processAnalysisQueue() {
  if (isProcessingAnalysis || analysisQueue.length === 0) {
    return;
  }

  isProcessingAnalysis = true;
  const { message, resolve } = analysisQueue.shift();
  console.log("[DEBUG] Processing queued analysis, queue remaining:", analysisQueue.length);

  try {
    const result = await handleAnalyzeRequest(message);
    resolve({ ok: true, spans: result });
  } catch (error) {
    const errorMsg = typeof error === "string" ? error : (error?.message || String(error) || "Analysis failed");
    console.error("[DEBUG] Queued analysis failed:", errorMsg);
    resolve({ ok: false, error: errorMsg });
  } finally {
    isProcessingAnalysis = false;
    if (analysisQueue.length > 0) {
      processAnalysisQueue();
    }
  }
}

async function handleAnalyzeRequest(message) {
  const text = typeof message.text === "string" ? message.text : "";
  if (!text.trim()) {
    console.info("semantic-highlight: empty text, skipping");
    return [];
  }

  const intensity = Math.min(Math.max(Number(message.intensity) || 0.35, 0.05), 1);

  // Check backend availability
  if (!backendAvailable) {
    const available = await checkBackendHealth();
    if (!available) {
      throw new Error("Backend server not available. Make sure 'python backend.py' is running on localhost:5000");
    }
  }

  console.log("[DEBUG] Sending analysis request to backend");
  console.info("semantic-highlight: analyzing", text.length, "chars");

  try {
    const response = await fetch(`${BACKEND_URL}/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        text: text,
        intensity: intensity
      })
    });

    if (!response.ok) {
      throw new Error(`Backend error: ${response.status} ${response.statusText}`);
    }

    const data = await response.json();

    if (!data.ok) {
      throw new Error(data.error || "Backend analysis failed");
    }

    console.info("semantic-highlight: found", data.spans?.length || 0, "spans");
    return data.spans || [];

  } catch (error) {
    console.error("[DEBUG] Analysis request failed:", error.message);
    throw error;
  }
}

// Set up storage listener for settings changes
chrome.storage.onChanged.addListener((changes) => {
  for (const key of Object.keys(changes)) {
    if (key === "enabled") {
      console.log("[DEBUG] Extension enabled:", changes[key].newValue);
    }
    if (key === "intensity") {
      console.log("[DEBUG] Intensity changed to:", changes[key].newValue);
    }
  }
});

chrome.runtime.onInstalled.addListener(() => {
  chrome.storage.sync.get(DEFAULT_SETTINGS, (current) => {
    const updates = {};
    if (typeof current.enabled !== "boolean") {
      updates.enabled = DEFAULT_SETTINGS.enabled;
    }
    if (typeof current.intensity !== "number") {
      updates.intensity = DEFAULT_SETTINGS.intensity;
    }
    if (Object.keys(updates).length) {
      chrome.storage.sync.set(updates);
    }
  });

  // Check backend health on install
  checkBackendHealth();
});

// Message handler for content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log("[DEBUG] Message received:", {
    type: message?.type,
    textLength: message?.text?.length,
    intensity: message?.intensity
  });

  if (message?.type === "analyze") {
    console.log("[DEBUG] Handling analyze request, queue length:", analysisQueue.length);
    queueAnalysis(message)
      .then((response) => {
        console.log("[DEBUG] Queued analyze completed");
        sendResponse(response);
      })
      .catch((error) => {
        const errorMsg = String(error);
        console.error("[DEBUG] Queued analyze failed:", errorMsg);
        sendResponse({ ok: false, error: errorMsg });
      });
    return true; // Will respond asynchronously
  }

  if (message?.type === "warmup") {
    console.log("[DEBUG] Handling warmup request");
    checkBackendHealth()
      .then((available) => {
        if (available) {
          console.log("[DEBUG] Warmup successful - backend is ready");
          sendResponse({ ok: true });
        } else {
          sendResponse({
            ok: false,
            error: "Backend not available. Run 'python backend.py' on localhost:5000"
          });
        }
      })
      .catch((error) => {
        console.error("[DEBUG] Warmup failed:", error.message);
        sendResponse({ ok: false, error: error.message });
      });
    return true; // Will respond asynchronously
  }

  console.log("[DEBUG] Unknown message type:", message?.type);
  return false;
});

// Check backend health periodically
setInterval(() => {
  checkBackendHealth();
}, 30000); // Every 30 seconds
