# Semantic Highlighter Chrome Extension

This extension loads the trained RoBERTa keyphrase tagger that lives in this repository and highlights the most relevant parts of long pages (200+ words) such as articles and blog posts.

## Contents

- `manifest.json` — manifest v3 definition.
- `background.js` — service worker that runs ONNX inference with the exported RoBERTa model.
- `contentScript.js` — scans pages, requests predictions, and applies highlights.
- `popup.html|js|css` — toggle/slider UI for enabling the feature and adjusting how much is highlighted.
- `model/` — ONNX-exported weights plus tokenizer/config files from `roberta_openkp_model`.
- `lib/` — vendored `@xenova/transformers` runtime and ONNX Runtime wasm binaries.

## Loading the extension

1. Open `chrome://extensions` and toggle on **Developer mode**.
2. Click **Load unpacked** and select the `browser-extension/` directory.
3. Pin the "Semantic Highlighter" action if you want quick access to the toggle/slider.
4. Visit a long article; relevant phrases should be highlighted automatically when the extension is enabled.

## Updating the model (optional)

If you retrain the RoBERTa tagger, export new ONNX weights into `browser-extension/model` with:

```bash
python -m transformers.onnx --model=roberta_openkp_model --feature=token-classification --framework=pt --opset 18 browser-extension/model
```

Copy tokenizer/config files from the new checkpoint into `browser-extension/model/` (overwriting the existing ones). The extension will load the updated weights next time it runs.
