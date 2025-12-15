#!/usr/bin/env python3
"""
Simple Semantic Highlight Backend
Uses ONNX Runtime with proper RoBERTa tokenization
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
import onnxruntime as rt
from transformers import AutoTokenizer

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global ONNX session and tokenizer
session = None
tokenizer = None
MODEL_PATH = None


def load_model():
    """Load the ONNX model and tokenizer."""
    global session, tokenizer, MODEL_PATH

    try:
        # Find the model directory
        extension_dir = Path(__file__).parent / "browser-extension"
        model_dir = extension_dir / "model"

        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found at {model_dir}")

        logger.info(f"Loading model from {model_dir}")

        # Load tokenizer from model directory
        logger.info("Loading RoBERTa tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        logger.info(f"Tokenizer loaded (vocab size: {len(tokenizer)})")

        # Load ONNX model
        onnx_model_path = model_dir / "onnx" / "model.onnx"
        if not onnx_model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {onnx_model_path}")

        logger.info("Loading ONNX model...")
        session = rt.InferenceSession(str(onnx_model_path), providers=['CPUExecutionProvider'])
        logger.info("ONNX model loaded successfully")

        MODEL_PATH = str(model_dir)
        logger.info("Backend ready for inference")

    except Exception as e:
        logger.error(f"Failed to load model: {e}", exc_info=True)
        raise


@app.route("/health", methods=["GET"])
def health():
    """Health check."""
    return jsonify({
        "status": "ok",
        "model_loaded": session is not None,
        "tokenizer_loaded": tokenizer is not None,
        "vocab_size": len(tokenizer) if tokenizer else 0
    })


@app.route("/analyze", methods=["POST"])
def analyze():
    """Analyze text for key phrases."""
    if session is None or tokenizer is None:
        return jsonify({"ok": False, "error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        intensity = min(max(data.get("intensity", 0.35), 0.01), 1.0)

        if not text:
            return jsonify({"ok": True, "spans": []})

        logger.info(f"Analyzing {len(text)} chars")

        # Tokenize with transformers (proper BPE tokenization)
        encoding = tokenizer(
            text,
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=512,
            return_offsets_mapping=True
        )

        input_ids = encoding["input_ids"].astype(np.int64)
        attention_mask = encoding["attention_mask"].astype(np.int64)
        offset_mapping = encoding["offset_mapping"][0]  # Character positions for each token

        # Run inference
        outputs = session.run(None, {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        })

        logits = outputs[0]  # Shape: (batch_size, seq_length, num_labels)
        predictions = np.argmax(logits[0], axis=1)  # Get best label for each token

        # Load label mapping
        config_path = Path(MODEL_PATH) / "config.json"
        with open(config_path) as f:
            config = json.load(f)

        id2label = config.get("id2label", {})
        id2label = {int(k) if isinstance(k, str) else k: v for k, v in id2label.items()}

        # Extract spans for KEY entities
        result_spans = []
        current_span = None
        current_score = 0.0

        for i in range(len(predictions)):
            label_id = int(predictions[i])
            label = id2label.get(label_id, "O")
            score = float(np.max(logits[0, i]))

            # Check if this is a KEY entity (B-KEY or I-KEY)
            is_key = "KEY" in label

            if is_key:
                if current_span is None:
                    # Start new span
                    current_span = {"start_token": i, "end_token": i, "score": score}
                    current_score = score
                else:
                    # Continue current span
                    current_span["end_token"] = i
                    current_score = max(current_score, score)
            else:
                if current_span is not None:
                    # End current span - convert to character positions
                    start_token = current_span["start_token"]
                    end_token = current_span["end_token"]

                    # Get character positions from offset mapping
                    if start_token < len(offset_mapping) and end_token < len(offset_mapping):
                        start_char = int(offset_mapping[start_token][0])
                        end_char = int(offset_mapping[end_token][1])

                        if start_char < end_char:
                            span_text = text[start_char:end_char]
                            result_spans.append({
                                "start": start_char,
                                "end": end_char,
                                "text": span_text.title(),
                                "score": current_score
                            })

                    current_span = None
                    current_score = 0.0

        # Handle last span if it extends to end
        if current_span is not None:
            start_token = current_span["start_token"]
            end_token = current_span["end_token"]

            if start_token < len(offset_mapping) and end_token < len(offset_mapping):
                start_char = int(offset_mapping[start_token][0])
                end_char = int(offset_mapping[end_token][1])

                if start_char < end_char:
                    span_text = text[start_char:end_char]
                    result_spans.append({
                        "start": start_char,
                        "end": end_char,
                        "text": span_text.title(),
                        "score": current_score
                    })

        # Apply intensity filtering - keep top spans by score
        if result_spans:
            result_spans.sort(key=lambda x: x["score"], reverse=True)
            keep_count = max(1, int(len(result_spans) * intensity))
            result_spans = result_spans[:keep_count]

        # Filter out spans shorter than 5 characters
        result_spans = [span for span in result_spans if len(span['text']) >= 5]

        logger.info(f"Found {len(result_spans)} spans")
        
        # Print highlighted text
        for i, span in enumerate(result_spans, 1):
            logger.info(f"HIGHLIGHTED TEXT {i}: '{span['text']}' (score: {span['score']:.4f}, pos: {span['start']}-{span['end']})")

        return jsonify({"ok": True, "spans": result_spans})

    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        return jsonify({"ok": False, "error": str(e)}), 500


if __name__ == "__main__":
    try:
        logger.info("Starting Semantic Highlight Backend")
        load_model()
        logger.info("Server starting on http://127.0.0.1:5000")
        app.run(host="127.0.0.1", port=5000, debug=True, threaded=False)
    except KeyboardInterrupt:
        logger.info("Server stopped")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)
