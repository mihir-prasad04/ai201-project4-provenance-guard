import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[],
    storage_uri="memory://"
)

AUDIT_LOG = {}

def calculate_stylometric_score(text: str) -> float:
    words = [w.lower().strip(".,!?;:()\"'") for w in text.split() if w]
    if not words:
        return 0.5
    unique_words = set(words)
    type_token_ratio = len(unique_words) / len(words)
    ai_probability = 1.0 - type_token_ratio
    return max(0.0, min(1.0, ai_probability))

def get_llm_judge_score(text: str) -> float:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return 0.5
    try:
        client = Groq(api_key=api_key)
        system_prompt = "You are an expert AI content detection judge. Analyze the user's text for generative AI signatures. Output exactly in this format with no extra text: Score: [A float value between 0.0 for human and 1.0 for AI]"
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.0
        )
        output = response.choices[0].message.content.strip()
        for line in output.split("\n"):
            if line.lower().startswith("score:"):
                score_val = float(line.split(":")[1].strip())
                return max(0.0, min(1.0, score_val))
    except Exception as e:
        print(f"Error calling Groq API: {e}")
    return 0.5

@app.route("/submit", methods=["POST"])
@limiter.limit("5 per minute")
def submit_content():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    creator_id = data.get("creator_id", "").strip()
    
    if not text or not creator_id:
        return jsonify({"error": "Missing parameters"}), 400
        
    content_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat() + "Z"
    
    heuristic_score = calculate_stylometric_score(text)
    llm_score = get_llm_judge_score(text)
    combined_score = (0.65 * llm_score) + (0.35 * heuristic_score)
    
    if combined_score <= 0.35:
        attribution = "likely_human"
        label = "Verified Human Content."
    elif combined_score >= 0.75:
        attribution = "likely_ai"
        label = "AI-Generated Content."
    else:
        attribution = "uncertain"
        label = "Uncertain Attribution."
        
    log_entry = {
        "content_id": content_id,
        "creator_id": creator_id,
        "timestamp": timestamp,
        "attribution": attribution,
        "confidence": round(combined_score, 2),
        "heuristic_score": round(heuristic_score, 2),
        "llm_score": round(llm_score, 2),
        "status": "classified",
        "appeal_reasoning": None
    }
    AUDIT_LOG[content_id] = log_entry
    
    return jsonify({
        "content_id": content_id,
        "attribution": attribution,
        "confidence": log_entry["confidence"],
        "label": label
    }), 200

@app.route("/appeal", methods=["POST"])
def appeal_classification():
    data = request.get_json() or {}
    content_id = data.get("content_id", "").strip()
    reasoning = data.get("creator_reasoning", "").strip()
    
    if not content_id or not reasoning:
        return jsonify({"error": "Missing parameters"}), 400
        
    if content_id not in AUDIT_LOG:
        return jsonify({"error": "Content record not found"}), 404
        
    AUDIT_LOG[content_id]["status"] = "under_review"
    AUDIT_LOG[content_id]["appeal_reasoning"] = reasoning
    
    return jsonify({
        "message": "Appeal successfully received.",
        "content_id": content_id,
        "status": "under_review"
    }), 200

@app.route("/log", methods=["GET"])
def view_audit_logs():
    return jsonify({"entries": list(AUDIT_LOG.values())}), 200

if __name__ == "__main__":
    app.run(port=5000, debug=True)